from __future__ import annotations

"""
Minimal Monte Carlo cashflow engine for residential project finance (214-FZ style).

Design goals
------------
1. State-by-state quarterly simulation.
2. Easy unit testing of each block separately.
3. Clear interfaces for future fitted price/sales models.
4. Metrics aligned with the uploaded cashflow workbook/description:
   - escrow balance and coverage
   - debt outstanding
   - interest / reserve fee / debt service
   - CFADS / DS / DSCR / ISCR
   - cumulative effective rate
   - collateral coverage
   - peak debt / ending debt / min DSCR
5. Current MVP intentionally avoids cross-factor correlations.

Important simplifications in this MVP
-------------------------------------
- Fitted models are not implemented; only stubs / simple deterministic models are provided.
- Hedonic price path does NOT affect sales model yet.
- Commercial / parking / storage lines are omitted for now; the engine focuses on the core
  apartment flow, while the structure is ready for extension.
- No stochastic cost noise yet.
- Taxes are kept simple and close to the workbook logic:
    * profit tax is recognized at RVZ quarter on accumulated pre-tax profit,
      if configured to do so.
    * VAT / property tax placeholders exist, but default to zero.

Sign convention
---------------
- Positive values = inflows to project / positive balance items.
- Costs and debt service components are stored as positive line items,
  but aggregate expense / debt-service measures are also exposed explicitly.
"""

import argparse
import csv
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence
import math
import random


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuarterId:
    year: int
    quarter: int

    def next(self) -> "QuarterId":
        if self.quarter == 4:
            return QuarterId(self.year + 1, 1)
        return QuarterId(self.year, self.quarter + 1)

    def label(self) -> str:
        return f"{self.quarter}Q{self.year}"


# ---------------------------------------------------------------------------
# Core inputs and state containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    start_year: int
    start_quarter: int
    horizon_quarters: int
    rns_quarter_index: int
    rvz_quarter_index: int

    # Inventory / sales units
    sellable_area_sqm: float
    avg_unit_area_sqm: float
    initial_remaining_lots: int

    # Financing assumptions
    key_rate_annual: float
    full_rate_spread_before_rvz: float
    full_rate_spread_after_rvz: float
    privileged_rate_annual: float
    reserve_fee_annual: float
    line_usage_fee_annual: float = 0.0
    ltc: float = 0.90

    # Cost assumptions
    land_cost_total: float = 0.0
    smr_cost_total: float = 0.0
    design_cost_total: float = 0.0
    marketing_cost_ratio: float = 0.0

    # Tax assumptions
    property_tax_rate: float = 0.0
    profit_tax_rate: float = 0.0
    vat_rate: float = 0.0
    recognize_profit_tax_at_rvz: bool = True

    # Pricing / collateral
    use_client_prices: bool = False
    initial_price_sqm: float = 0.0
    collateral_discount: float = 0.30
    collateral_price_sqm: float = 0.0

    # Equity / line sizing
    debt_limit: Optional[float] = None
    initial_equity_contribution: Optional[float] = None

    def __post_init__(self) -> None:
        if self.horizon_quarters <= 0:
            raise ValueError("horizon_quarters must be positive")
        if self.initial_remaining_lots < 0:
            raise ValueError("initial_remaining_lots must be non-negative")
        if self.avg_unit_area_sqm <= 0:
            raise ValueError("avg_unit_area_sqm must be positive")
        if not (1 <= self.start_quarter <= 4):
            raise ValueError("start_quarter must be in 1..4")
        if not (0 <= self.rns_quarter_index < self.horizon_quarters):
            raise ValueError("rns_quarter_index out of range")
        if not (0 <= self.rvz_quarter_index < self.horizon_quarters):
            raise ValueError("rvz_quarter_index out of range")
        if self.rvz_quarter_index < self.rns_quarter_index:
            raise ValueError("rvz_quarter_index must be >= rns_quarter_index")

    @property
    def total_project_cost(self) -> float:
        return self.land_cost_total + self.smr_cost_total + self.design_cost_total

    @property
    def resolved_debt_limit(self) -> float:
        return self.debt_limit if self.debt_limit is not None else self.total_project_cost * self.ltc

    @property
    def resolved_initial_equity(self) -> float:
        if self.initial_equity_contribution is not None:
            return self.initial_equity_contribution
        return max(self.total_project_cost - self.resolved_debt_limit, 0.0)


@dataclass(frozen=True)
class CostSchedule:
    land_by_quarter: Sequence[float]
    smr_by_quarter: Sequence[float]
    design_by_quarter: Sequence[float] = field(default_factory=list)
    other_opex_by_quarter: Sequence[float] = field(default_factory=list)
    post_completion_cost_by_quarter: Sequence[float] = field(default_factory=list)
    property_tax_by_quarter: Sequence[float] = field(default_factory=list)
    vat_by_quarter: Sequence[float] = field(default_factory=list)

    def validate(self, horizon: int) -> None:
        for name, seq in [
            ("land_by_quarter", self.land_by_quarter),
            ("smr_by_quarter", self.smr_by_quarter),
            ("design_by_quarter", self.design_by_quarter),
            ("other_opex_by_quarter", self.other_opex_by_quarter),
            ("post_completion_cost_by_quarter", self.post_completion_cost_by_quarter),
            ("property_tax_by_quarter", self.property_tax_by_quarter),
            ("vat_by_quarter", self.vat_by_quarter),
        ]:
            if seq and len(seq) != horizon:
                raise ValueError(f"{name} length must equal horizon ({horizon})")

    def value(self, seq: Sequence[float], idx: int) -> float:
        return float(seq[idx]) if seq else 0.0

    def land(self, idx: int) -> float:
        return self.value(self.land_by_quarter, idx)

    def smr(self, idx: int) -> float:
        return self.value(self.smr_by_quarter, idx)

    def design(self, idx: int) -> float:
        return self.value(self.design_by_quarter, idx)

    def other_opex(self, idx: int) -> float:
        return self.value(self.other_opex_by_quarter, idx)

    def post_completion(self, idx: int) -> float:
        return self.value(self.post_completion_cost_by_quarter, idx)

    def property_tax(self, idx: int) -> float:
        return self.value(self.property_tax_by_quarter, idx)

    def vat(self, idx: int) -> float:
        return self.value(self.vat_by_quarter, idx)


@dataclass(frozen=True)
class GlobalScenario:
    name: str = "base"
    key_rate_shift_annual: float = 0.0
    price_level_multiplier: float = 1.0
    sales_level_multiplier: float = 1.0
    project_price_premium: float = 0.0
    rvz_delay_quarters: int = 0


@dataclass(frozen=True)
class LocalScenarioPath:
    price_shocks: Sequence[float] = field(default_factory=list)
    sales_shocks: Sequence[float] = field(default_factory=list)

    def validate(self, horizon: int) -> None:
        for name, seq in [("price_shocks", self.price_shocks), ("sales_shocks", self.sales_shocks)]:
            if seq and len(seq) != horizon:
                raise ValueError(f"{name} length must equal horizon ({horizon})")

    def price_shock(self, idx: int) -> float:
        return float(self.price_shocks[idx]) if self.price_shocks else 0.0

    def sales_shock(self, idx: int) -> float:
        return float(self.sales_shocks[idx]) if self.sales_shocks else 0.0


@dataclass(frozen=True)
class Scenario:
    global_state: GlobalScenario = field(default_factory=GlobalScenario)
    local_path: LocalScenarioPath = field(default_factory=LocalScenarioPath)

    def validate(self, horizon: int) -> None:
        self.local_path.validate(horizon)


@dataclass(frozen=True)
class QuarterContext:
    index: int
    quarter_id: QuarterId
    is_pre_rns: bool
    is_post_rvz: bool
    is_rvz_quarter: bool
    rvz_effective_index: int


@dataclass
class QuarterState:
    index: int
    quarter_id: QuarterId
    remaining_lots_start: int
    remaining_lots_end: int

    # Sales / pricing
    price_sqm: float
    sold_lots: int
    sold_area_sqm: float
    revenue: float

    # Escrow
    escrow_inflow: float
    escrow_release: float
    escrow_balance_end: float
    project_cash_inflow: float
    equity_inflow: float
    project_cash_balance_end: float
    escrow_coverage_ratio: float

    # Cost items
    land_cost: float
    smr_cost: float
    design_cost: float
    marketing_cost: float
    other_opex: float
    post_completion_cost: float
    property_tax: float
    vat: float
    profit_tax: float
    total_costs_excl_financing: float

    # Financing
    debt_draw: float
    debt_repayment: float
    debt_outstanding_end: float
    debt_limit: float
    unused_debt_limit: float
    full_rate_annual: float
    interest_payment: float
    line_usage_fee_payment: float
    reserve_fee_payment: float
    total_debt_service_cost: float
    effective_rate_cumulative_annual: float

    # Cashflow / metrics
    net_cash_flow: float
    cfads: float
    debt_service_for_ratio: float
    dscr: float
    iscr: float

    # Diagnostics
    notes: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PathSummary:
    project_name: str
    scenario_name: str
    total_revenue: float
    total_costs_excl_financing: float
    total_interest_and_fees: float
    total_debt_service_for_ratio: float
    total_cfads: float
    total_net_cash_flow: float
    ending_escrow_balance: float
    ending_project_cash_balance: float
    peak_debt: float
    ending_debt: float
    min_dscr: float
    min_iscr: float
    max_liquidity_shortfall: float
    final_effective_rate_annual: float
    collateral_value_net: float
    collateral_coverage_ratio: float
    repaid_at_rvz: bool
    debt_fully_repaid: bool


@dataclass(frozen=True)
class PathResult:
    states: List[QuarterState]
    summary: PathSummary


# ---------------------------------------------------------------------------
# Model interfaces and simple placeholders
# ---------------------------------------------------------------------------


class PriceModel(ABC):
    @abstractmethod
    def predict_price_sqm(
        self,
        *,
        ctx: QuarterContext,
        config: ProjectConfig,
        scenario: Scenario,
        previous_state: Optional[QuarterState],
    ) -> float:
        raise NotImplementedError


class SalesModel(ABC):
    @abstractmethod
    def predict_sold_lots(
        self,
        *,
        ctx: QuarterContext,
        config: ProjectConfig,
        scenario: Scenario,
        remaining_lots: int,
        previous_state: Optional[QuarterState],
    ) -> int:
        raise NotImplementedError


@dataclass
class StubPriceModel(PriceModel):
    """
    Placeholder price model.

    The model is intentionally simple:
    - starts from initial_price_sqm
    - applies global price level multiplier
    - applies project premium (as an additive log-like approximation via simple multiplier)
    - applies local per-quarter multiplicative shocks
    """

    def predict_price_sqm(
        self,
        *,
        ctx: QuarterContext,
        config: ProjectConfig,
        scenario: Scenario,
        previous_state: Optional[QuarterState],
    ) -> float:
        base = config.initial_price_sqm
        base *= scenario.global_state.price_level_multiplier
        base *= (1.0 + scenario.global_state.project_price_premium)
        base *= (1.0 + scenario.local_path.price_shock(ctx.index))
        return max(base, 0.0)


@dataclass
class StubSalesModel(SalesModel):
    """
    Placeholder sales model.

    `quarterly_sales_share` contains expected sales shares of total project lots
    for each quarter. This matches the workbook layout where the sales profile is
    entered as a quarterly split of the full sellable pool rather than a share of
    the then-remaining inventory.
    """

    quarterly_sales_share: Sequence[float]

    def predict_sold_lots(
        self,
        *,
        ctx: QuarterContext,
        config: ProjectConfig,
        scenario: Scenario,
        remaining_lots: int,
        previous_state: Optional[QuarterState],
    ) -> int:
        if remaining_lots <= 0:
            return 0
        if ctx.is_pre_rns:
            return 0
        if ctx.index >= len(self.quarterly_sales_share):
            return 0
        share = self.quarterly_sales_share[ctx.index]
        share *= scenario.global_state.sales_level_multiplier
        share *= (1.0 + scenario.local_path.sales_shock(ctx.index))
        share = max(share, 0.0)
        expected = config.initial_remaining_lots * share
        sold = int(round(expected))
        return max(0, min(remaining_lots, sold))


# ---------------------------------------------------------------------------
# Fitted NB sales model (pre-trained coefficients, no training at runtime)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureScaler:
    """
    Per-feature winsorization + z-score standardization parameters from training.

    Must match the statistics computed over the training split in the notebook
    (prepare_design_sim). Without correct scalers predictions will be wrong.
    Export them from the notebook and pass alongside the weight CSV.
    """

    clip_lo: float
    clip_hi: float
    mean: float
    std: float

    def transform(self, x: float) -> float:
        x = max(self.clip_lo, min(self.clip_hi, float(x)))
        s = self.std if self.std > 1e-8 else 1.0
        return max(-6.0, min(6.0, (x - self.mean) / s))


# Numeric features that are standardized at training time.
# Categorical (class_group_*, quarter_num_*) are one-hot and not scaled.
_NB_NUMERIC_FEATURES = frozenset(
    {
        "project_age_q",
        "quarters_to_delivery",
        "remaining_inventory_share",
        "log_avg_price_sqm_q",
        "log_total_lots",
        "log_price_to_market_year_ratio",
        "log_remaining_lots_feat",
    }
)


@dataclass
class FittedNBFeatureSalesModel(SalesModel):
    """
    Negative Binomial GLM with log(remaining_lots) as a feature.

    Loaded from pre-trained coefficients exported by the notebook
    (model_weights_summary*.csv, row model_name='NB_feature_remaining_lots').
    No training happens at runtime — only inference.

    Feature vector mirrors prepare_design_sim(remaining_lots_mode="feature"):
        const
        project_age_q          quarters since project start
        quarters_to_delivery   quarters until rvz (clipped −8..20)
        remaining_inventory_share  remaining_lots / initial_lots
        log_avg_price_sqm_q    log of current price per sqm
        log_total_lots         log of initial_remaining_lots
        log_price_to_market_year_ratio  log(project_price / market_price)
        class_group_<X>        one-hot dummies (reference class is omitted)
        quarter_num_<2|3|4>    seasonality dummies (Q1 is reference)
        log_remaining_lots_feat  log of remaining_lots at start of quarter

    Parameters
    ----------
    coef : coefficients in the same column order as feature_names (intercept first).
    feature_names : column names matching the coef vector.
    alpha : NB dispersion parameter (alpha in NB2 / statsmodels convention).
    scalers : per-feature scaler; numeric features must be present.
              Pass None only for debugging (predictions will be inaccurate).
    class_group : class label for this project, e.g. "Комфорт".
                  Must match the suffix of a dummy column, or no dummy fires.
    market_price_sqm : market average price per sqm used for the ratio feature.
    stochastic : if True, sample from NB distribution; if False, return round(mu).
    seed : optional RNG seed for reproducibility.
    """

    coef: Sequence[float]
    feature_names: Sequence[str]
    alpha: float
    scalers: Optional[Dict[str, FeatureScaler]]
    class_group: str
    market_price_sqm: float
    stochastic: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.coef) != len(self.feature_names):
            raise ValueError(
                f"coef length {len(self.coef)} != feature_names length {len(self.feature_names)}"
            )
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    #: Default path to the bundled NB coefficients file (lives next to this module).
    DEFAULT_CSV_PATH: str = str(Path(__file__).parent / "nb_sales_models.csv")

    @classmethod
    def from_csv(
        cls,
        csv_path: Optional[str] = None,
        *,
        region: str,
        class_group: str,
        market_price_sqm: float,
        train_class: str = "all_classes",
        scalers: Optional[Dict[str, FeatureScaler]] = None,
        stochastic: bool = True,
        seed: Optional[int] = None,
    ) -> "FittedNBFeatureSalesModel":
        """
        Load NB_feature_remaining_lots coefficients from the bundled weights CSV.

        By default reads from ``nb_sales_models.csv`` shipped next to this module,
        which contains the NB_feature_remaining_lots / all_classes coefficients for
        Moscow (msk), St. Petersburg (spb) and Krasnodar (krd). Pass ``csv_path``
        only if you want to override the source file.

        Parameters
        ----------
        csv_path : path to the weights CSV. Defaults to the bundled
                   ``nb_sales_models.csv`` next to this module.
        region : region key filter on the 'region' column (e.g. 'msk', 'spb', 'krd').
        class_group : project class label (used to fire class_group_* dummies).
        market_price_sqm : market average price per sqm for ratio feature.
        train_class : row filter on the 'train_class' column (default 'all_classes').
        scalers : mapping from numeric feature name → FeatureScaler.
                  If None, raw (un-standardized) values are used — only for testing.
        """
        import csv as _csv

        if csv_path is None:
            csv_path = cls.DEFAULT_CSV_PATH

        rows: List[Dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                if (
                    row.get("region", "").strip() == region
                    and row.get("model_name", "").strip() == "NB_feature_remaining_lots"
                    and row.get("train_class", "").strip() == train_class
                ):
                    rows.append(row)

        if not rows:
            raise ValueError(
                f"No NB_feature_remaining_lots / region={region} / {train_class} "
                f"row found in {csv_path}"
            )
        row = rows[0]

        _meta = {"region", "model_name", "train_class", "__alpha__", "__kappa__"}
        feature_names: List[str] = [k for k in row if k not in _meta and row[k].strip() not in ("", "nan")]
        coef: List[float] = [float(row[fn]) for fn in feature_names]
        raw_alpha = row.get("__alpha__", "").strip()
        alpha = float(raw_alpha) if raw_alpha and raw_alpha != "nan" else 0.3

        return cls(
            coef=coef,
            feature_names=feature_names,
            alpha=alpha,
            scalers=scalers,
            class_group=class_group,
            market_price_sqm=market_price_sqm,
            stochastic=stochastic,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale(self, feature: str, value: float) -> float:
        """Apply scaler if available, otherwise return raw value."""
        if self.scalers and feature in self.scalers:
            return self.scalers[feature].transform(value)
        return float(value)

    def _build_feature_vector(
        self,
        *,
        ctx: "QuarterContext",
        config: "ProjectConfig",
        remaining_lots: int,
        price_sqm: float,
    ) -> List[float]:
        """
        Build the feature vector in the same order as self.feature_names.

        price_sqm is the current quarter's price (taken from previous_state
        or config.initial_price_sqm if first quarter).
        """
        market = max(self.market_price_sqm, 1.0)
        raw: Dict[str, float] = {
            "const": 1.0,
            "project_age_q": self._scale("project_age_q", float(ctx.index)),
            "quarters_to_delivery": self._scale(
                "quarters_to_delivery",
                float(max(-8, min(20, config.rvz_quarter_index - ctx.index))),
            ),
            "remaining_inventory_share": self._scale(
                "remaining_inventory_share",
                remaining_lots / max(config.initial_remaining_lots, 1),
            ),
            "log_avg_price_sqm_q": self._scale(
                "log_avg_price_sqm_q",
                math.log(max(price_sqm, 1.0)),
            ),
            "log_total_lots": self._scale(
                "log_total_lots",
                math.log(max(config.initial_remaining_lots, 1)),
            ),
            "log_price_to_market_year_ratio": self._scale(
                "log_price_to_market_year_ratio",
                math.log(max(price_sqm / market, 1e-3)),
            ),
            "log_remaining_lots_feat": self._scale(
                "log_remaining_lots_feat",
                math.log(max(remaining_lots, 1)),
            ),
            # Seasonality dummies (Q1 is reference category)
            "quarter_num_2": 1.0 if ctx.quarter_id.quarter == 2 else 0.0,
            "quarter_num_3": 1.0 if ctx.quarter_id.quarter == 3 else 0.0,
            "quarter_num_4": 1.0 if ctx.quarter_id.quarter == 4 else 0.0,
            # Class group dummies (reference class depends on training data sort)
            f"class_group_{self.class_group}": 1.0,
        }
        return [raw.get(fn, 0.0) for fn in self.feature_names]

    def _nb_sample(self, mu: float) -> int:
        """Sample from NB2(mu, alpha) via Gamma-Poisson mixture."""
        if mu <= 1e-9:
            return 0
        r = 1.0 / max(self.alpha, 1e-9)
        try:
            lam = self._rng.gammavariate(r, mu / r)
        except Exception:
            lam = mu
        return self._poisson_sample(lam)

    def _poisson_sample(self, lam: float) -> int:
        """Knuth algorithm for small lam; normal approximation for large lam."""
        if lam <= 0:
            return 0
        if lam > 700:
            return max(0, int(round(self._rng.gauss(lam, math.sqrt(lam)))))
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self._rng.random()
        return k - 1

    # ------------------------------------------------------------------
    # SalesModel interface
    # ------------------------------------------------------------------

    def predict_sold_lots(
        self,
        *,
        ctx: "QuarterContext",
        config: "ProjectConfig",
        scenario: "Scenario",
        remaining_lots: int,
        previous_state: Optional["QuarterState"],
    ) -> int:
        if remaining_lots <= 0 or ctx.is_pre_rns:
            return 0

        # Use previous quarter's price as proxy for current (one-quarter lag).
        # First quarter falls back to config.initial_price_sqm.
        base_price = previous_state.price_sqm if previous_state else config.initial_price_sqm
        price_sqm = base_price * scenario.global_state.price_level_multiplier

        x = self._build_feature_vector(
            ctx=ctx,
            config=config,
            remaining_lots=remaining_lots,
            price_sqm=price_sqm,
        )
        eta = sum(c * xv for c, xv in zip(self.coef, x))
        # Apply scenario sales-level multiplier in log-space
        eta += math.log(max(scenario.global_state.sales_level_multiplier, 1e-9))
        eta += scenario.local_path.sales_shock(ctx.index)
        mu = math.exp(max(-15.0, min(15.0, eta)))

        sold = self._nb_sample(mu) if self.stochastic else int(round(mu))
        return max(0, min(remaining_lots, sold))


# ---------------------------------------------------------------------------
# Scenario generation helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomScenarioSpec:
    key_rate_shift_min: float = 0.0
    key_rate_shift_max: float = 0.0
    price_level_min: float = 1.0
    price_level_max: float = 1.0
    sales_level_min: float = 1.0
    sales_level_max: float = 1.0
    project_price_premium_min: float = 0.0
    project_price_premium_max: float = 0.0
    rvz_delay_choices: Sequence[int] = (0,)
    local_price_shock_std: float = 0.0
    local_sales_shock_std: float = 0.0


class ScenarioFactory:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def make_random(self, *, name: str, horizon: int, spec: RandomScenarioSpec) -> Scenario:
        def unif(a: float, b: float) -> float:
            return self._rng.uniform(a, b)

        def gauss(std: float) -> float:
            return self._rng.gauss(0.0, std) if std > 0 else 0.0

        global_state = GlobalScenario(
            name=name,
            key_rate_shift_annual=unif(spec.key_rate_shift_min, spec.key_rate_shift_max),
            price_level_multiplier=unif(spec.price_level_min, spec.price_level_max),
            sales_level_multiplier=unif(spec.sales_level_min, spec.sales_level_max),
            project_price_premium=unif(spec.project_price_premium_min, spec.project_price_premium_max),
            rvz_delay_quarters=self._rng.choice(list(spec.rvz_delay_choices)),
        )
        local_path = LocalScenarioPath(
            price_shocks=[gauss(spec.local_price_shock_std) for _ in range(horizon)],
            sales_shocks=[gauss(spec.local_sales_shock_std) for _ in range(horizon)],
        )
        return Scenario(global_state=global_state, local_path=local_path)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OnePathCashflowEngine:
    def __init__(
        self,
        *,
        config: ProjectConfig,
        cost_schedule: CostSchedule,
        price_model: PriceModel,
        sales_model: SalesModel,
    ) -> None:
        self.config = config
        self.cost_schedule = cost_schedule
        self.price_model = price_model
        self.sales_model = sales_model
        self.cost_schedule.validate(config.horizon_quarters)

    def run(self, scenario: Optional[Scenario] = None) -> PathResult:
        scenario = scenario or Scenario()
        scenario.validate(self.config.horizon_quarters)

        states: List[QuarterState] = []
        quarter_id = QuarterId(self.config.start_year, self.config.start_quarter)
        remaining_lots = self.config.initial_remaining_lots
        escrow_balance = 0.0
        project_cash_balance = 0.0
        debt_outstanding = 0.0
        cumulative_financing_cost = 0.0
        cumulative_positive_osz = 0.0
        repaid_at_rvz = False
        profit_before_tax_accum = 0.0

        rvz_effective_index = min(
            self.config.horizon_quarters - 1,
            self.config.rvz_quarter_index + scenario.global_state.rvz_delay_quarters,
        )

        for idx in range(self.config.horizon_quarters):
            ctx = QuarterContext(
                index=idx,
                quarter_id=quarter_id,
                is_pre_rns=idx < self.config.rns_quarter_index,
                is_post_rvz=idx > rvz_effective_index,
                is_rvz_quarter=idx == rvz_effective_index,
                rvz_effective_index=rvz_effective_index,
            )
            prev_state = states[-1] if states else None

            price_sqm = self.price_model.predict_price_sqm(
                ctx=ctx,
                config=self.config,
                scenario=scenario,
                previous_state=prev_state,
            )

            sold_lots = self.sales_model.predict_sold_lots(
                ctx=ctx,
                config=self.config,
                scenario=scenario,
                remaining_lots=remaining_lots,
                previous_state=prev_state,
            )
            sold_lots = min(sold_lots, remaining_lots)
            sold_area = sold_lots * self.config.avg_unit_area_sqm
            revenue = sold_area * price_sqm

            equity_inflow = self.config.resolved_initial_equity if idx == 0 else 0.0
            project_cash_balance += equity_inflow

            # Pre-RVZ sales go to escrow. At RVZ they are released to project cash.
            # Post-RVZ sales are treated as direct project cash inflow.
            escrow_balance_for_coverage = 0.0
            if ctx.is_post_rvz:
                escrow_inflow = 0.0
                escrow_release = 0.0
                project_cash_inflow = revenue
                escrow_balance = 0.0
            else:
                escrow_inflow = revenue
                escrow_balance_for_coverage = escrow_balance + escrow_inflow
                if ctx.is_rvz_quarter:
                    escrow_release = escrow_balance_for_coverage
                    project_cash_inflow = escrow_release
                    escrow_balance = 0.0
                else:
                    escrow_release = 0.0
                    project_cash_inflow = 0.0
                    escrow_balance = escrow_balance_for_coverage

            project_cash_balance += project_cash_inflow

            land_cost = self.cost_schedule.land(idx)
            smr_cost = self.cost_schedule.smr(idx)
            design_cost = self.cost_schedule.design(idx)
            other_opex = self.cost_schedule.other_opex(idx)
            post_completion_cost = self.cost_schedule.post_completion(idx)
            marketing_cost = revenue * self.config.marketing_cost_ratio
            property_tax = self.cost_schedule.property_tax(idx)
            vat = self.cost_schedule.vat(idx)

            pre_tax_profit_proxy = revenue - land_cost - smr_cost - design_cost - marketing_cost - other_opex
            profit_before_tax_accum += pre_tax_profit_proxy
            profit_tax = 0.0
            if self.config.recognize_profit_tax_at_rvz and ctx.is_rvz_quarter and profit_before_tax_accum > 0:
                profit_tax = profit_before_tax_accum * self.config.profit_tax_rate

            total_costs_excl_financing = (
                land_cost
                + smr_cost
                + design_cost
                + marketing_cost
                + other_opex
                + post_completion_cost
                + property_tax
                + vat
                + profit_tax
            )

            debt_limit = self.config.resolved_debt_limit
            cash_before_costs = project_cash_balance
            operating_funding_gap = max(total_costs_excl_financing - project_cash_balance, 0.0)
            operating_debt_draw = min(operating_funding_gap, max(debt_limit - debt_outstanding, 0.0))
            project_cash_balance = project_cash_balance + operating_debt_draw - total_costs_excl_financing

            current_key_rate = self.config.key_rate_annual + scenario.global_state.key_rate_shift_annual
            spread = (
                self.config.full_rate_spread_before_rvz
                if idx <= rvz_effective_index
                else self.config.full_rate_spread_after_rvz
            )
            full_rate_annual = current_key_rate + spread

            debt_after_operating_draw = debt_outstanding + operating_debt_draw
            escrow_coverage_ratio = (
                min(safe_div(escrow_balance_for_coverage, debt_after_operating_draw), 1.0)
                if debt_after_operating_draw > 0
                else 0.0
            )
            interest_payment = debt_after_operating_draw * (
                (1.0 - escrow_coverage_ratio) * full_rate_annual + escrow_coverage_ratio * self.config.privileged_rate_annual
            ) / 4.0
            line_usage_fee_payment = debt_limit * self.config.line_usage_fee_annual / 4.0 if debt_limit > 0 else 0.0
            unused_debt_limit_for_fee = max(debt_limit - debt_after_operating_draw, 0.0)
            reserve_fee_payment = unused_debt_limit_for_fee * self.config.reserve_fee_annual / 4.0
            total_debt_service_cost = interest_payment + line_usage_fee_payment + reserve_fee_payment

            financing_funding_gap = max(total_debt_service_cost - project_cash_balance, 0.0)
            financing_debt_draw = min(financing_funding_gap, max(debt_limit - debt_after_operating_draw, 0.0))
            debt_draw = operating_debt_draw + financing_debt_draw
            project_cash_balance = project_cash_balance + financing_debt_draw - total_debt_service_cost
            debt_before_repayment = debt_outstanding + debt_draw
            unused_debt_limit = max(debt_limit - debt_before_repayment, 0.0)

            cumulative_financing_cost += total_debt_service_cost
            if debt_after_operating_draw > 0:
                cumulative_positive_osz += debt_after_operating_draw
            effective_rate_cum = safe_div(cumulative_financing_cost, cumulative_positive_osz) * 4.0

            debt_repayment = 0.0
            if ctx.is_rvz_quarter or ctx.is_post_rvz:
                debt_repayment = min(max(project_cash_balance, 0.0), debt_before_repayment)
                project_cash_balance -= debt_repayment
            debt_outstanding = max(debt_before_repayment - debt_repayment, 0.0)
            repaid_at_rvz = repaid_at_rvz or (ctx.is_rvz_quarter and debt_repayment > 0)

            # Ratios: align with workbook spirit.
            debt_service_for_ratio = debt_repayment + total_debt_service_cost
            cfads = project_cash_inflow - total_costs_excl_financing
            dscr = safe_div(cfads, debt_service_for_ratio)
            iscr = safe_div(cfads, interest_payment)

            net_cash_flow = (
                equity_inflow
                + project_cash_inflow
                + debt_draw
                - total_costs_excl_financing
                - total_debt_service_cost
                - debt_repayment
            )

            remaining_lots_end = max(remaining_lots - sold_lots, 0)
            liquidity_shortfall = max(-project_cash_balance, 0.0)

            states.append(
                QuarterState(
                    index=idx,
                    quarter_id=quarter_id,
                    remaining_lots_start=remaining_lots,
                    remaining_lots_end=remaining_lots_end,
                    price_sqm=price_sqm,
                    sold_lots=sold_lots,
                    sold_area_sqm=sold_area,
                    revenue=revenue,
                    escrow_inflow=escrow_inflow,
                    escrow_release=escrow_release,
                    escrow_balance_end=escrow_balance,
                    project_cash_inflow=project_cash_inflow,
                    equity_inflow=equity_inflow,
                    project_cash_balance_end=project_cash_balance,
                    escrow_coverage_ratio=escrow_coverage_ratio,
                    land_cost=land_cost,
                    smr_cost=smr_cost,
                    design_cost=design_cost,
                    marketing_cost=marketing_cost,
                    other_opex=other_opex,
                    post_completion_cost=post_completion_cost,
                    property_tax=property_tax,
                    vat=vat,
                    profit_tax=profit_tax,
                    total_costs_excl_financing=total_costs_excl_financing,
                    debt_draw=debt_draw,
                    debt_repayment=debt_repayment,
                    debt_outstanding_end=debt_outstanding,
                    debt_limit=debt_limit,
                    unused_debt_limit=unused_debt_limit,
                    full_rate_annual=full_rate_annual,
                    interest_payment=interest_payment,
                    line_usage_fee_payment=line_usage_fee_payment,
                    reserve_fee_payment=reserve_fee_payment,
                    total_debt_service_cost=total_debt_service_cost,
                    effective_rate_cumulative_annual=effective_rate_cum,
                    net_cash_flow=net_cash_flow,
                    cfads=cfads,
                    debt_service_for_ratio=debt_service_for_ratio,
                    dscr=dscr,
                    iscr=iscr,
                    notes={
                        "current_key_rate": current_key_rate,
                        "profit_before_tax_accum": profit_before_tax_accum,
                        "cash_before_costs": cash_before_costs,
                        "operating_funding_gap": operating_funding_gap,
                        "financing_funding_gap": financing_funding_gap,
                        "debt_before_repayment": debt_before_repayment,
                        "liquidity_shortfall": liquidity_shortfall,
                    },
                )
            )

            remaining_lots = remaining_lots_end
            quarter_id = quarter_id.next()

        collateral_value_net = self.config.sellable_area_sqm * self.config.collateral_price_sqm * (1.0 - self.config.collateral_discount)
        collateral_coverage_ratio = safe_div(collateral_value_net, self.config.resolved_debt_limit + cumulative_financing_cost)
        dscr_observation_points = [
            state.dscr
            for state in states
            if state.debt_service_for_ratio > 0 and (state.project_cash_inflow > 0 or state.debt_repayment > 0)
        ]
        iscr_observation_points = [
            state.iscr
            for state in states
            if state.interest_payment > 0 and (state.project_cash_inflow > 0 or state.debt_repayment > 0)
        ]

        summary = PathSummary(
            project_name=self.config.name,
            scenario_name=scenario.global_state.name,
            total_revenue=sum(s.revenue for s in states),
            total_costs_excl_financing=sum(s.total_costs_excl_financing for s in states),
            total_interest_and_fees=sum(s.total_debt_service_cost for s in states),
            total_debt_service_for_ratio=sum(s.debt_service_for_ratio for s in states),
            total_cfads=sum(s.cfads for s in states),
            total_net_cash_flow=sum(s.net_cash_flow for s in states),
            ending_escrow_balance=states[-1].escrow_balance_end if states else 0.0,
            ending_project_cash_balance=states[-1].project_cash_balance_end if states else 0.0,
            peak_debt=max((s.notes.get("debt_before_repayment", s.debt_outstanding_end) for s in states), default=0.0),
            ending_debt=states[-1].debt_outstanding_end if states else 0.0,
            min_dscr=min(dscr_observation_points, default=0.0),
            min_iscr=min(iscr_observation_points, default=0.0),
            max_liquidity_shortfall=max((s.notes.get("liquidity_shortfall", 0.0) for s in states), default=0.0),
            final_effective_rate_annual=states[-1].effective_rate_cumulative_annual if states else 0.0,
            collateral_value_net=collateral_value_net,
            collateral_coverage_ratio=collateral_coverage_ratio,
            repaid_at_rvz=repaid_at_rvz,
            debt_fully_repaid=(states[-1].debt_outstanding_end <= 1e-6 if states else True),
        )
        return PathResult(states=states, summary=summary)


# ---------------------------------------------------------------------------
# Monte Carlo wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloSummary:
    runs: int
    mean_min_dscr: float
    p05_min_dscr: float
    p50_min_dscr: float
    p95_min_dscr: float
    probability_dscr_below_1_0: float
    probability_dscr_below_1_2: float
    mean_peak_debt: float
    mean_ending_debt: float
    mean_final_effective_rate_annual: float
    mean_collateral_coverage_ratio: float


@dataclass(frozen=True)
class MonteCarloResult:
    path_results: List[PathResult]
    summary: MonteCarloSummary


class MonteCarloRunner:
    def __init__(self, engine: OnePathCashflowEngine, scenario_factory: ScenarioFactory) -> None:
        self.engine = engine
        self.scenario_factory = scenario_factory

    def run_random(self, *, n_runs: int, spec: RandomScenarioSpec, name_prefix: str = "mc") -> MonteCarloResult:
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")

        results: List[PathResult] = []
        for i in range(n_runs):
            scenario = self.scenario_factory.make_random(
                name=f"{name_prefix}_{i+1}",
                horizon=self.engine.config.horizon_quarters,
                spec=spec,
            )
            results.append(self.engine.run(scenario))
        return MonteCarloResult(path_results=results, summary=self._summarize(results))

    @staticmethod
    def _summarize(results: Sequence[PathResult]) -> MonteCarloSummary:
        min_dscrs = sorted(r.summary.min_dscr for r in results)
        peak_debts = [r.summary.peak_debt for r in results]
        ending_debts = [r.summary.ending_debt for r in results]
        eff_rates = [r.summary.final_effective_rate_annual for r in results]
        coverage = [r.summary.collateral_coverage_ratio for r in results]

        def q(xs: Sequence[float], p: float) -> float:
            if not xs:
                return 0.0
            if len(xs) == 1:
                return xs[0]
            idx = (len(xs) - 1) * p
            lo = math.floor(idx)
            hi = math.ceil(idx)
            if lo == hi:
                return xs[lo]
            w = idx - lo
            return xs[lo] * (1.0 - w) + xs[hi] * w

        return MonteCarloSummary(
            runs=len(results),
            mean_min_dscr=mean(min_dscrs) if min_dscrs else 0.0,
            p05_min_dscr=q(min_dscrs, 0.05),
            p50_min_dscr=q(min_dscrs, 0.50),
            p95_min_dscr=q(min_dscrs, 0.95),
            probability_dscr_below_1_0=safe_div(sum(v < 1.0 for v in min_dscrs), len(min_dscrs)),
            probability_dscr_below_1_2=safe_div(sum(v < 1.2 for v in min_dscrs), len(min_dscrs)),
            mean_peak_debt=mean(peak_debts) if peak_debts else 0.0,
            mean_ending_debt=mean(ending_debts) if ending_debts else 0.0,
            mean_final_effective_rate_annual=mean(eff_rates) if eff_rates else 0.0,
            mean_collateral_coverage_ratio=mean(coverage) if coverage else 0.0,
        )


# ---------------------------------------------------------------------------
# Convenience helpers for tests / quick manual runs
# ---------------------------------------------------------------------------


def build_equal_share_schedule(horizon_quarters: int, total_share: float) -> List[float]:
    if horizon_quarters <= 0:
        raise ValueError("horizon_quarters must be positive")
    return [total_share / horizon_quarters] * horizon_quarters


def _legacy_make_demo_engine() -> OnePathCashflowEngine:
    config = ProjectConfig(
        name="Demo ЖК",
        start_year=2026,
        start_quarter=1,
        horizon_quarters=10,
        rns_quarter_index=0,
        rvz_quarter_index=8,
        sellable_area_sqm=11113.0,
        avg_unit_area_sqm=50.29,
        initial_remaining_lots=221,
        key_rate_annual=0.21,
        full_rate_spread_before_rvz=0.035,
        full_rate_spread_after_rvz=0.035,
        privileged_rate_annual=0.037,
        reserve_fee_annual=0.005,
        ltc=0.926,
        land_cost_total=125_637_000.0,
        smr_cost_total=1_034_039_000.0,
        design_cost_total=0.0,
        marketing_cost_ratio=0.05,
        profit_tax_rate=0.25,
        initial_price_sqm=192_459.0,
        collateral_discount=0.30,
        collateral_price_sqm=168_000.0,
    )
    cost_schedule = CostSchedule(
        land_by_quarter=[125_637_000.0] + [0.0] * 9,
        smr_by_quarter=[0.0, 246_434_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 0.0],
    )
    price_model = StubPriceModel()
    sales_model = StubSalesModel(
        quarterly_sales_share=[0.118, 0.138, 0.144, 0.144, 0.138, 0.100, 0.080, 0.060, 0.040, 0.020]
    )
    return OnePathCashflowEngine(
        config=config,
        cost_schedule=cost_schedule,
        price_model=price_model,
        sales_model=sales_model,
    )


def scale_values(values: Sequence[float], multiplier: float) -> List[float]:
    return [float(value) * multiplier for value in values]


def make_demo_inputs() -> tuple[ProjectConfig, CostSchedule, List[float]]:
    config = ProjectConfig(
        name="demo_base",
        start_year=2026,
        start_quarter=1,
        horizon_quarters=10,
        rns_quarter_index=0,
        rvz_quarter_index=8,
        sellable_area_sqm=11113.0,
        avg_unit_area_sqm=50.29,
        initial_remaining_lots=221,
        key_rate_annual=0.21,
        full_rate_spread_before_rvz=0.035,
        full_rate_spread_after_rvz=0.035,
        privileged_rate_annual=0.037,
        reserve_fee_annual=0.005,
        ltc=0.926,
        land_cost_total=125_637_000.0,
        smr_cost_total=1_034_039_000.0,
        design_cost_total=0.0,
        marketing_cost_ratio=0.05,
        profit_tax_rate=0.25,
        initial_price_sqm=192_459.0,
        collateral_discount=0.30,
        collateral_price_sqm=168_000.0,
    )
    cost_schedule = CostSchedule(
        land_by_quarter=[125_637_000.0] + [0.0] * 9,
        smr_by_quarter=[0.0, 246_434_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0, 0.0],
    )
    sales_shares = [0.118, 0.138, 0.144, 0.144, 0.138, 0.128, 0.110, 0.081, 0.0, 0.0]
    return config, cost_schedule, sales_shares


def make_engine_from_inputs(
    *,
    config: ProjectConfig,
    cost_schedule: CostSchedule,
    sales_shares: Sequence[float],
) -> OnePathCashflowEngine:
    return OnePathCashflowEngine(
        config=config,
        cost_schedule=cost_schedule,
        price_model=StubPriceModel(),
        sales_model=StubSalesModel(quarterly_sales_share=list(sales_shares)),
    )


def make_demo_engine() -> OnePathCashflowEngine:
    config, cost_schedule, sales_shares = make_demo_inputs()
    return make_engine_from_inputs(
        config=replace(config, name="demo_base"),
        cost_schedule=cost_schedule,
        sales_shares=sales_shares,
    )


@dataclass(frozen=True)
class SyntheticExperimentCase:
    name: str
    description: str
    engine: OnePathCashflowEngine
    scenario_spec: RandomScenarioSpec


def make_synthetic_experiment_cases() -> List[SyntheticExperimentCase]:
    base_config, base_schedule, base_sales_shares = make_demo_inputs()

    balanced_schedule = replace(
        base_schedule,
        post_completion_cost_by_quarter=[0.0] * 9 + [4_500_000.0],
        other_opex_by_quarter=[2_500_000.0] * 10,
    )
    balanced_case = SyntheticExperimentCase(
        name="balanced_base",
        description="Balanced synthetic plan close to the demo case.",
        engine=make_engine_from_inputs(
            config=replace(base_config, name="balanced_base"),
            cost_schedule=balanced_schedule,
            sales_shares=base_sales_shares,
        ),
        scenario_spec=RandomScenarioSpec(
            key_rate_shift_min=-0.01,
            key_rate_shift_max=0.02,
            price_level_min=0.96,
            price_level_max=1.05,
            sales_level_min=0.90,
            sales_level_max=1.10,
            rvz_delay_choices=(0, 1),
            local_price_shock_std=0.010,
            local_sales_shock_std=0.040,
        ),
    )

    stress_schedule = replace(
        base_schedule,
        smr_by_quarter=scale_values(base_schedule.smr_by_quarter, 1.08),
        post_completion_cost_by_quarter=[0.0] * 8 + [6_000_000.0, 9_000_000.0],
        other_opex_by_quarter=[3_500_000.0] * 10,
    )
    stress_case = SyntheticExperimentCase(
        name="sales_stress",
        description="Lower prices, slower sales, slightly higher costs and rates.",
        engine=make_engine_from_inputs(
            config=replace(
                base_config,
                name="sales_stress",
                key_rate_annual=0.23,
                reserve_fee_annual=0.007,
                marketing_cost_ratio=0.055,
                initial_price_sqm=176_000.0,
                collateral_price_sqm=160_000.0,
                smr_cost_total=base_config.smr_cost_total * 1.08,
            ),
            cost_schedule=stress_schedule,
            sales_shares=[0.080, 0.100, 0.110, 0.110, 0.100, 0.080, 0.070, 0.050, 0.040, 0.030],
        ),
        scenario_spec=RandomScenarioSpec(
            key_rate_shift_min=0.00,
            key_rate_shift_max=0.04,
            price_level_min=0.88,
            price_level_max=1.00,
            sales_level_min=0.75,
            sales_level_max=1.00,
            rvz_delay_choices=(0, 1, 2),
            local_price_shock_std=0.015,
            local_sales_shock_std=0.070,
        ),
    )

    upside_schedule = replace(
        base_schedule,
        smr_by_quarter=scale_values(base_schedule.smr_by_quarter, 0.97),
        other_opex_by_quarter=[2_000_000.0] * 10,
    )
    upside_case = SyntheticExperimentCase(
        name="fast_sales_upside",
        description="Higher prices, faster sales and slightly cheaper debt.",
        engine=make_engine_from_inputs(
            config=replace(
                base_config,
                name="fast_sales_upside",
                key_rate_annual=0.18,
                full_rate_spread_before_rvz=0.030,
                full_rate_spread_after_rvz=0.028,
                privileged_rate_annual=0.032,
                marketing_cost_ratio=0.045,
                initial_price_sqm=210_000.0,
                collateral_price_sqm=180_000.0,
                smr_cost_total=base_config.smr_cost_total * 0.97,
            ),
            cost_schedule=upside_schedule,
            sales_shares=[0.150, 0.150, 0.140, 0.130, 0.110, 0.090, 0.070, 0.050, 0.030, 0.020],
        ),
        scenario_spec=RandomScenarioSpec(
            key_rate_shift_min=-0.03,
            key_rate_shift_max=0.01,
            price_level_min=0.98,
            price_level_max=1.10,
            sales_level_min=0.95,
            sales_level_max=1.15,
            rvz_delay_choices=(0,),
            local_price_shock_std=0.008,
            local_sales_shock_std=0.035,
        ),
    )

    return [balanced_case, stress_case, upside_case]


def quarter_state_to_row(state: QuarterState) -> Dict[str, float | int | str]:
    row: Dict[str, float | int | str] = {
        "index": state.index,
        "quarter_label": state.quarter_id.label(),
        "quarter_year": state.quarter_id.year,
        "quarter_number": state.quarter_id.quarter,
    }
    for key, value in asdict(state).items():
        if key in {"quarter_id", "notes"}:
            continue
        row[key] = value
    for key, value in state.notes.items():
        row[f"note_{key}"] = value
    return row


def write_csv_rows(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def plot_path_result(path_result: PathResult, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quarters = [state.quarter_id.label() for state in path_result.states]
    revenue = [state.revenue for state in path_result.states]
    costs = [state.total_costs_excl_financing for state in path_result.states]
    cfads = [state.cfads for state in path_result.states]
    debt = [state.debt_outstanding_end for state in path_result.states]
    escrow = [state.escrow_balance_end for state in path_result.states]
    cash = [state.project_cash_balance_end for state in path_result.states]
    sold_lots = [state.sold_lots for state in path_result.states]
    remaining_lots = [state.remaining_lots_end for state in path_result.states]
    dscr = [state.dscr for state in path_result.states]
    iscr = [state.iscr for state in path_result.states]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(quarters, revenue, marker="o", label="Revenue")
    axes[0, 0].plot(quarters, costs, marker="o", label="Costs excl fin")
    axes[0, 0].plot(quarters, cfads, marker="o", label="CFADS")
    axes[0, 0].set_title("Revenue, costs and CFADS")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].plot(quarters, debt, marker="o", label="Debt outstanding")
    axes[0, 1].plot(quarters, escrow, marker="o", label="Escrow balance")
    axes[0, 1].plot(quarters, cash, marker="o", label="Project cash")
    axes[0, 1].set_title("Debt, escrow and project cash")
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(quarters, sold_lots, label="Sold lots")
    axes[1, 0].plot(quarters, remaining_lots, color="black", marker="o", label="Remaining lots")
    axes[1, 0].set_title("Sales pace")
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].plot(quarters, dscr, marker="o", label="DSCR")
    axes[1, 1].plot(quarters, iscr, marker="o", label="ISCR")
    axes[1, 1].axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="1.0x")
    axes[1, 1].axhline(1.2, color="orange", linestyle="--", linewidth=1.0, label="1.2x")
    axes[1, 1].set_title("Coverage ratios")
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"{path_result.summary.project_name} | {path_result.summary.scenario_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_monte_carlo_result(mc_result: MonteCarloResult, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    min_dscr = [result.summary.min_dscr for result in mc_result.path_results]
    peak_debt = [result.summary.peak_debt for result in mc_result.path_results]
    ending_debt = [result.summary.ending_debt for result in mc_result.path_results]
    eff_rate = [result.summary.final_effective_rate_annual for result in mc_result.path_results]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(min_dscr, bins=20, color="#4C72B0", edgecolor="white")
    axes[0, 0].axvline(1.0, color="red", linestyle="--", linewidth=1.0, label="1.0x")
    axes[0, 0].axvline(1.2, color="orange", linestyle="--", linewidth=1.0, label="1.2x")
    axes[0, 0].set_title("Distribution of min DSCR")
    axes[0, 0].legend()

    axes[0, 1].hist(peak_debt, bins=20, color="#55A868", edgecolor="white")
    axes[0, 1].set_title("Distribution of peak debt")

    axes[1, 0].hist(ending_debt, bins=20, color="#C44E52", edgecolor="white")
    axes[1, 0].set_title("Distribution of ending debt")

    axes[1, 1].scatter(eff_rate, min_dscr, alpha=0.7, color="#8172B2")
    axes[1, 1].set_xlabel("Final effective rate annual")
    axes[1, 1].set_ylabel("Min DSCR")
    axes[1, 1].set_title("Rate vs min DSCR")

    fig.suptitle(f"Monte Carlo | runs={mc_result.summary.runs}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_case_results(
    *,
    case: SyntheticExperimentCase,
    base_result: PathResult,
    mc_result: MonteCarloResult,
    case_dir: Path,
) -> Dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)

    write_json(asdict(case.engine.config), case_dir / "config.json")
    write_json(asdict(case.engine.cost_schedule), case_dir / "cost_schedule.json")
    if isinstance(case.engine.sales_model, StubSalesModel):
        write_json({"quarterly_sales_share": list(case.engine.sales_model.quarterly_sales_share)}, case_dir / "sales_model.json")
    write_json(asdict(case.scenario_spec), case_dir / "mc_spec.json")

    write_json(asdict(base_result.summary), case_dir / "base_summary.json")
    write_csv_rows([quarter_state_to_row(state) for state in base_result.states], case_dir / "base_states.csv")
    plot_path_result(base_result, case_dir / "base_path.png")

    write_json(asdict(mc_result.summary), case_dir / "mc_summary.json")
    write_csv_rows([asdict(result.summary) for result in mc_result.path_results], case_dir / "mc_path_summaries.csv")
    plot_monte_carlo_result(mc_result, case_dir / "mc_summary.png")

    return {
        "case_name": case.name,
        "description": case.description,
        "base_total_revenue": base_result.summary.total_revenue,
        "base_peak_debt": base_result.summary.peak_debt,
        "base_ending_debt": base_result.summary.ending_debt,
        "base_ending_project_cash_balance": base_result.summary.ending_project_cash_balance,
        "base_min_dscr": base_result.summary.min_dscr,
        "base_max_liquidity_shortfall": base_result.summary.max_liquidity_shortfall,
        "mc_runs": mc_result.summary.runs,
        "mc_mean_min_dscr": mc_result.summary.mean_min_dscr,
        "mc_p05_min_dscr": mc_result.summary.p05_min_dscr,
        "mc_p50_min_dscr": mc_result.summary.p50_min_dscr,
        "mc_p95_min_dscr": mc_result.summary.p95_min_dscr,
        "mc_prob_dscr_lt_1_0": mc_result.summary.probability_dscr_below_1_0,
        "mc_prob_dscr_lt_1_2": mc_result.summary.probability_dscr_below_1_2,
        "mc_mean_peak_debt": mc_result.summary.mean_peak_debt,
        "mc_mean_ending_debt": mc_result.summary.mean_ending_debt,
        "artifacts_dir": str(case_dir),
    }


def run_synthetic_experiments(*, output_dir: Path, mc_runs: int, seed: int) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for offset, case in enumerate(make_synthetic_experiment_cases()):
        base_result = case.engine.run()
        factory = ScenarioFactory(seed=seed + offset)
        mc_result = MonteCarloRunner(case.engine, factory).run_random(
            n_runs=mc_runs,
            spec=case.scenario_spec,
            name_prefix=case.name,
        )
        summary_rows.append(
            export_case_results(
                case=case,
                base_result=base_result,
                mc_result=mc_result,
                case_dir=output_dir / case.name,
            )
        )
    return summary_rows


def resolve_output_dir(base_output_dir: Optional[Path]) -> Path:
    root = base_output_dir or (Path(__file__).resolve().parent / "synthetic_runs")
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic Monte Carlo cashflow experiments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory where a timestamped run folder will be created.",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=120,
        help="Monte Carlo runs per synthetic case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for synthetic scenarios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    summary_rows = run_synthetic_experiments(
        output_dir=output_dir,
        mc_runs=args.mc_runs,
        seed=args.seed,
    )
    write_csv_rows(summary_rows, output_dir / "suite_summary.csv")

    print(f"Synthetic experiment suite saved to: {output_dir}")
    for row in summary_rows:
        print(
            f"{row['case_name']}: "
            f"base_ending_debt={float(row['base_ending_debt']):,.0f}, "
            f"base_cash={float(row['base_ending_project_cash_balance']):,.0f}, "
            f"mc_p05_min_dscr={float(row['mc_p05_min_dscr']):.2f}, "
            f"mc_prob_dscr_lt_1={float(row['mc_prob_dscr_lt_1_0']):.1%}"
        )


if __name__ == "__main__":
    main()
