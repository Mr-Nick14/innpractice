from __future__ import annotations

"""
Stochastic driver layer for Monte Carlo cashflow simulations.

The module is intentionally inference-only:
- no training
- no feature fitting
- deterministic reproducibility with seeds

It separates path-level and quarter-level uncertainty and exposes a runtime
cost adapter so the main cashflow engine can keep its deterministic baseline
schedule while applying stochastic deviations at runtime.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _as_float_list(seq: Sequence[float], *, horizon: int) -> List[float]:
    out = [float(x) for x in seq[:horizon]]
    if len(out) < horizon:
        out.extend([0.0] * (horizon - len(out)))
    return out


def _normalize_probs(probs: Sequence[float]) -> np.ndarray:
    arr = np.asarray([max(float(x), 0.0) for x in probs], dtype=float)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full_like(arr, 1.0 / max(len(arr), 1), dtype=float)
    return arr / total


def _repair_correlation_matrix(matrix: Sequence[Sequence[float]], n: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"Correlation matrix shape {arr.shape} does not match expected {(n, n)}")
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 1.0)

    # Project to nearest positive semidefinite matrix, then renormalize to a correlation matrix.
    eigvals, eigvecs = np.linalg.eigh(arr)
    eigvals = np.clip(eigvals, 1e-8, None)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    scale = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    corr = psd / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return corr


def _sample_correlated_normals(
    rng: np.random.Generator,
    *,
    names: Sequence[str],
    stds: Sequence[float],
    corr_matrix: Optional[Sequence[Sequence[float]]],
    use_correlations: bool,
) -> Dict[str, float]:
    names_list = list(names)
    stds_arr = np.asarray([max(float(x), 0.0) for x in stds], dtype=float)
    if len(names_list) != len(stds_arr):
        raise ValueError("names and stds must have equal length")

    if len(names_list) == 0:
        return {}

    if not use_correlations or corr_matrix is None:
        draws = rng.standard_normal(len(names_list))
        return {name: float(draw * std) for name, draw, std in zip(names_list, draws, stds_arr)}

    corr = _repair_correlation_matrix(corr_matrix, len(names_list))
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal(len(names_list))
    correlated = chol @ z
    return {name: float(draw * std) for name, draw, std in zip(names_list, correlated, stds_arr)}


def _shift_series(values: Sequence[float], shift_quarters: int, horizon: int) -> List[float]:
    out = [0.0] * horizon
    if horizon <= 0:
        return out
    for idx, value in enumerate(values[:horizon]):
        new_idx = idx + int(shift_quarters)
        if 0 <= new_idx < horizon:
            out[new_idx] += float(value)
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloDriverConfig:
    """Parameters controlling the stochastic driver layer."""

    market_path_mode: str = "stochastic"
    market_regime_shift_std: float = 0.015
    market_shock_std: float = 0.012
    market_mean_reversion: float = 0.35
    market_drift_annual: float = 0.0
    macro_shift_std: float = 0.010
    project_premium_std: float = 0.020
    final_price_residual_std: float = 0.008
    demand_shift_std: float = 0.060
    seasonal_sales_shock_std: float = 0.040
    delay_sales_penalty_log_per_quarter: float = 0.025
    rvz_delay_values: Sequence[int] = (0, 1, 2, 3)
    rvz_delay_probs: Sequence[float] = (0.55, 0.25, 0.12, 0.08)
    path_delay_risk_sensitivity: float = 0.35
    key_rate_innovation_std: float = 0.008
    spread_shock_std: float = 0.004
    cost_inflation_std: float = 0.012
    overrun_log_std: float = 0.070
    cost_quarterly_shock_std: float = 0.010
    use_correlations: bool = False
    path_corr_matrix: Optional[Sequence[Sequence[float]]] = None
    quarter_corr_matrix: Optional[Sequence[Sequence[float]]] = None
    max_cost_timing_shift_quarters: int = 2
    apply_cost_timing_shift: bool = True

    @staticmethod
    def default_path_corr_matrix() -> List[List[float]]:
        return [
            [1.00, 0.20, 0.10, 0.30, 0.15, 0.05, 0.00, 0.10, 0.10],
            [0.20, 1.00, 0.05, 0.10, 0.10, 0.35, 0.45, 0.30, 0.50],
            [0.10, 0.05, 1.00, 0.25, 0.10, 0.05, 0.00, 0.05, 0.10],
            [0.30, 0.10, 0.25, 1.00, 0.45, 0.15, 0.10, 0.25, 0.40],
            [0.15, 0.10, 0.10, 0.45, 1.00, 0.20, 0.15, 0.35, 0.65],
            [0.05, 0.35, 0.05, 0.15, 0.20, 1.00, 0.30, 0.15, 0.35],
            [0.00, 0.45, 0.00, 0.10, 0.15, 0.30, 1.00, 0.20, 0.35],
            [0.10, 0.30, 0.05, 0.25, 0.35, 0.15, 0.20, 1.00, 0.55],
            [0.10, 0.50, 0.10, 0.40, 0.65, 0.35, 0.35, 0.55, 1.00],
        ]

    @staticmethod
    def default_quarter_corr_matrix() -> List[List[float]]:
        return [
            [1.00, 0.15, 0.35, 0.20, -0.20, 0.05, 0.25],
            [0.15, 1.00, 0.10, 0.05, -0.05, 0.00, 0.05],
            [0.35, 0.10, 1.00, 0.10, -0.15, 0.05, 0.45],
            [0.20, 0.05, 0.10, 1.00, 0.20, 0.35, 0.15],
            [-0.20, -0.05, -0.15, 0.20, 1.00, 0.15, -0.10],
            [0.05, 0.00, 0.05, 0.35, 0.15, 1.00, 0.10],
            [0.25, 0.05, 0.45, 0.15, -0.10, 0.10, 1.00],
        ]

    def resolved_path_corr_matrix(self) -> Sequence[Sequence[float]]:
        return self.path_corr_matrix if self.path_corr_matrix is not None else self.default_path_corr_matrix()

    def resolved_quarter_corr_matrix(self) -> Sequence[Sequence[float]]:
        return self.quarter_corr_matrix if self.quarter_corr_matrix is not None else self.default_quarter_corr_matrix()

    def market_noise_scale(self) -> float:
        mode = self.market_path_mode.strip().lower()
        if mode == "replay":
            return 0.0
        if mode == "trend":
            return 0.5
        if mode in {"stochastic", "scenario_stochastic", "macro_guided"}:
            return 1.0
        return 1.0

    def macro_price_scale(self) -> float:
        mode = self.market_path_mode.strip().lower()
        if mode == "macro_guided":
            return 1.5
        if mode == "trend":
            return 0.75
        return 1.0


# ---------------------------------------------------------------------------
# Path / quarter states
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathDriverState:
    market_regime_shift_log: float = 0.0
    macro_shift_annual: float = 0.0
    project_premium_shift_log: float = 0.0
    demand_shift_log: float = 0.0
    project_overrun_multiplier: float = 1.0
    rvz_delay_quarters: int = 0
    sales_start_shift_quarters: int = 0
    cost_inflation_annual_shift: float = 0.0
    spread_shift_annual: float = 0.0
    key_rate_shift_annual: float = 0.0
    cost_timing_shift_quarters: int = 0
    delay_sales_penalty_log_per_quarter: float = 0.0
    execution_risk_z: float = 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "market_regime_shift_log": float(self.market_regime_shift_log),
            "macro_shift_annual": float(self.macro_shift_annual),
            "project_premium_shift_log": float(self.project_premium_shift_log),
            "demand_shift_log": float(self.demand_shift_log),
            "project_overrun_multiplier": float(self.project_overrun_multiplier),
            "rvz_delay_quarters": float(self.rvz_delay_quarters),
            "sales_start_shift_quarters": float(self.sales_start_shift_quarters),
            "cost_inflation_annual_shift": float(self.cost_inflation_annual_shift),
            "spread_shift_annual": float(self.spread_shift_annual),
            "key_rate_shift_annual": float(self.key_rate_shift_annual),
            "cost_timing_shift_quarters": float(self.cost_timing_shift_quarters),
            "delay_sales_penalty_log_per_quarter": float(self.delay_sales_penalty_log_per_quarter),
            "execution_risk_z": float(self.execution_risk_z),
        }


@dataclass(frozen=True)
class QuarterDriverState:
    market_quarterly_shock_log: float = 0.0
    price_residual_shock_log: float = 0.0
    seasonal_sales_shock_log: float = 0.0
    demand_quarterly_shock_log: float = 0.0
    cost_quarterly_shock_log: float = 0.0
    rate_innovation_annual: float = 0.0
    spread_quarterly_shock_annual: float = 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "market_quarterly_shock_log": float(self.market_quarterly_shock_log),
            "price_residual_shock_log": float(self.price_residual_shock_log),
            "seasonal_sales_shock_log": float(self.seasonal_sales_shock_log),
            "demand_quarterly_shock_log": float(self.demand_quarterly_shock_log),
            "cost_quarterly_shock_log": float(self.cost_quarterly_shock_log),
            "rate_innovation_annual": float(self.rate_innovation_annual),
            "spread_quarterly_shock_annual": float(self.spread_quarterly_shock_annual),
        }


@dataclass(frozen=True)
class DriverScenario:
    name: str
    config: MonteCarloDriverConfig
    path: PathDriverState
    quarter_states: Tuple[QuarterDriverState, ...]

    def validate(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if len(self.quarter_states) != horizon:
            raise ValueError(f"driver quarter states length {len(self.quarter_states)} != horizon {horizon}")

    def quarter(self, idx: int) -> QuarterDriverState:
        return self.quarter_states[idx]

    def market_modulation_enabled(self) -> bool:
        """Return True when the market block should consume driver perturbations.

        The deterministic market modes (replay / trend) intentionally ignore
        stochastic market-layer perturbations even if a driver scenario is
        attached. Other blocks can still use the same driver scenario.
        """
        mode = self.config.market_path_mode.strip().lower()
        return mode in {"stochastic", "scenario_stochastic", "macro_guided"}

    def market_log_price_modulation(self, idx: int) -> float:
        if not self.market_modulation_enabled():
            return 0.0
        q = self.quarter(idx)
        return float(
            self.path.market_regime_shift_log
            + (self.path.macro_shift_annual * (idx / 4.0))
            + q.market_quarterly_shock_log
        )

    def cache_key(self) -> Tuple[Any, ...]:
        quarter_sig = tuple(
            (
                round(q.market_quarterly_shock_log, 6),
                round(q.price_residual_shock_log, 6),
                round(q.seasonal_sales_shock_log, 6),
                round(q.demand_quarterly_shock_log, 6),
                round(q.cost_quarterly_shock_log, 6),
                round(q.rate_innovation_annual, 6),
                round(q.spread_quarterly_shock_annual, 6),
            )
            for q in self.quarter_states
        )
        return (
            self.name,
            self.config.market_path_mode.strip().lower(),
            round(self.config.market_mean_reversion, 6),
            round(self.config.market_drift_annual, 6),
            int(bool(self.config.use_correlations)),
            round(self.path.market_regime_shift_log, 6),
            round(self.path.macro_shift_annual, 6),
            round(self.path.project_premium_shift_log, 6),
            round(self.path.demand_shift_log, 6),
            round(self.path.project_overrun_multiplier, 6),
            int(self.path.rvz_delay_quarters),
            int(self.path.sales_start_shift_quarters),
            round(self.path.cost_inflation_annual_shift, 6),
            round(self.path.spread_shift_annual, 6),
            round(self.path.key_rate_shift_annual, 6),
            int(self.path.cost_timing_shift_quarters),
            round(self.path.delay_sales_penalty_log_per_quarter, 6),
            round(self.path.execution_risk_z, 6),
            quarter_sig,
        )

    def market_log_price_adjustment(self, idx: int) -> float:
        return self.market_log_price_modulation(idx)

    def price_residual_shock(self, idx: int) -> float:
        return float(self.quarter(idx).price_residual_shock_log)

    def sales_log_multiplier(self, idx: int) -> float:
        q = self.quarter(idx)
        delay_penalty = self.path.delay_sales_penalty_log_per_quarter * max(int(self.path.rvz_delay_quarters), 0)
        return float(
            self.path.demand_shift_log
            + q.demand_quarterly_shock_log
            + q.seasonal_sales_shock_log
            - delay_penalty
        )

    def sales_log_modulation(self, idx: int) -> float:
        return self.sales_log_multiplier(idx)

    def effective_key_rate_annual(self, *, base_key_rate_annual: float, idx: int) -> float:
        q = self.quarter(idx)
        return float(
            base_key_rate_annual
            + self.path.key_rate_shift_annual
            + self.path.macro_shift_annual
            + q.rate_innovation_annual
        )

    def effective_spread_annual(self, *, base_spread_annual: float, idx: int) -> float:
        q = self.quarter(idx)
        return float(base_spread_annual + self.path.spread_shift_annual + q.spread_quarterly_shock_annual)

    def effective_cost_multiplier(self, *, idx: int) -> float:
        q = self.quarter(idx)
        annual_drift = self.path.cost_inflation_annual_shift
        quarterly = q.cost_quarterly_shock_log
        return float(
            max(0.5, self.path.project_overrun_multiplier)
            * math.exp(annual_drift * (idx / 4.0) + quarterly)
        )

    def cost_multiplier(self, *, idx: int) -> float:
        return self.effective_cost_multiplier(idx=idx)

    def delay_penalty_log(self) -> float:
        return float(self.path.delay_sales_penalty_log_per_quarter * max(int(self.path.rvz_delay_quarters), 0))

    def summary(self) -> Dict[str, float]:
        market_shocks = [q.market_quarterly_shock_log for q in self.quarter_states]
        price_residuals = [q.price_residual_shock_log for q in self.quarter_states]
        sales_shocks = [q.seasonal_sales_shock_log for q in self.quarter_states]
        return {
            **self.path.summary(),
            "avg_market_quarterly_shock_log": _safe_mean(market_shocks),
            "max_abs_market_quarterly_shock_log": float(max((abs(v) for v in market_shocks), default=0.0)),
            "avg_price_residual_shock_log": _safe_mean(price_residuals),
            "avg_seasonal_sales_shock_log": _safe_mean(sales_shocks),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class DriverScenarioFactory:
    """Generates path-wise and quarter-wise stochastic driver realizations."""

    PATH_NAMES: Tuple[str, ...] = (
        "market_regime_shift_log",
        "macro_shift_annual",
        "project_premium_shift_log",
        "demand_shift_log",
        "project_overrun_log",
        "spread_shift_annual",
        "key_rate_shift_annual",
        "cost_inflation_annual_shift",
        "execution_risk_z",
    )

    QUARTER_NAMES: Tuple[str, ...] = (
        "market_quarterly_shock_log",
        "price_residual_shock_log",
        "seasonal_sales_shock_log",
        "demand_quarterly_shock_log",
        "cost_quarterly_shock_log",
        "rate_innovation_annual",
        "spread_quarterly_shock_annual",
    )

    def __init__(self, config: Optional[MonteCarloDriverConfig] = None, *, seed: Optional[int] = None) -> None:
        self.config = config or MonteCarloDriverConfig()
        self.rng = np.random.default_rng(seed)

    def make(self, *, name: str, horizon: int) -> DriverScenario:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        path = self._sample_path_state()
        quarters = tuple(self._sample_quarter_state(idx=idx, horizon=horizon, path=path) for idx in range(horizon))
        scenario = DriverScenario(name=name, config=self.config, path=path, quarter_states=quarters)
        scenario.validate(horizon)
        return scenario

    def _sample_path_state(self) -> PathDriverState:
        market_scale = self.config.market_noise_scale()
        macro_scale = self.config.macro_price_scale()

        stds = [
            self.config.market_regime_shift_std * market_scale,
            self.config.macro_shift_std * macro_scale,
            self.config.project_premium_std,
            self.config.demand_shift_std,
            self.config.overrun_log_std,
            self.config.spread_shock_std,
            self.config.key_rate_innovation_std,
            self.config.cost_inflation_std,
            1.0,  # execution risk latent
        ]
        corr = self.config.resolved_path_corr_matrix()
        latent = _sample_correlated_normals(
            self.rng,
            names=self.PATH_NAMES,
            stds=stds,
            corr_matrix=corr,
            use_correlations=self.config.use_correlations,
        )

        execution_risk_z = float(latent["execution_risk_z"])
        delay_values = np.asarray(list(self.config.rvz_delay_values), dtype=int)
        delay_probs = _normalize_probs(self.config.rvz_delay_probs)
        if len(delay_values) != len(delay_probs):
            raise ValueError("rvz_delay_values and rvz_delay_probs must have equal length")

        if len(delay_values) == 1:
            rvz_delay = int(delay_values[0])
        else:
            bias = self.config.path_delay_risk_sensitivity * execution_risk_z
            tilt = np.exp(bias * np.linspace(-1.0, 1.0, len(delay_probs)))
            probs = _normalize_probs(delay_probs * tilt)
            rvz_delay = int(self.rng.choice(delay_values, p=probs))

        cost_timing_shift = rvz_delay if self.config.apply_cost_timing_shift else 0
        cost_timing_shift = int(_clip(cost_timing_shift, -self.config.max_cost_timing_shift_quarters, self.config.max_cost_timing_shift_quarters))

        # Convert log-overrun to multiplier and keep it in a sane range.
        project_overrun_multiplier = float(_clip(math.exp(latent["project_overrun_log"]), 0.70, 1.50))

        return PathDriverState(
            market_regime_shift_log=float(latent["market_regime_shift_log"]),
            macro_shift_annual=float(latent["macro_shift_annual"]),
            project_premium_shift_log=float(latent["project_premium_shift_log"]),
            demand_shift_log=float(latent["demand_shift_log"]),
            project_overrun_multiplier=project_overrun_multiplier,
            rvz_delay_quarters=rvz_delay,
            sales_start_shift_quarters=0,
            cost_inflation_annual_shift=float(latent["cost_inflation_annual_shift"]),
            spread_shift_annual=float(latent["spread_shift_annual"]),
            key_rate_shift_annual=float(latent["key_rate_shift_annual"]),
            cost_timing_shift_quarters=cost_timing_shift,
            delay_sales_penalty_log_per_quarter=float(self.config.delay_sales_penalty_log_per_quarter),
            execution_risk_z=execution_risk_z,
        )

    def _sample_quarter_state(self, *, idx: int, horizon: int, path: PathDriverState) -> QuarterDriverState:
        market_scale = self.config.market_noise_scale()
        stds = [
            self.config.market_shock_std * market_scale,
            self.config.final_price_residual_std,
            self.config.seasonal_sales_shock_std,
            self.config.demand_shift_std * 0.35,
            self.config.cost_quarterly_shock_std,
            self.config.key_rate_innovation_std,
            self.config.spread_shock_std,
        ]
        corr = self.config.resolved_quarter_corr_matrix()
        latent = _sample_correlated_normals(
            self.rng,
            names=self.QUARTER_NAMES,
            stds=stds,
            corr_matrix=corr,
            use_correlations=self.config.use_correlations,
        )
        # Slightly damp market shocks over long horizons if the path is trend-like.
        horizon_scale = 1.0 if horizon <= 1 else 1.0 - 0.15 * (idx / max(horizon - 1, 1))
        return QuarterDriverState(
            market_quarterly_shock_log=float(latent["market_quarterly_shock_log"]) * horizon_scale,
            price_residual_shock_log=float(latent["price_residual_shock_log"]),
            seasonal_sales_shock_log=float(latent["seasonal_sales_shock_log"]),
            demand_quarterly_shock_log=float(latent["demand_quarterly_shock_log"]),
            cost_quarterly_shock_log=float(latent["cost_quarterly_shock_log"]),
            rate_innovation_annual=float(latent["rate_innovation_annual"]),
            spread_quarterly_shock_annual=float(latent["spread_quarterly_shock_annual"]),
        )


def default_driver_factory(*, seed: Optional[int] = None) -> DriverScenarioFactory:
    return DriverScenarioFactory(MonteCarloDriverConfig(), seed=seed)


# ---------------------------------------------------------------------------
# Runtime cost adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeCostSchedule:
    land_by_quarter: Tuple[float, ...]
    smr_by_quarter: Tuple[float, ...]
    design_by_quarter: Tuple[float, ...]
    other_opex_by_quarter: Tuple[float, ...]
    post_completion_cost_by_quarter: Tuple[float, ...]
    property_tax_by_quarter: Tuple[float, ...]
    vat_by_quarter: Tuple[float, ...]
    cost_index_by_quarter: Tuple[float, ...] = field(default_factory=tuple)
    project_overrun_multiplier: float = 1.0
    cost_timing_shift_quarters: int = 0
    quarterly_cost_shocks: Tuple[float, ...] = field(default_factory=tuple)

    def validate(self, horizon: int) -> None:
        for name, seq in [
            ("land_by_quarter", self.land_by_quarter),
            ("smr_by_quarter", self.smr_by_quarter),
            ("design_by_quarter", self.design_by_quarter),
            ("other_opex_by_quarter", self.other_opex_by_quarter),
            ("post_completion_cost_by_quarter", self.post_completion_cost_by_quarter),
            ("property_tax_by_quarter", self.property_tax_by_quarter),
            ("vat_by_quarter", self.vat_by_quarter),
            ("cost_index_by_quarter", self.cost_index_by_quarter),
            ("quarterly_cost_shocks", self.quarterly_cost_shocks),
        ]:
            if seq and len(seq) != horizon:
                raise ValueError(f"{name} length must equal horizon ({horizon})")

    def _value(self, seq: Sequence[float], idx: int) -> float:
        return float(seq[idx]) if seq else 0.0

    def land(self, idx: int) -> float:
        return self._value(self.land_by_quarter, idx)

    def smr(self, idx: int) -> float:
        return self._value(self.smr_by_quarter, idx)

    def design(self, idx: int) -> float:
        return self._value(self.design_by_quarter, idx)

    def other_opex(self, idx: int) -> float:
        return self._value(self.other_opex_by_quarter, idx)

    def post_completion(self, idx: int) -> float:
        return self._value(self.post_completion_cost_by_quarter, idx)

    def property_tax(self, idx: int) -> float:
        return self._value(self.property_tax_by_quarter, idx)

    def vat(self, idx: int) -> float:
        return self._value(self.vat_by_quarter, idx)


class RuntimeCostAdapter:
    """Transforms a deterministic cost schedule into a driver-aware schedule."""

    @staticmethod
    def build_effective_schedule(*, base_schedule: Any, horizon: int, scenario: Any) -> RuntimeCostSchedule:
        land = _as_float_list(getattr(base_schedule, "land_by_quarter", []), horizon=horizon)
        smr = _as_float_list(getattr(base_schedule, "smr_by_quarter", []), horizon=horizon)
        design = _as_float_list(getattr(base_schedule, "design_by_quarter", []), horizon=horizon)
        other_opex = _as_float_list(getattr(base_schedule, "other_opex_by_quarter", []), horizon=horizon)
        post_completion = _as_float_list(getattr(base_schedule, "post_completion_cost_by_quarter", []), horizon=horizon)
        property_tax = _as_float_list(getattr(base_schedule, "property_tax_by_quarter", []), horizon=horizon)
        vat = _as_float_list(getattr(base_schedule, "vat_by_quarter", []), horizon=horizon)

        driver = getattr(scenario, "driver_scenario", None)
        if driver is None:
            return RuntimeCostSchedule(
                land_by_quarter=tuple(land),
                smr_by_quarter=tuple(smr),
                design_by_quarter=tuple(design),
                other_opex_by_quarter=tuple(other_opex),
                post_completion_cost_by_quarter=tuple(post_completion),
                property_tax_by_quarter=tuple(property_tax),
                vat_by_quarter=tuple(vat),
                cost_index_by_quarter=tuple([1.0] * horizon),
                project_overrun_multiplier=1.0,
                cost_timing_shift_quarters=0,
                quarterly_cost_shocks=tuple([0.0] * horizon),
            )

        timing_shift = int(driver.path.cost_timing_shift_quarters)
        if timing_shift != 0:
            smr = _shift_series(smr, timing_shift, horizon)
            design = _shift_series(design, timing_shift, horizon)
            other_opex = _shift_series(other_opex, timing_shift, horizon)
            post_completion = _shift_series(post_completion, timing_shift, horizon)

        cost_index_path = [driver.effective_cost_multiplier(idx=i) for i in range(horizon)]
        quarterly_cost_shocks = [driver.quarter(i).cost_quarterly_shock_log for i in range(horizon)]

        adjusted_smr = [smr[i] * cost_index_path[i] for i in range(horizon)]
        adjusted_design = [design[i] * cost_index_path[i] for i in range(horizon)]
        adjusted_other = [other_opex[i] * cost_index_path[i] for i in range(horizon)]
        adjusted_post = [post_completion[i] * cost_index_path[i] for i in range(horizon)]

        return RuntimeCostSchedule(
            land_by_quarter=tuple(land),
            smr_by_quarter=tuple(adjusted_smr),
            design_by_quarter=tuple(adjusted_design),
            other_opex_by_quarter=tuple(adjusted_other),
            post_completion_cost_by_quarter=tuple(adjusted_post),
            property_tax_by_quarter=tuple(property_tax),
            vat_by_quarter=tuple(vat),
            cost_index_by_quarter=tuple(cost_index_path),
            project_overrun_multiplier=float(driver.path.project_overrun_multiplier),
            cost_timing_shift_quarters=timing_shift,
            quarterly_cost_shocks=tuple(quarterly_cost_shocks),
        )
