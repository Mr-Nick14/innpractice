from __future__ import annotations

"""
Inference-only hedonic integration for Monte Carlo cashflow simulations.

Design notes
------------
- No training inside runtime.
- All model objects are loaded from persisted artifacts.
- Feature builders support explicit inputs + automatic fallback layers.
- Spatial features (H3/kNN) are optional and degrade gracefully.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import pickle
import random
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - sklearn variant without this warning type
    InconsistentVersionWarning = None
from .mc_stochastic_drivers import DriverScenario

from .mc_cashflow_engine import (
    PriceModel,
    ProjectConfig,
    QuarterContext,
    QuarterId,
    QuarterState,
    Scenario,
    OnePathCashflowEngine,
    FittedNBFeatureSalesModel,
    CostSchedule,
)


# ---------------------------------------------------------------------------
# Constants and small helpers
# ---------------------------------------------------------------------------


DIST_COLS: Sequence[str] = (
    "dist_center_m",
    "dist_metro_m",
    "dist_bus_m",
    "dist_kindergarten_m",
    "dist_school_m",
    "dist_mall_m",
    "dist_park_m",
    "dist_rail_m",
    "dist_hospital_m",
)

GEO_LAMBDAS_MULTI: Dict[str, Sequence[int]] = {
    "dist_center_m": (5_000, 12_000, 25_000),
    "dist_metro_m": (400, 900, 2_000),
    "dist_bus_m": (150, 350, 800),
    "dist_kindergarten_m": (300, 700, 1_500),
    "dist_school_m": (400, 900, 1_800),
    "dist_mall_m": (1_000, 2_500, 5_000),
    "dist_park_m": (500, 1_200, 3_000),
    "dist_rail_m": (1_000, 2_500, 6_000),
    "dist_hospital_m": (800, 1_800, 4_000),
}

DEFAULT_DISTANCE_FALLBACKS: Dict[str, float] = {
    "dist_center_m": 14_000.0,
    "dist_metro_m": 1_200.0,
    "dist_bus_m": 500.0,
    "dist_kindergarten_m": 900.0,
    "dist_school_m": 1_000.0,
    "dist_mall_m": 2_500.0,
    "dist_park_m": 1_100.0,
    "dist_rail_m": 2_000.0,
    "dist_hospital_m": 1_800.0,
}

KNN_RADII_METERS: Sequence[int] = (500, 1_000, 2_000)
KNN_LOOKBACK_QUARTERS_DEFAULT = 6
EARTH_RADIUS_M = 6_371_000.0

_RE_QUARTER_YEAR_FIRST = re.compile(r"^\s*(\d{4})Q([1-4])\s*$")
_RE_QUARTER_Q_FIRST = re.compile(r"^\s*([1-4])Q(\d{4})\s*$")
_WARNED_MESSAGES: set[str] = set()


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _safe_log(x: float, floor: float = 1e-9) -> float:
    return math.log(max(float(x), floor))


def _normalize_quarter_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, QuarterId):
        return f"{value.year}Q{value.quarter}"
    text = str(value).strip()
    if not text:
        return None
    m = _RE_QUARTER_YEAR_FIRST.match(text)
    if m:
        return f"{int(m.group(1))}Q{int(m.group(2))}"
    m = _RE_QUARTER_Q_FIRST.match(text)
    if m:
        return f"{int(m.group(2))}Q{int(m.group(1))}"
    return None


def _quarter_label_to_order(label: str) -> int:
    norm = _normalize_quarter_label(label)
    if norm is None:
        raise ValueError(f"Invalid quarter label: {label}")
    year = int(norm[:4])
    quarter = int(norm[-1])
    return year * 4 + quarter


def _quarter_id_to_order(qid: QuarterId) -> int:
    return qid.year * 4 + qid.quarter


def _quarter_id_to_label_year_first(qid: QuarterId) -> str:
    return f"{qid.year}Q{qid.quarter}"


def _build_feature_defaults(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    defaults: Dict[str, float] = {}
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            defaults[col] = float(series.median())
    return defaults


def _warn_with_prefix(prefix: str, message: str) -> None:
    full = f"[{prefix}] {message}"
    if full in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(full)
    warnings.warn(full, RuntimeWarning, stacklevel=2)


def _patch_pickle_compatibility(obj: Any) -> int:
    """
    Applies minimal runtime compatibility patch for sklearn 1.7 artifacts loaded in 1.8+.

    In sklearn 1.8, SimpleImputer.transform expects fitted attribute `_fill_dtype`.
    Older pickles may not have it and raise:
      AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
    """
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        return 0

    patched = 0

    def _visit(node: Any) -> None:
        nonlocal patched
        if node is None:
            return

        if isinstance(node, SimpleImputer) and not hasattr(node, "_fill_dtype"):
            stats = getattr(node, "statistics_", None)
            if stats is not None:
                node._fill_dtype = np.asarray(stats).dtype
            else:
                node._fill_dtype = np.float64
            patched += 1

        if hasattr(node, "steps"):
            for _, step_obj in getattr(node, "steps", []):
                _visit(step_obj)

        if hasattr(node, "transformers"):
            for _, tr_obj, _ in getattr(node, "transformers", []):
                if tr_obj in ("drop", "passthrough"):
                    continue
                _visit(tr_obj)

        if hasattr(node, "transformers_"):
            for _, tr_obj, _ in getattr(node, "transformers_", []):
                if tr_obj in ("drop", "passthrough"):
                    continue
                _visit(tr_obj)

    _visit(obj)
    return patched


# ---------------------------------------------------------------------------
# Input payload
# ---------------------------------------------------------------------------


@dataclass
class ProjectSimulationInput:
    region_key: str
    project_name: str
    sellable_area_sqm: float
    avg_unit_area_sqm: float
    project_id: Optional[str | int] = None
    building_lat: Optional[float] = None
    building_lon: Optional[float] = None
    class_final: Optional[str] = None
    construction_type_final: Optional[str] = None
    finishing: Optional[str] = None
    lots_total: Optional[float] = None
    floor_max_pd: Optional[float] = None
    area_project_total: Optional[float] = None
    area_project_mean: Optional[float] = None
    area_project_median: Optional[float] = None
    start_sales_quarter: Optional[str] = None
    planned_rve_quarter: Optional[str] = None
    start_sales_date: Optional[str] = None
    planned_rve_date: Optional[str] = None
    developer: Optional[str] = None
    builder: Optional[str] = None
    known_geo_features: Dict[str, float] = field(default_factory=dict)
    known_quality_features: Dict[str, Any] = field(default_factory=dict)
    representative_unit: Dict[str, Any] = field(default_factory=dict)
    reference_market_metadata: Dict[str, Any] = field(default_factory=dict)
    project_age_months_at_start: Optional[float] = None
    months_to_rve_at_start: Optional[float] = None
    floor: Optional[float] = None
    rooms_num: Optional[float] = None
    area_sqm: Optional[float] = None
    ceiling_m_final: Optional[float] = None
    class_group: Optional[str] = None

    def normalized_project_id(self) -> Optional[str]:
        if self.project_id is None:
            return None
        text = str(self.project_id).strip()
        return text if text else None


# ---------------------------------------------------------------------------
# Project adapter (one-project payload from deals parquet)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectCostStubConfig:
    horizon_quarters: int = 12
    ltc: float = 0.90
    construction_cost_per_sqm: float = 120_000.0
    land_share_of_total_cost: float = 0.12
    design_share_of_total_cost: float = 0.03
    reserve_fee_annual: float = 0.005
    key_rate_annual: float = 0.21
    spread_before_rvz: float = 0.035
    spread_after_rvz: float = 0.035
    privileged_rate_annual: float = 0.037
    marketing_cost_ratio: float = 0.05
    profit_tax_rate: float = 0.25
    collateral_discount: float = 0.30
    collateral_price_sqm_ratio_to_initial: float = 0.88
    rvz_buffer_quarters: int = 1
    min_smr_quarters: int = 6


class ProjectInputAdapter:
    """
    Builds one-project simulation payload from historical deals parquet.

    Uses robust medians/modes and explicit fallbacks so the runtime remains
    inference-only and resilient to sparse project rows.
    """

    def __init__(self, deals_parquet_path: str | Path) -> None:
        self.deals_parquet_path = Path(deals_parquet_path).resolve()
        self._deals_cache: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._deals_cache is not None:
            return self._deals_cache
        df = pd.read_parquet(self.deals_parquet_path)
        if "project_id" not in df.columns or "project_name" not in df.columns:
            raise ValueError("deals parquet must contain project_id and project_name")
        self._deals_cache = df
        return df

    @staticmethod
    def _mode_or_default(series: pd.Series, default: str = "Unknown") -> str:
        non_null = series.dropna()
        if non_null.empty:
            return default
        mode_vals = non_null.mode()
        if mode_vals.empty:
            return str(non_null.iloc[0])
        return str(mode_vals.iloc[0])

    @staticmethod
    def _median_or(series: pd.Series, fallback: float) -> float:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if vals.empty:
            return float(fallback)
        return float(vals.median())

    @staticmethod
    def _normalize_class_group(value: Optional[str]) -> str:
        if value is None:
            return "Комфорт"
        key = str(value).strip().lower()
        mapping = {
            "comfort": "Комфорт",
            "business": "Бизнес",
            "premium": "Премиум",
            "economy": "Эконом",
            "комфорт": "Комфорт",
            "бизнес": "Бизнес",
            "премиум": "Премиум",
            "эконом": "Эконом",
        }
        return mapping.get(key, "Комфорт")

    @staticmethod
    def _quarter_index_from_label(label: str) -> int:
        return _quarter_label_to_order(label)

    @staticmethod
    def _label_from_quarter_order(order: int) -> str:
        year = order // 4
        quarter = order % 4
        if quarter == 0:
            year -= 1
            quarter = 4
        return f"{year}Q{quarter}"

    def list_top_projects(self, n: int = 20) -> pd.DataFrame:
        df = self._load()
        g = (
            df.groupby(["project_id", "project_name"], as_index=False)
            .size()
            .rename(columns={"size": "deals_count"})
            .sort_values("deals_count", ascending=False)
            .head(n)
        )
        return g

    def build_project_input(
        self,
        *,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        region_key: str = "Moscow",
    ) -> ProjectSimulationInput:
        if not project_id and not project_name:
            raise ValueError("Provide project_id or project_name")
        df = self._load()
        if project_id:
            mask = df["project_id"].astype(str) == str(project_id)
        else:
            mask = df["project_name"].astype(str) == str(project_name)
        sub = df.loc[mask].copy()
        if sub.empty:
            raise ValueError(f"Project not found: project_id={project_id}, project_name={project_name}")

        # Normalize quarter ordering.
        q_norm = sub["quarter"].map(_normalize_quarter_label)
        sub = sub.loc[q_norm.notna()].copy()
        sub["quarter_norm"] = q_norm.loc[q_norm.notna()]
        sub["quarter_order"] = sub["quarter_norm"].map(self._quarter_index_from_label)
        sub = sub.sort_values("quarter_order")
        if sub.empty:
            raise ValueError("Project rows do not contain valid quarter labels")

        first = sub.iloc[0]
        avg_unit = self._median_or(sub.get("area_sqm", pd.Series(dtype=float)), fallback=50.0)
        lots_total = self._median_or(sub.get("lots_total", pd.Series(dtype=float)), fallback=0.0)
        if lots_total <= 0:
            lots_total = max(round(float(sub.shape[0]) / 4.0), 50.0)
        area_project_total = self._median_or(
            sub.get("area_project_total", pd.Series(dtype=float)),
            fallback=lots_total * avg_unit,
        )
        sellable_area = area_project_total if area_project_total > 0 else lots_total * avg_unit

        class_final = self._mode_or_default(sub.get("class_final", pd.Series(dtype=object)), default="comfort")
        construction_type_final = self._mode_or_default(
            sub.get("construction_type_final", pd.Series(dtype=object)),
            default="Unknown",
        )
        finishing = self._mode_or_default(sub.get("finishing", pd.Series(dtype=object)), default="Unknown")
        class_group = self._mode_or_default(sub.get("class_group", pd.Series(dtype=object)), default=class_final)

        known_geo = {}
        for col in DIST_COLS:
            if col in sub.columns:
                known_geo[col] = self._median_or(sub[col], fallback=DEFAULT_DISTANCE_FALLBACKS.get(col, 0.0))

        start_q = str(sub["quarter_norm"].min())
        # Estimate planned RVE by the latest observed "quarter + months_to_rve/3" median.
        latest_q = int(sub["quarter_order"].max())
        months_to_rve_latest = self._median_or(
            sub.loc[sub["quarter_order"] == latest_q, "months_to_rve"] if "months_to_rve" in sub.columns else pd.Series(dtype=float),
            fallback=12.0,
        )
        planned_rve_order = latest_q + int(round(months_to_rve_latest / 3.0))
        planned_rve_q = self._label_from_quarter_order(planned_rve_order)

        project_age_start = self._median_or(
            sub.loc[sub["quarter_norm"] == start_q, "project_age_months"] if "project_age_months" in sub.columns else pd.Series(dtype=float),
            fallback=0.0,
        )
        months_to_rve_start = self._median_or(
            sub.loc[sub["quarter_norm"] == start_q, "months_to_rve"] if "months_to_rve" in sub.columns else pd.Series(dtype=float),
            fallback=max((planned_rve_order - self._quarter_index_from_label(start_q)) * 3, 0),
        )

        known_quality: Dict[str, Any] = {
            "class_final": class_final,
            "construction_type_final": construction_type_final,
            "finishing": finishing,
            "floor": self._median_or(sub.get("floor", pd.Series(dtype=float)), fallback=5.0),
            "ceiling_m_pd": self._median_or(sub.get("ceiling_m_pd", pd.Series(dtype=float)), fallback=2.8),
            "rooms": self._median_or(sub.get("rooms", pd.Series(dtype=float)), fallback=2.0),
            "floor_max_pd": self._median_or(sub.get("floor_max_pd", pd.Series(dtype=float)), fallback=16.0),
            "area_project_total": area_project_total,
            "area_project_mean": self._median_or(sub.get("area_project_mean", pd.Series(dtype=float)), fallback=avg_unit),
            "area_project_median": self._median_or(sub.get("area_project_median", pd.Series(dtype=float)), fallback=avg_unit),
        }

        building_lat = _to_float_or_none(first.get("building_lat")) if "building_lat" in sub.columns else None
        building_lon = _to_float_or_none(first.get("building_lon")) if "building_lon" in sub.columns else None

        return ProjectSimulationInput(
            region_key=region_key,
            project_id=str(first["project_id"]),
            project_name=str(first["project_name"]),
            building_lat=building_lat,
            building_lon=building_lon,
            class_final=class_final,
            construction_type_final=construction_type_final,
            finishing=finishing,
            class_group=self._normalize_class_group(class_group),
            lots_total=float(lots_total),
            sellable_area_sqm=float(sellable_area),
            avg_unit_area_sqm=float(avg_unit),
            floor_max_pd=float(known_quality["floor_max_pd"]),
            area_project_total=float(known_quality["area_project_total"]),
            area_project_mean=float(known_quality["area_project_mean"]),
            area_project_median=float(known_quality["area_project_median"]),
            start_sales_quarter=start_q,
            planned_rve_quarter=planned_rve_q,
            known_geo_features=known_geo,
            known_quality_features=known_quality,
            representative_unit={
                "area_sqm": float(avg_unit),
                "floor": float(known_quality["floor"]),
                "rooms_num": float(known_quality["rooms"]),
            },
            project_age_months_at_start=float(project_age_start),
            months_to_rve_at_start=float(months_to_rve_start),
            floor=float(known_quality["floor"]),
            rooms_num=float(known_quality["rooms"]),
            area_sqm=float(avg_unit),
            ceiling_m_final=float(known_quality["ceiling_m_pd"]),
        )

    def build_stub_project_config_and_schedule(
        self,
        *,
        project_input: ProjectSimulationInput,
        cost_stub: ProjectCostStubConfig = ProjectCostStubConfig(),
    ) -> Tuple[ProjectConfig, CostSchedule]:
        if project_input.lots_total is None or project_input.lots_total <= 0:
            initial_lots = max(int(round(project_input.sellable_area_sqm / max(project_input.avg_unit_area_sqm, 1.0))), 1)
        else:
            initial_lots = max(int(round(project_input.lots_total)), 1)

        # Infer RVZ horizon from project payload when possible.
        rvz_q_index = cost_stub.horizon_quarters - cost_stub.rvz_buffer_quarters - 1
        if project_input.months_to_rve_at_start is not None:
            inferred = int(round(float(project_input.months_to_rve_at_start) / 3.0))
            rvz_q_index = max(1, min(cost_stub.horizon_quarters - 1, inferred))

        start_label = _normalize_quarter_label(project_input.start_sales_quarter) or "2026Q1"
        start_year = int(start_label[:4])
        start_quarter = int(start_label[-1])

        initial_price = self._median_or(
            self._load().loc[self._load()["project_id"].astype(str) == str(project_input.project_id), "price_sqm"],
            fallback=180_000.0,
        )

        total_cost = max(project_input.sellable_area_sqm, 1.0) * cost_stub.construction_cost_per_sqm
        land_cost = total_cost * cost_stub.land_share_of_total_cost
        design_cost = total_cost * cost_stub.design_share_of_total_cost
        smr_cost = max(total_cost - land_cost - design_cost, total_cost * 0.6)
        collateral_price = initial_price * cost_stub.collateral_price_sqm_ratio_to_initial

        cfg = ProjectConfig(
            name=project_input.project_name,
            start_year=start_year,
            start_quarter=start_quarter,
            horizon_quarters=cost_stub.horizon_quarters,
            rns_quarter_index=0,
            rvz_quarter_index=rvz_q_index,
            sellable_area_sqm=float(project_input.sellable_area_sqm),
            avg_unit_area_sqm=float(project_input.avg_unit_area_sqm),
            initial_remaining_lots=initial_lots,
            key_rate_annual=cost_stub.key_rate_annual,
            full_rate_spread_before_rvz=cost_stub.spread_before_rvz,
            full_rate_spread_after_rvz=cost_stub.spread_after_rvz,
            privileged_rate_annual=cost_stub.privileged_rate_annual,
            reserve_fee_annual=cost_stub.reserve_fee_annual,
            ltc=cost_stub.ltc,
            land_cost_total=float(land_cost),
            smr_cost_total=float(smr_cost),
            design_cost_total=float(design_cost),
            marketing_cost_ratio=cost_stub.marketing_cost_ratio,
            profit_tax_rate=cost_stub.profit_tax_rate,
            initial_price_sqm=float(initial_price),
            collateral_discount=cost_stub.collateral_discount,
            collateral_price_sqm=float(collateral_price),
        )

        # Keep SMR horizon robust for short synthetic horizons (e.g. smoke runs).
        target_smr_q = max(cost_stub.min_smr_quarters, min(rvz_q_index + 1, max(cost_stub.horizon_quarters - 1, 1)))
        smr_quarters = max(1, min(target_smr_q, cost_stub.horizon_quarters))
        smr_series = [0.0] * cost_stub.horizon_quarters
        if smr_quarters > 0:
            per_q = smr_cost / smr_quarters
            for i in range(smr_quarters):
                smr_series[i] = per_q
        design_series = [0.0] * cost_stub.horizon_quarters
        if smr_quarters > 0:
            design_series[0] = design_cost * 0.6
            if smr_quarters > 1:
                design_series[1] = design_cost * 0.4
            else:
                design_series[0] = design_cost

        schedule = CostSchedule(
            land_by_quarter=[land_cost] + [0.0] * (cost_stub.horizon_quarters - 1),
            smr_by_quarter=smr_series,
            design_by_quarter=design_series,
            other_opex_by_quarter=[2_000_000.0] * cost_stub.horizon_quarters,
            post_completion_cost_by_quarter=[0.0] * (cost_stub.horizon_quarters - 1) + [3_000_000.0],
        )
        return cfg, schedule


# ---------------------------------------------------------------------------
# Artifact container
# ---------------------------------------------------------------------------


@dataclass
class HedonicArtifacts:
    artifact_dir: Path
    region_key: str
    strict: bool = False

    geo_model: Any = None
    quality_model: Any = None
    pca_geo: Optional[Dict[str, Any]] = None
    h3_encoder: Optional[Dict[str, Any]] = None
    lgb_residual_model: Any = None
    lgb_features_num: List[str] = field(default_factory=list)
    lgb_features_cat: List[str] = field(default_factory=list)
    final_model_pipeline: Any = None
    project_premium_table: Optional[pd.DataFrame] = None
    market_index_table: Optional[pd.DataFrame] = None
    reference_deals: Optional[pd.DataFrame] = None
    feature_defaults: Dict[str, float] = field(default_factory=dict)
    warnings_log: List[str] = field(default_factory=list)

    @classmethod
    def load_from_dir(
        cls,
        *,
        artifact_dir: str | Path,
        region_key: str,
        strict: bool = False,
        reference_deals_path: Optional[str | Path] = None,
    ) -> "HedonicArtifacts":
        obj = cls(
            artifact_dir=Path(artifact_dir).resolve(),
            region_key=region_key,
            strict=strict,
        )
        obj.load_geo_model()
        obj.load_quality_model()
        obj.load_pca_geo()
        obj.load_h3_encoder()
        obj.load_lgb_model()
        obj.load_project_premium()
        obj.load_market_index()
        if reference_deals_path is not None:
            obj.load_reference_deals(reference_deals_path)
        return obj

    def _record_warning(self, message: str) -> None:
        self.warnings_log.append(message)
        _warn_with_prefix("HedonicArtifacts", message)

    def _pickle_path(self, stem: str) -> Path:
        return self.artifact_dir / f"{stem}_{self.region_key}.pkl"

    def _parquet_path(self, stem: str) -> Path:
        return self.artifact_dir / f"{stem}_{self.region_key}.parquet"

    def _load_pickle_optional(self, path: Path, label: str) -> Any:
        if not path.exists():
            self._record_warning(f"{label} is missing: {path}")
            return None
        if path.stat().st_size == 0:
            self._record_warning(f"{label} is empty: {path}")
            return None
        try:
            with path.open("rb") as fh:
                with warnings.catch_warnings():
                    if InconsistentVersionWarning is not None:
                        warnings.simplefilter("ignore", InconsistentVersionWarning)
                    loaded = pickle.load(fh)
            patched_cnt = _patch_pickle_compatibility(loaded)
            if patched_cnt > 0:
                self._record_warning(
                    f"{label} loaded with sklearn compatibility patch: restored _fill_dtype for {patched_cnt} SimpleImputer nodes."
                )
            return loaded
        except Exception as exc:
            msg = f"failed to load {label} from {path}: {exc}"
            if self.strict:
                raise RuntimeError(msg) from exc
            self._record_warning(msg)
            return None

    def load_geo_model(self) -> Any:
        self.geo_model = self._load_pickle_optional(self._pickle_path("geo_model"), "geo_model")
        return self.geo_model

    def load_quality_model(self) -> Any:
        self.quality_model = self._load_pickle_optional(self._pickle_path("quality_model"), "quality_model")
        return self.quality_model

    def load_pca_geo(self) -> Optional[Dict[str, Any]]:
        obj = self._load_pickle_optional(self._pickle_path("pca_geo"), "pca_geo")
        self.pca_geo = obj if isinstance(obj, dict) else None
        if obj is not None and not isinstance(obj, dict):
            self._record_warning("pca_geo artifact has unexpected type; expected dict with pca/scaler/input_cols")
        return self.pca_geo

    def load_h3_encoder(self) -> Optional[Dict[str, Any]]:
        obj = self._load_pickle_optional(self._pickle_path("h3_encoder"), "h3_encoder")
        self.h3_encoder = obj if isinstance(obj, dict) else None
        if obj is not None and not isinstance(obj, dict):
            self._record_warning("h3_encoder artifact has unexpected type; expected dict")
        return self.h3_encoder

    def load_lgb_model(self) -> None:
        lgb_obj = self._load_pickle_optional(self._pickle_path("lgb_residual_model"), "lgb_residual_model")
        if isinstance(lgb_obj, dict) and "model" in lgb_obj:
            self.lgb_residual_model = lgb_obj.get("model")
            self.lgb_features_num = list(lgb_obj.get("features_num") or [])
            self.lgb_features_cat = list(lgb_obj.get("features_cat") or [])
            return
        if lgb_obj is not None:
            self._record_warning("lgb_residual_model artifact is not a dict with model/features schema")

        final_obj = self._load_pickle_optional(self._pickle_path("final_model"), "final_model")
        if final_obj is not None:
            self.final_model_pipeline = final_obj
            self._record_warning(
                "lgb_residual_model artifact not found; using final_model pipeline fallback for final log-price."
            )

    def load_project_premium(self) -> Optional[pd.DataFrame]:
        path = self._parquet_path("project_premium")
        if not path.exists() or path.stat().st_size == 0:
            self._record_warning(f"project_premium table is missing: {path}")
            self.project_premium_table = None
            return None
        try:
            self.project_premium_table = pd.read_parquet(path)
            return self.project_premium_table
        except Exception as exc:
            msg = f"failed to load project_premium table: {exc}"
            if self.strict:
                raise RuntimeError(msg) from exc
            self._record_warning(msg)
            self.project_premium_table = None
            return None

    def load_market_index(self) -> Optional[pd.DataFrame]:
        path = self._parquet_path("market_index")
        if not path.exists() or path.stat().st_size == 0:
            self._record_warning(f"market_index table is missing: {path}")
            self.market_index_table = None
            return None
        try:
            df = pd.read_parquet(path)
            if "quarter" not in df.columns or "market_log_price" not in df.columns:
                self._record_warning("market_index table missing required columns quarter/market_log_price")
                self.market_index_table = None
                return None
            out = df.copy()
            out["quarter_norm"] = out["quarter"].map(_normalize_quarter_label)
            out = out.dropna(subset=["quarter_norm", "market_log_price"]).copy()
            out["quarter_order"] = out["quarter_norm"].map(_quarter_label_to_order)
            out = out.sort_values("quarter_order").drop_duplicates("quarter_norm", keep="last")
            self.market_index_table = out.reset_index(drop=True)
            return self.market_index_table
        except Exception as exc:
            msg = f"failed to load market_index table: {exc}"
            if self.strict:
                raise RuntimeError(msg) from exc
            self._record_warning(msg)
            self.market_index_table = None
            return None

    def load_reference_deals(self, path: str | Path) -> Optional[pd.DataFrame]:
        p = Path(path).resolve()
        if not p.exists():
            self._record_warning(f"reference deals file does not exist: {p}")
            self.reference_deals = None
            return None
        try:
            df = pd.read_parquet(p)
            self.reference_deals = df
            self.feature_defaults = _build_feature_defaults(df)
            return df
        except Exception as exc:
            msg = f"failed to load reference deals from {p}: {exc}"
            if self.strict:
                raise RuntimeError(msg) from exc
            self._record_warning(msg)
            self.reference_deals = None
            return None


# ---------------------------------------------------------------------------
# Market path model
# ---------------------------------------------------------------------------


@dataclass
class MarketPathModel:
    mode: str = "stochastic"
    market_index_table: Optional[pd.DataFrame] = None
    trend_window_quarters: int = 12
    damping_phi: float = 0.85
    stochastic_vol: float = 0.01
    mean_reversion: float = 0.30
    seed: Optional[int] = None

    _cached_path: Dict[int, float] = field(default_factory=dict, init=False)
    _cached_signature: Optional[Tuple[Any, ...]] = field(default=None, init=False)
    _rng: random.Random = field(default_factory=random.Random, init=False)

    def __post_init__(self) -> None:
        if self.seed is not None:
            self._rng = random.Random(self.seed)

    def initialize_path(self, *, config: ProjectConfig, scenario: Scenario) -> None:
        driver = getattr(scenario, "driver_scenario", None)
        market_driver_active = bool(driver is not None and driver.market_modulation_enabled())
        signature = (
            config.start_year,
            config.start_quarter,
            config.horizon_quarters,
            scenario.global_state.name,
            driver.cache_key() if driver is not None else None,
        )
        if signature == self._cached_signature and self._cached_path:
            return

        self._cached_path = {}
        history = self.market_index_table
        hist_lookup: Dict[str, float] = {}
        history_orders: List[int] = []
        history_values: List[float] = []
        if history is not None and not history.empty:
            hist_lookup = dict(zip(history["quarter_norm"], history["market_log_price"]))
            history_orders = history["quarter_order"].astype(int).tolist()
            history_values = history["market_log_price"].astype(float).tolist()

        slope = 0.0
        intercept = 0.0
        if history_orders:
            x = np.asarray(history_orders[-self.trend_window_quarters :], dtype=float)
            y = np.asarray(history_values[-self.trend_window_quarters :], dtype=float)
            if len(x) >= 2 and np.var(x) > 0:
                coeffs = np.polyfit(x, y, deg=1)
                slope = float(coeffs[0])
                intercept = float(coeffs[1])
            else:
                intercept = float(y[-1])

        shift_log = _safe_log(scenario.global_state.price_level_multiplier) if scenario else 0.0
        start_qid = QuarterId(config.start_year, config.start_quarter)
        prev_val: Optional[float] = None
        mean_reversion = self.mean_reversion
        market_drift_annual = 0.0
        if market_driver_active and driver is not None:
            mean_reversion = float(driver.config.market_mean_reversion)
            market_drift_annual = float(driver.config.market_drift_annual)

        for idx in range(config.horizon_quarters):
            qid = QuarterId(start_qid.year, start_qid.quarter)
            for _ in range(idx):
                qid = qid.next()
            label = _quarter_id_to_label_year_first(qid)
            order = _quarter_id_to_order(qid)
            trend_val = intercept + slope * order if history_orders else _safe_log(max(config.initial_price_sqm, 1.0))
            trend_val += market_drift_annual * (idx / 4.0)
            driver_adjust = float(driver.market_log_price_modulation(idx)) if market_driver_active and driver is not None else 0.0

            if self.mode == "replay":
                base = hist_lookup.get(label, trend_val)
            elif self.mode == "trend":
                base = trend_val
            elif self.mode in {"stochastic", "scenario_stochastic"}:
                local_shock = scenario.local_path.price_shock(idx) + driver_adjust
                if prev_val is None:
                    base = hist_lookup.get(label, trend_val) + local_shock
                else:
                    mean_target = trend_val
                    noise = self._rng.gauss(0.0, self.stochastic_vol)
                    base = prev_val + mean_reversion * (mean_target - prev_val) + local_shock + noise
            elif self.mode == "macro_guided":
                base = trend_val + driver_adjust
            else:
                raise ValueError(f"Unknown market mode: {self.mode}")

            value = float(base + shift_log)
            self._cached_path[idx] = value
            prev_val = value

        self._cached_signature = signature

    def get_market_log_price(self, *, ctx: QuarterContext, config: ProjectConfig, scenario: Scenario) -> float:
        self.initialize_path(config=config, scenario=scenario)
        if ctx.index not in self._cached_path:
            raise KeyError(f"Missing market path value for quarter index {ctx.index}")
        return float(self._cached_path[ctx.index])


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------


@dataclass
class GeoFeatureBuilder:
    feature_defaults: Dict[str, float] = field(default_factory=dict)

    def _resolve_feature(self, key: str, project_input: ProjectSimulationInput) -> float:
        if key in project_input.known_geo_features:
            return float(project_input.known_geo_features[key])
        if key in project_input.known_quality_features:
            value = _to_float_or_none(project_input.known_quality_features.get(key))
            if value is not None:
                return value
        direct = _to_float_or_none(getattr(project_input, key, None))
        if direct is not None:
            return direct
        if key in self.feature_defaults:
            return float(self.feature_defaults[key])
        return float(DEFAULT_DISTANCE_FALLBACKS.get(key, 0.0))

    def build_distance_features(self, project_input: ProjectSimulationInput) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for col in DIST_COLS:
            out[col] = self._resolve_feature(col, project_input)
        return out

    def build_log_distance_features(self, dist_features: Dict[str, float]) -> Dict[str, float]:
        return {f"log1p_{k}": math.log1p(max(v, 0.0)) for k, v in dist_features.items()}

    def build_accessibility_features(self, dist_features: Dict[str, float]) -> Dict[str, float]:
        acc: Dict[str, float] = {}
        for col, lambdas in GEO_LAMBDAS_MULTI.items():
            distance = max(dist_features.get(col, 0.0), 0.0)
            base = col.replace("dist_", "").replace("_m", "")
            for lam in lambdas:
                acc[f"acc_{base}_l{lam}"] = float(math.exp(-distance / float(lam)))
        acc["acc_transit"] = float(np.mean([acc["acc_metro_l900"], acc["acc_bus_l350"], acc["acc_rail_l2500"]]))
        acc["acc_family"] = float(
            np.mean([acc["acc_kindergarten_l700"], acc["acc_school_l900"], acc["acc_hospital_l1800"]])
        )
        acc["acc_retail_green"] = float(np.mean([acc["acc_mall_l2500"], acc["acc_park_l1200"]]))
        acc["acc_centrality"] = float(acc["acc_center_l12000"])
        return acc

    def build_geo_pca_features(
        self,
        *,
        base_features: Dict[str, float],
        pca_geo_artifact: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not pca_geo_artifact:
            return {f"geo_pc{i}": 0.0 for i in range(1, 6)}
        pca = pca_geo_artifact.get("pca")
        scaler = pca_geo_artifact.get("scaler")
        input_cols = list(pca_geo_artifact.get("input_cols") or [])
        if pca is None or scaler is None or not input_cols:
            return {f"geo_pc{i}": 0.0 for i in range(1, 6)}
        x_df = pd.DataFrame(
            [{col: float(base_features.get(col, self.feature_defaults.get(col, 0.0))) for col in input_cols}],
            columns=input_cols,
        )
        try:
            x_scaled = scaler.transform(x_df)
            pcs = pca.transform(x_scaled)[0]
            out = {f"geo_pc{i + 1}": float(pcs[i]) for i in range(min(5, len(pcs)))}
            for i in range(len(out) + 1, 6):
                out[f"geo_pc{i}"] = 0.0
            return out
        except Exception as exc:
            _warn_with_prefix("GeoFeatureBuilder", f"PCA transform failed, using zero geo_pc features: {exc}")
            return {f"geo_pc{i}": 0.0 for i in range(1, 6)}

    def build_geo_feature_frame(
        self,
        *,
        project_input: ProjectSimulationInput,
        pca_geo_artifact: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        dist = self.build_distance_features(project_input)
        log_dist = self.build_log_distance_features(dist)
        acc = self.build_accessibility_features(dist)
        pca_feats = self.build_geo_pca_features(base_features={**dist, **log_dist, **acc}, pca_geo_artifact=pca_geo_artifact)
        merged = {**dist, **log_dist, **acc, **pca_feats}
        return pd.DataFrame([merged])


@dataclass
class QualityFeatureBuilder:
    feature_defaults: Dict[str, float] = field(default_factory=dict)

    def _project_age_months(
        self,
        *,
        project_input: ProjectSimulationInput,
        ctx: QuarterContext,
    ) -> float:
        if project_input.project_age_months_at_start is not None:
            return float(project_input.project_age_months_at_start + 3 * ctx.index)
        if project_input.start_sales_quarter:
            start = _normalize_quarter_label(project_input.start_sales_quarter)
            if start:
                age_q = _quarter_id_to_order(ctx.quarter_id) - _quarter_label_to_order(start)
                return float(max(age_q, 0) * 3.0)
        return float(self.feature_defaults.get("project_age_months", max(ctx.index, 0) * 3.0))

    def _months_to_rve(
        self,
        *,
        project_input: ProjectSimulationInput,
        ctx: QuarterContext,
        config: ProjectConfig,
    ) -> float:
        if project_input.months_to_rve_at_start is not None:
            return float(project_input.months_to_rve_at_start - 3 * ctx.index)
        if project_input.planned_rve_quarter:
            planned = _normalize_quarter_label(project_input.planned_rve_quarter)
            if planned:
                remain_q = _quarter_label_to_order(planned) - _quarter_id_to_order(ctx.quarter_id)
                return float(remain_q * 3.0)
        remain_q = config.rvz_quarter_index - ctx.index
        return float(remain_q * 3.0)

    def build_time_varying_quality_features(
        self,
        *,
        project_input: ProjectSimulationInput,
        ctx: QuarterContext,
        config: ProjectConfig,
    ) -> Dict[str, Any]:
        known = project_input.known_quality_features
        rep = project_input.representative_unit

        area_sqm = _to_float_or_none(rep.get("area_sqm")) or _to_float_or_none(project_input.area_sqm) or float(config.avg_unit_area_sqm)
        floor = _to_float_or_none(rep.get("floor")) or _to_float_or_none(project_input.floor) or float(self.feature_defaults.get("floor", 5.0))
        floor_max = _to_float_or_none(project_input.floor_max_pd) or float(self.feature_defaults.get("floor_max_pd", max(floor, 10.0)))
        rooms_num = _to_float_or_none(rep.get("rooms_num")) or _to_float_or_none(project_input.rooms_num) or float(
            self.feature_defaults.get("rooms", 2.0)
        )
        ceiling = _to_float_or_none(project_input.ceiling_m_final) or float(self.feature_defaults.get("ceiling_m_pd", 2.8))

        lots_total = _to_float_or_none(project_input.lots_total) or float(config.initial_remaining_lots)
        area_project_total = _to_float_or_none(project_input.area_project_total) or float(
            self.feature_defaults.get("area_project_total", config.sellable_area_sqm)
        )
        area_project_mean = _to_float_or_none(project_input.area_project_mean) or float(
            self.feature_defaults.get("area_project_mean", config.avg_unit_area_sqm)
        )
        area_project_median = _to_float_or_none(project_input.area_project_median) or float(
            self.feature_defaults.get("area_project_median", config.avg_unit_area_sqm)
        )

        floor_rel = float(floor / floor_max) if floor_max > 0 else 0.0
        out: Dict[str, Any] = {
            "area_sqm": float(area_sqm),
            "floor": float(floor),
            "floor_rel": float(floor_rel),
            "rooms_num": float(rooms_num),
            "ceiling_m_final": float(ceiling),
            "project_age_months": self._project_age_months(project_input=project_input, ctx=ctx),
            "months_to_rve": self._months_to_rve(project_input=project_input, ctx=ctx, config=config),
            "lots_total": float(lots_total),
            "area_project_total": float(area_project_total),
            "area_project_mean": float(area_project_mean),
            "area_project_median": float(area_project_median),
            "class_final": project_input.class_final or known.get("class_final") or "Unknown",
            "construction_type_final": project_input.construction_type_final
            or known.get("construction_type_final")
            or "Unknown",
            "finishing": project_input.finishing or known.get("finishing") or "Unknown",
        }
        for key, val in known.items():
            if key not in out:
                out[key] = val
        return out


@dataclass
class SpatialContextBuilder:
    project_input: ProjectSimulationInput
    feature_defaults: Dict[str, float] = field(default_factory=dict)
    knn_radii_m: Sequence[int] = KNN_RADII_METERS
    knn_lookback_quarters: int = KNN_LOOKBACK_QUARTERS_DEFAULT

    _prepared: bool = field(default=False, init=False)
    _has_knn: bool = field(default=False, init=False)
    _reference: Optional[pd.DataFrame] = field(default=None, init=False)
    _coords_rad: Optional[np.ndarray] = field(default=None, init=False)
    _quarter_orders: Optional[np.ndarray] = field(default=None, init=False)
    _log_prices: Optional[np.ndarray] = field(default=None, init=False)
    _tree: Optional[BallTree] = field(default=None, init=False)
    _neighbor_indices: Optional[np.ndarray] = field(default=None, init=False)
    _neighbor_dist_m: Optional[np.ndarray] = field(default=None, init=False)
    _h3_warned: bool = field(default=False, init=False)

    def prepare(self, reference_deals: Optional[pd.DataFrame]) -> None:
        self._prepared = True
        self._reference = reference_deals
        if reference_deals is None or reference_deals.empty:
            self._has_knn = False
            return
        required = {"building_lat", "building_lon", "log_price_sqm"}
        if not required.issubset(set(reference_deals.columns)):
            self._has_knn = False
            return
        valid = (
            reference_deals["building_lat"].notna()
            & reference_deals["building_lon"].notna()
            & reference_deals["log_price_sqm"].notna()
        )
        ref = reference_deals.loc[valid].copy()
        if ref.empty:
            self._has_knn = False
            return
        if "quarter" in ref.columns:
            ref["quarter_norm"] = ref["quarter"].map(_normalize_quarter_label)
            ref = ref.dropna(subset=["quarter_norm"])
            ref["quarter_order"] = ref["quarter_norm"].map(_quarter_label_to_order)
        elif "deal_date" in ref.columns:
            dates = pd.to_datetime(ref["deal_date"], errors="coerce")
            ref = ref.loc[dates.notna()].copy()
            q_period = dates.dt.to_period("Q")
            ref["quarter_order"] = q_period.dt.year * 4 + q_period.dt.quarter
        else:
            self._has_knn = False
            return
        if ref.empty:
            self._has_knn = False
            return
        self._reference = ref
        self._coords_rad = np.radians(ref[["building_lat", "building_lon"]].to_numpy(dtype=float))
        self._quarter_orders = ref["quarter_order"].to_numpy(dtype=int)
        self._log_prices = ref["log_price_sqm"].to_numpy(dtype=float)
        self._tree = BallTree(self._coords_rad, metric="haversine", leaf_size=40)
        self._cache_neighbors()
        self._has_knn = self._neighbor_indices is not None

    def _cache_neighbors(self) -> None:
        lat = _to_float_or_none(self.project_input.building_lat)
        lon = _to_float_or_none(self.project_input.building_lon)
        if lat is None or lon is None or self._tree is None:
            self._neighbor_indices = None
            self._neighbor_dist_m = None
            return
        point = np.radians(np.array([[lat, lon]], dtype=float))
        max_r = max(self.knn_radii_m) / EARTH_RADIUS_M
        idx, dist = self._tree.query_radius(point, r=max_r, return_distance=True, sort_results=False)
        if len(idx) == 0:
            self._neighbor_indices = None
            self._neighbor_dist_m = None
            return
        self._neighbor_indices = idx[0]
        self._neighbor_dist_m = dist[0] * EARTH_RADIUS_M

    def _ensure_prepared(self) -> None:
        if not self._prepared:
            self.prepare(None)

    def build_h3_features(self, h3_encoder: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not h3_encoder:
            return {}
        resolutions = list(h3_encoder.get("resolutions") or [])
        global_mean = float(h3_encoder.get("global_mean", 0.0))
        out: Dict[str, float] = {}
        lat = _to_float_or_none(self.project_input.building_lat)
        lon = _to_float_or_none(self.project_input.building_lon)
        if lat is None or lon is None:
            for res in resolutions:
                out[f"h3_enc_r{res}"] = global_mean
            return out
        try:
            import h3 as h3lib  # type: ignore
        except Exception:
            if not self._h3_warned:
                _warn_with_prefix("SpatialContextBuilder", "python package 'h3' not installed; using H3 global_mean fallback")
                self._h3_warned = True
            for res in resolutions:
                out[f"h3_enc_r{res}"] = global_mean
            return out
        encodings = h3_encoder.get("encodings") or {}
        for res in resolutions:
            cell = h3lib.latlng_to_cell(float(lat), float(lon), int(res))
            mapping = encodings.get(res) or encodings.get(str(res))
            if isinstance(mapping, pd.Series):
                out[f"h3_enc_r{res}"] = float(mapping.get(cell, global_mean))
            elif isinstance(mapping, dict):
                out[f"h3_enc_r{res}"] = float(mapping.get(cell, global_mean))
            else:
                out[f"h3_enc_r{res}"] = global_mean
        return out

    def build_knn_features(self, *, ctx: QuarterContext) -> Dict[str, float]:
        self._ensure_prepared()
        defaults: Dict[str, float] = {}
        default_log = float(self.feature_defaults.get("log_price_sqm", 0.0))
        for r in self.knn_radii_m:
            defaults[f"knn_count_{r}m"] = 0.0
            defaults[f"knn_mean_{r}m"] = default_log
            defaults[f"knn_std_{r}m"] = 0.0
        if not self._has_knn:
            return defaults
        if self._neighbor_indices is None or self._neighbor_dist_m is None:
            return defaults
        if self._quarter_orders is None or self._log_prices is None:
            return defaults

        current_order = _quarter_id_to_order(ctx.quarter_id)
        lower_order = current_order - max(1, self.knn_lookback_quarters)
        idx = self._neighbor_indices
        dist = self._neighbor_dist_m
        q = self._quarter_orders[idx]
        p = self._log_prices[idx]
        valid_time = (q < current_order) & (q >= lower_order)

        out = defaults.copy()
        for r in self.knn_radii_m:
            m = valid_time & (dist <= float(r))
            if not np.any(m):
                continue
            vals = p[m]
            out[f"knn_count_{r}m"] = float(vals.size)
            out[f"knn_mean_{r}m"] = float(np.mean(vals))
            out[f"knn_std_{r}m"] = float(np.std(vals)) if vals.size > 1 else 0.0
        return out

    def build_spatial_features(
        self,
        *,
        ctx: QuarterContext,
        h3_encoder: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        return {
            **self.build_knn_features(ctx=ctx),
            **self.build_h3_features(h3_encoder),
        }


@dataclass
class HedonicFeatureBuilder:
    geo_builder: GeoFeatureBuilder
    quality_builder: QualityFeatureBuilder
    spatial_builder: SpatialContextBuilder

    def build_static_project_features(
        self,
        *,
        project_input: ProjectSimulationInput,
        pca_geo_artifact: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        frame = self.geo_builder.build_geo_feature_frame(project_input=project_input, pca_geo_artifact=pca_geo_artifact)
        return {k: float(v) for k, v in frame.iloc[0].to_dict().items()}

    def build_time_varying_project_features(
        self,
        *,
        project_input: ProjectSimulationInput,
        ctx: QuarterContext,
        config: ProjectConfig,
    ) -> Dict[str, Any]:
        return self.quality_builder.build_time_varying_quality_features(project_input=project_input, ctx=ctx, config=config)

    def build_market_path_features(self, *, market_log_price: float) -> Dict[str, float]:
        return {
            "market_log_price": float(market_log_price),
            "market_price_sqm": float(math.exp(market_log_price)),
        }

    def build_spatial_context_features(
        self,
        *,
        ctx: QuarterContext,
        h3_encoder: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        return self.spatial_builder.build_spatial_features(ctx=ctx, h3_encoder=h3_encoder)


# ---------------------------------------------------------------------------
# Hedonic PriceModel implementation
# ---------------------------------------------------------------------------


@dataclass
class HedonicPriceModel(PriceModel):
    project_input: ProjectSimulationInput
    artifacts: HedonicArtifacts
    market_model: MarketPathModel
    feature_builder: HedonicFeatureBuilder
    enable_geo: bool = True
    enable_quality: bool = True
    enable_lgb: bool = True

    _static_geo_features: Dict[str, float] = field(default_factory=dict, init=False)
    _cached_geo_score: Optional[float] = field(default=None, init=False)
    _project_premium: float = field(default=0.0, init=False)
    _precomputed: bool = field(default=False, init=False)
    _last_components: Dict[str, float] = field(default_factory=dict, init=False)
    _last_missing_lgb_cols: List[str] = field(default_factory=list, init=False)
    _last_source_map: Dict[str, str] = field(default_factory=dict, init=False)
    _final_model_feature_cols: List[str] = field(default_factory=list, init=False)
    _final_model_disabled: bool = field(default=False, init=False)

    @classmethod
    def from_artifacts(
        cls,
        *,
        project_input: ProjectSimulationInput,
        artifact_dir: str | Path,
        reference_deals_path: Optional[str | Path] = None,
        market_mode: str = "stochastic",
        market_seed: Optional[int] = None,
        strict_artifacts: bool = False,
    ) -> "HedonicPriceModel":
        artifacts = HedonicArtifacts.load_from_dir(
            artifact_dir=artifact_dir,
            region_key=project_input.region_key,
            strict=strict_artifacts,
            reference_deals_path=reference_deals_path,
        )
        market_model = MarketPathModel(
            mode=market_mode,
            market_index_table=artifacts.market_index_table,
            seed=market_seed,
        )
        geo_builder = GeoFeatureBuilder(feature_defaults=artifacts.feature_defaults)
        quality_builder = QualityFeatureBuilder(feature_defaults=artifacts.feature_defaults)
        spatial_builder = SpatialContextBuilder(
            project_input=project_input,
            feature_defaults=artifacts.feature_defaults,
        )
        feature_builder = HedonicFeatureBuilder(
            geo_builder=geo_builder,
            quality_builder=quality_builder,
            spatial_builder=spatial_builder,
        )
        model = cls(
            project_input=project_input,
            artifacts=artifacts,
            market_model=market_model,
            feature_builder=feature_builder,
            enable_geo=artifacts.geo_model is not None,
            enable_quality=artifacts.quality_model is not None,
            enable_lgb=(artifacts.lgb_residual_model is not None or artifacts.final_model_pipeline is not None),
        )
        return model

    def precompute_static_components(self) -> None:
        if self._precomputed:
            return
        self.feature_builder.spatial_builder.prepare(self.artifacts.reference_deals)
        self._static_geo_features = self.feature_builder.build_static_project_features(
            project_input=self.project_input,
            pca_geo_artifact=self.artifacts.pca_geo,
        )
        self._project_premium = self._resolve_project_premium()
        self._cached_geo_score = None
        if self.enable_geo and self.artifacts.geo_model is not None:
            self._cached_geo_score = self._predict_geo_score(self._static_geo_features)
        self._final_model_feature_cols = self._extract_final_model_feature_cols(self.artifacts.final_model_pipeline)
        self._precomputed = True

    def _resolve_project_premium(self) -> float:
        table = self.artifacts.project_premium_table
        if table is None or table.empty:
            return 0.0
        project_id = self.project_input.normalized_project_id()
        if project_id and "project_id" in table.columns:
            m = table.loc[table["project_id"].astype(str) == project_id, "project_premium"]
            if not m.empty and pd.notna(m.iloc[0]):
                return float(m.iloc[0])
        if "project_name" in table.columns:
            m = table.loc[table["project_name"].astype(str) == str(self.project_input.project_name), "project_premium"]
            if not m.empty and pd.notna(m.iloc[0]):
                return float(m.iloc[0])
        return 0.0

    def _predict_geo_score(self, geo_features: Dict[str, float]) -> float:
        if self.artifacts.geo_model is None:
            return 0.0
        try:
            df = pd.DataFrame([geo_features])
            pred = self.artifacts.geo_model.predict(df)
            return float(pred[0])
        except Exception as exc:
            _warn_with_prefix("HedonicPriceModel", f"geo_model.predict failed, using geo_score=0: {exc}")
            return 0.0

    def _predict_quality_score(self, quality_features: Dict[str, Any]) -> float:
        if self.artifacts.quality_model is None or not self.enable_quality:
            return 0.0
        try:
            df = pd.DataFrame([quality_features])
            pred = self.artifacts.quality_model.predict(df)
            return float(pred[0])
        except Exception as exc:
            _warn_with_prefix("HedonicPriceModel", f"quality_model.predict failed, using quality_score=0: {exc}")
            return 0.0

    def _extract_final_model_feature_cols(self, model: Any) -> List[str]:
        if model is None:
            return []
        try:
            prep = model.named_steps.get("prep")
            cols: List[str] = []
            for _, _, tr_cols in getattr(prep, "transformers", []):
                cols.extend(list(tr_cols))
            return cols
        except Exception:
            return []

    def _prepare_lgb_input_row(self, base: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        feature_order = list(self.artifacts.lgb_features_num) + list(self.artifacts.lgb_features_cat)
        missing: List[str] = []
        if not feature_order:
            return pd.DataFrame([base]), missing
        row: Dict[str, Any] = {}
        for col in feature_order:
            if col in base:
                row[col] = base[col]
            elif col in self.artifacts.feature_defaults:
                row[col] = self.artifacts.feature_defaults[col]
                missing.append(col)
            elif col in self.artifacts.lgb_features_cat:
                row[col] = "Unknown"
                missing.append(col)
            else:
                row[col] = 0.0
                missing.append(col)
        df = pd.DataFrame([row], columns=feature_order)
        for col in self.artifacts.lgb_features_cat:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df, missing

    def _predict_lgb_boost(
        self,
        *,
        additive_features: Dict[str, Any],
        additive_log_price: float,
    ) -> Tuple[float, List[str]]:
        if not self.enable_lgb:
            return 0.0, []
        if self.artifacts.lgb_residual_model is not None:
            df, missing = self._prepare_lgb_input_row(additive_features)
            try:
                pred = self.artifacts.lgb_residual_model.predict(df)
                return float(pred[0]), missing
            except Exception as exc:
                _warn_with_prefix("HedonicPriceModel", f"lgb residual predict failed, using boost=0: {exc}")
                return 0.0, missing
        if self.artifacts.final_model_pipeline is not None:
            if self._final_model_disabled:
                return 0.0, []
            cols = self._final_model_feature_cols
            if not cols:
                return 0.0, []
            row: Dict[str, Any] = {}
            missing: List[str] = []
            for col in cols:
                if col in additive_features:
                    row[col] = additive_features[col]
                elif col in self.artifacts.feature_defaults:
                    row[col] = self.artifacts.feature_defaults[col]
                    missing.append(col)
                else:
                    row[col] = "Unknown" if isinstance(additive_features.get(col), str) else 0.0
                    missing.append(col)
            df = pd.DataFrame([row], columns=cols)
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype("category")
            try:
                final_log = float(self.artifacts.final_model_pipeline.predict(df)[0])
                return final_log - additive_log_price, missing
            except Exception as exc:
                _warn_with_prefix("HedonicPriceModel", f"final_model fallback predict failed, using boost=0: {exc}")
                self._final_model_disabled = True
                return 0.0, missing
        return 0.0, []

    def predict_log_price_sqm(
        self,
        *,
        ctx: QuarterContext,
        config: ProjectConfig,
        scenario: Scenario,
        previous_state: Optional[QuarterState],
    ) -> float:
        self.precompute_static_components()
        market_log = self.market_model.get_market_log_price(ctx=ctx, config=config, scenario=scenario)
        market_payload = self.feature_builder.build_market_path_features(market_log_price=market_log)
        driver = getattr(scenario, "driver_scenario", None)
        market_driver_active = bool(driver is not None and driver.market_modulation_enabled())
        driver_q = driver.quarter(ctx.index) if driver is not None else None
        global_project_premium = math.log1p(max(float(scenario.global_state.project_price_premium), -0.95))
        project_premium_path_shift = float(driver.path.project_premium_shift_log) if driver is not None else 0.0
        market_quarterly_shock = (
            float(scenario.local_path.price_shock(ctx.index))
            + (float(driver_q.market_quarterly_shock_log) if market_driver_active and driver_q is not None else 0.0)
        )
        final_price_residual_shock = float(driver_q.price_residual_shock_log) if driver_q is not None else 0.0
        macro_shift_annual = float(driver.path.macro_shift_annual) if market_driver_active and driver is not None else 0.0
        market_regime_shift_log = float(driver.path.market_regime_shift_log) if market_driver_active and driver is not None else 0.0

        geo_features = dict(self._static_geo_features)
        geo_score = float(self._cached_geo_score if self._cached_geo_score is not None else 0.0)
        if self.enable_geo and self._cached_geo_score is None:
            geo_score = self._predict_geo_score(geo_features)

        quality_features = self.feature_builder.build_time_varying_project_features(
            project_input=self.project_input,
            ctx=ctx,
            config=config,
        )
        quality_score = self._predict_quality_score(quality_features)

        spatial_features = self.feature_builder.build_spatial_context_features(
            ctx=ctx,
            h3_encoder=self.artifacts.h3_encoder,
        )

        additive_features: Dict[str, Any] = {}
        additive_features.update(geo_features)
        additive_features.update(quality_features)
        additive_features.update(spatial_features)
        additive_features.update(
            {
                "market_log_price": market_log,
                "geo_score": geo_score,
                "quality_score": quality_score,
                "project_premium": self._project_premium,
                "rooms_num": quality_features.get("rooms_num", self.artifacts.feature_defaults.get("rooms", 2.0)),
                "area_sqm": quality_features.get("area_sqm", config.avg_unit_area_sqm),
                "floor": quality_features.get("floor", self.artifacts.feature_defaults.get("floor", 5.0)),
                "floor_rel": quality_features.get("floor_rel", 0.5),
                "class_final": quality_features.get("class_final", self.project_input.class_final or "Unknown"),
            }
        )

        additive_log = market_log + geo_score + quality_score + self._project_premium + global_project_premium + project_premium_path_shift
        lgb_boost, missing_lgb = self._predict_lgb_boost(
            additive_features=additive_features,
            additive_log_price=additive_log,
        )
        residual_ml_boost = float(lgb_boost)
        final_log = additive_log + lgb_boost + final_price_residual_shock
        final_price = float(math.exp(final_log))

        # Lightweight diagnostics for engine states and external debugging.
        self._last_missing_lgb_cols = missing_lgb
        self._last_source_map = {
            "market_log_price": "market_path_model",
            "geo_score": "geo_model" if self.enable_geo else "disabled",
            "quality_score": "quality_model" if self.enable_quality else "disabled",
            "project_premium": "project_premium_table" if self.artifacts.project_premium_table is not None else "fallback_zero",
            "lgb_boost": "lgb_residual_model"
            if self.artifacts.lgb_residual_model is not None
            else ("final_model_fallback" if self.artifacts.final_model_pipeline is not None else "disabled"),
        }
        self._last_components = {
            "market_log_price": float(market_log),
            "market_quarterly_shock": float(market_quarterly_shock),
            "market_regime_shift_log": float(market_regime_shift_log),
            "macro_shift_annual": float(macro_shift_annual),
            "geo_score": float(geo_score),
            "quality_score": float(quality_score),
            "project_premium_base": float(self._project_premium),
            "project_premium_path_shift": float(project_premium_path_shift),
            "global_project_premium": float(global_project_premium),
            "project_premium": float(self._project_premium + global_project_premium + project_premium_path_shift),
            "project_premium_component": float(self._project_premium + global_project_premium + project_premium_path_shift),
            "lgb_boost": float(lgb_boost),
            "residual_ml_boost": float(residual_ml_boost),
            "final_price_residual_shock": float(final_price_residual_shock),
            "final_log_price_sqm": float(final_log),
            "final_price_sqm": float(final_price),
            "market_price_sqm": float(market_payload["market_price_sqm"]),
            "knn_mean_1000m": float(spatial_features.get("knn_mean_1000m", 0.0)),
            "knn_count_1000m": float(spatial_features.get("knn_count_1000m", 0.0)),
            "h3_enc_r8": float(spatial_features.get("h3_enc_r8", 0.0)),
            "h3_enc_r9": float(spatial_features.get("h3_enc_r9", 0.0)),
            "lgb_missing_feature_count": float(len(missing_lgb)),
        }
        return float(final_log)

    def predict_price_sqm(
        self,
        *,
        ctx: QuarterContext,
        config: ProjectConfig,
        scenario: Scenario,
        previous_state: Optional[QuarterState],
    ) -> float:
        final_log = self.predict_log_price_sqm(
            ctx=ctx,
            config=config,
            scenario=scenario,
            previous_state=previous_state,
        )
        return float(math.exp(final_log))

    def explain_components(
        self,
        *,
        ctx: Optional[QuarterContext] = None,
        config: Optional[ProjectConfig] = None,
        scenario: Optional[Scenario] = None,
        previous_state: Optional[QuarterState] = None,
    ) -> Dict[str, float]:
        if ctx is not None and config is not None and scenario is not None:
            _ = self.predict_price_sqm(
                ctx=ctx,
                config=config,
                scenario=scenario,
                previous_state=previous_state,
            )
        return dict(self._last_components)

    def get_last_components(self) -> Dict[str, float]:
        return dict(self._last_components)

    @property
    def last_missing_lgb_columns(self) -> List[str]:
        return list(self._last_missing_lgb_cols)

    @property
    def last_source_map(self) -> Dict[str, str]:
        return dict(self._last_source_map)


# ---------------------------------------------------------------------------
# Demo helper
# ---------------------------------------------------------------------------


def build_hedonic_demo_engine(
    *,
    artifacts_dir: str | Path,
    reference_deals_path: str | Path,
    region_key: str,
    project_name: str,
    project_id: Optional[str | int],
    sellable_area_sqm: float,
    avg_unit_area_sqm: float,
    initial_remaining_lots: int,
    class_final: str = "Комфорт",
    construction_type_final: str = "монолит",
    finishing: str = "без отделки",
    known_geo_features: Optional[Dict[str, float]] = None,
) -> Tuple[OnePathCashflowEngine, HedonicPriceModel]:
    """
    End-to-end helper:
    - loads artifacts
    - builds hedonic price model
    - plugs it into existing cashflow engine
    - uses inference-only NB sales model
    """
    project_input = ProjectSimulationInput(
        region_key=region_key,
        project_id=project_id,
        project_name=project_name,
        class_final=class_final,
        construction_type_final=construction_type_final,
        finishing=finishing,
        lots_total=float(initial_remaining_lots),
        sellable_area_sqm=float(sellable_area_sqm),
        avg_unit_area_sqm=float(avg_unit_area_sqm),
        known_geo_features=known_geo_features or {},
    )
    price_model = HedonicPriceModel.from_artifacts(
        project_input=project_input,
        artifact_dir=artifacts_dir,
        reference_deals_path=reference_deals_path,
        market_mode="stochastic",
    )
    config = ProjectConfig(
        name=project_name,
        start_year=2026,
        start_quarter=1,
        horizon_quarters=10,
        rns_quarter_index=0,
        rvz_quarter_index=8,
        sellable_area_sqm=sellable_area_sqm,
        avg_unit_area_sqm=avg_unit_area_sqm,
        initial_remaining_lots=initial_remaining_lots,
        key_rate_annual=0.21,
        full_rate_spread_before_rvz=0.035,
        full_rate_spread_after_rvz=0.035,
        privileged_rate_annual=0.037,
        reserve_fee_annual=0.005,
        ltc=0.92,
        land_cost_total=125_000_000.0,
        smr_cost_total=1_030_000_000.0,
        marketing_cost_ratio=0.05,
        profit_tax_rate=0.25,
        collateral_discount=0.30,
        collateral_price_sqm=168_000.0,
        initial_price_sqm=190_000.0,
    )
    cost_schedule = CostSchedule(
        land_by_quarter=[125_000_000.0] + [0.0] * 9,
        smr_by_quarter=[0.0, 246_000_000.0, 112_000_000.0, 112_000_000.0, 112_000_000.0, 112_000_000.0, 112_000_000.0, 112_000_000.0, 112_000_000.0, 0.0],
    )

    sales_model = FittedNBFeatureSalesModel.from_csv(
        region="msk",
        class_group=class_final,
        market_price_sqm=190_000.0,
        stochastic=True,
        seed=42,
    )
    engine = OnePathCashflowEngine(
        config=config,
        cost_schedule=cost_schedule,
        price_model=price_model,
        sales_model=sales_model,
    )
    return engine, price_model


def demo_run_hedonic_integration(
    *,
    artifacts_dir: str | Path,
    reference_deals_path: str | Path,
) -> Dict[str, Any]:
    engine, price_model = build_hedonic_demo_engine(
        artifacts_dir=artifacts_dir,
        reference_deals_path=reference_deals_path,
        region_key="Moscow",
        project_name="demo_hedonic_project",
        project_id=None,
        sellable_area_sqm=11_113.0,
        avg_unit_area_sqm=50.29,
        initial_remaining_lots=221,
    )
    result = engine.run()
    price_component_rows: List[Dict[str, float]] = []
    for state in result.states:
        price_component_rows.append(
            {
                "quarter": _quarter_id_to_label_year_first(state.quarter_id),
                "market_log_price": state.market_log_price,
                "geo_score": state.geo_score,
                "quality_score": state.quality_score,
                "project_premium_component": state.project_premium_component,
                "lgb_boost": state.lgb_boost,
                "final_log_price_sqm": state.final_log_price_sqm,
                "price_sqm": state.price_sqm,
            }
        )
    return {
        "summary": result.summary,
        "price_components": price_component_rows,
        "artifact_warnings": list(price_model.artifacts.warnings_log),
    }
