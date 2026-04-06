from __future__ import annotations

"""
Shared hedonic residualization helpers extracted from the logic in
`hedonic_modelv3.ipynb`.

Purpose:
- keep geo/project residualization outside market-specific notebooks;
- let research notebooks consume prepared residual targets instead of
  redefining geo/project blocks inline.
"""

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
PROJECT_PREMIUM_TAU = 20
BASELINE_TREND_RECENT_Q = 12
BASELINE_DAMPING_PHI = 0.85

DIST_COLS = [
    "dist_center_m",
    "dist_metro_m",
    "dist_bus_m",
    "dist_kindergarten_m",
    "dist_school_m",
    "dist_mall_m",
    "dist_park_m",
    "dist_rail_m",
    "dist_hospital_m",
]

GEO_LAMBDAS_MULTI = {
    "dist_center_m": [5_000, 12_000, 25_000],
    "dist_metro_m": [400, 900, 2_000],
    "dist_bus_m": [150, 350, 800],
    "dist_kindergarten_m": [300, 700, 1_500],
    "dist_school_m": [400, 900, 1_800],
    "dist_mall_m": [1_000, 2_500, 5_000],
    "dist_park_m": [500, 1_200, 3_000],
    "dist_rail_m": [1_000, 2_500, 6_000],
    "dist_hospital_m": [800, 1_800, 4_000],
}

QUALITY_NUM_COLS = [
    "area_sqm",
    "floor",
    "floor_rel",
    "rooms_num",
    "ceiling_m_final",
    "project_age_months",
    "months_to_rve",
    "lots_total",
    "area_project_total",
    "area_project_mean",
    "area_project_median",
]

QUALITY_CAT_COLS = [
    "class_final",
    "construction_type_final",
    "finishing",
]


def resolve_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of the candidate paths exist:\n{searched}")


def resolve_deals_path(base_dir: str | Path) -> Path:
    base_dir = Path(base_dir).resolve()
    return resolve_existing_path(
        [
            base_dir / "data" / "deals_with_real_geo_Moscow.parquet",
            base_dir.parent / "data" / "deals_with_real_geo_Moscow.parquet",
            base_dir.parent.parent / "data" / "deals_with_real_geo_Moscow.parquet",
        ]
    )


def mode_or_nan(values: pd.Series) -> Any:
    values = values.dropna()
    if values.empty:
        return np.nan
    mode_values = values.mode()
    return mode_values.iloc[0] if not mode_values.empty else values.iloc[0]


def build_model_split(quarters: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(pd.Series(quarters).astype("string").unique().tolist(), key=lambda q: pd.Period(q, freq="Q"))
    split_idx = max(1, int(len(ordered) * 0.7))
    return ordered[:split_idx], ordered[split_idx:]


def build_baseline_market_block(deals: pd.DataFrame, train_quarters: list[str]) -> pd.DataFrame:
    enriched = deals.copy()
    train_mask = enriched["quarter"].isin(train_quarters)

    city_q_train = (
        enriched.loc[train_mask]
        .groupby("quarter", as_index=False)
        .agg(
            city_log_price_median=("log_price_sqm", "median"),
            city_n=("log_price_sqm", "size"),
        )
    )

    quarter_order = sorted(enriched["quarter"].unique().tolist(), key=lambda q: pd.Period(q, freq="Q"))
    quarter_to_idx = {quarter: idx for idx, quarter in enumerate(quarter_order)}
    enriched["quarter_idx"] = enriched["quarter"].map(quarter_to_idx)
    city_q_train["quarter_idx"] = city_q_train["quarter"].map(quarter_to_idx)

    recent_q = city_q_train.sort_values("quarter_idx").tail(BASELINE_TREND_RECENT_Q)
    x = recent_q["quarter_idx"].to_numpy(dtype=float)
    y = recent_q["city_log_price_median"].to_numpy(dtype=float)
    w = recent_q["city_n"].to_numpy(dtype=float)

    if len(recent_q) <= 1 or np.allclose(np.var(x), 0.0):
        slope = 0.0
        intercept = float(y[-1]) if len(y) else float(enriched.loc[train_mask, "log_price_sqm"].median())
    else:
        x_mean = np.average(x, weights=w)
        y_mean = np.average(y, weights=w)
        cov_xy = np.average((x - x_mean) * (y - y_mean), weights=w)
        var_x = np.average((x - x_mean) ** 2, weights=w)
        slope = 0.0 if np.isclose(var_x, 0.0) else cov_xy / var_x
        intercept = y_mean - slope * x_mean

    last_train_idx = int(city_q_train["quarter_idx"].max())
    last_train_fitted = intercept + slope * last_train_idx

    def extrapolate_damped(q_idx: float | int | None) -> float:
        if pd.isna(q_idx):
            return np.nan
        q_idx = int(q_idx)
        if q_idx <= last_train_idx:
            return float(intercept + slope * q_idx)
        horizon = q_idx - last_train_idx
        damped_sum = BASELINE_DAMPING_PHI * (1 - BASELINE_DAMPING_PHI**horizon) / (1 - BASELINE_DAMPING_PHI)
        return float(last_train_fitted + slope * damped_sum)

    enriched["baseline_market_trend"] = enriched["quarter_idx"].map(extrapolate_damped)
    enriched = enriched.merge(city_q_train[["quarter", "city_log_price_median"]], on="quarter", how="left")
    enriched["baseline_market_log_price"] = enriched["city_log_price_median"].fillna(enriched["baseline_market_trend"])
    enriched["baseline_market_residual"] = enriched["log_price_sqm"] - enriched["baseline_market_log_price"]
    return enriched


def engineer_geo_features(deals: pd.DataFrame, train_quarters: list[str]) -> tuple[pd.DataFrame, list[str]]:
    enriched = deals.copy()
    train_mask = enriched["quarter"].isin(train_quarters)

    for column in DIST_COLS:
        enriched[f"log1p_{column}"] = np.log1p(enriched[column].clip(lower=0))

    acc_cols: list[str] = []
    for column, lambdas in GEO_LAMBDAS_MULTI.items():
        base = column.replace("dist_", "").replace("_m", "")
        for lam in lambdas:
            acc_name = f"acc_{base}_l{lam}"
            enriched[acc_name] = np.exp(-enriched[column].clip(lower=0) / lam)
            acc_cols.append(acc_name)

    enriched["acc_transit"] = enriched[["acc_metro_l900", "acc_bus_l350", "acc_rail_l2500"]].mean(axis=1)
    enriched["acc_family"] = enriched[["acc_kindergarten_l700", "acc_school_l900", "acc_hospital_l1800"]].mean(axis=1)
    enriched["acc_retail_green"] = enriched[["acc_mall_l2500", "acc_park_l1200"]].mean(axis=1)
    enriched["acc_centrality"] = enriched["acc_center_l12000"]

    pca_scaler = StandardScaler()
    train_acc = enriched.loc[train_mask, acc_cols]
    train_acc_filled = train_acc.fillna(train_acc.median())
    train_acc_scaled = pca_scaler.fit_transform(train_acc_filled)

    pca_geo = PCA(n_components=5, random_state=RANDOM_STATE)
    pca_geo.fit(train_acc_scaled)

    all_acc_filled = enriched[acc_cols].fillna(train_acc.median())
    all_acc_scaled = pca_scaler.transform(all_acc_filled)
    pcs = pca_geo.transform(all_acc_scaled)
    for idx in range(5):
        enriched[f"geo_pc{idx + 1}"] = pcs[:, idx]

    geo_num_cols = (
        DIST_COLS
        + [f"log1p_{column}" for column in DIST_COLS]
        + acc_cols
        + [f"geo_pc{idx + 1}" for idx in range(5)]
        + ["acc_transit", "acc_family", "acc_retail_green", "acc_centrality"]
    )
    return enriched, geo_num_cols


def fit_geo_block(deals: pd.DataFrame, train_quarters: list[str]) -> tuple[pd.DataFrame, Pipeline, list[str]]:
    enriched, geo_cols = engineer_geo_features(deals, train_quarters)
    train_mask = enriched["quarter"].isin(train_quarters)

    geo_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "reg",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    alphas=np.logspace(-4, 0, 15),
                    cv=3,
                    random_state=RANDOM_STATE,
                    max_iter=10_000,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        geo_model.fit(
            enriched.loc[train_mask, geo_cols],
            enriched.loc[train_mask, "baseline_market_residual"],
        )

    enriched["geo_score"] = geo_model.predict(enriched[geo_cols])
    enriched["resid_after_geo"] = enriched["baseline_market_residual"] - enriched["geo_score"]
    return enriched, geo_model, geo_cols


def fit_project_block(
    deals: pd.DataFrame,
    train_quarters: list[str],
) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    enriched = deals.copy()
    train_mask = enriched["quarter"].isin(train_quarters)

    enriched["rooms_num"] = pd.to_numeric(enriched["rooms"], errors="coerce").fillna(
        pd.to_numeric(enriched["rooms"], errors="coerce").median()
    )
    enriched["ceiling_m_final"] = pd.to_numeric(enriched["ceiling_m_pd"], errors="coerce")
    floor_num = pd.to_numeric(enriched["floor"], errors="coerce")
    floor_max = pd.to_numeric(enriched["floor_max_pd"], errors="coerce").replace(0, np.nan)
    enriched["floor_rel"] = (floor_num / floor_max).replace([np.inf, -np.inf], np.nan)

    quality_features = QUALITY_NUM_COLS + QUALITY_CAT_COLS
    train_num = enriched.loc[train_mask, QUALITY_NUM_COLS].copy().apply(pd.to_numeric, errors="coerce")
    train_num_imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(train_num),
        columns=QUALITY_NUM_COLS,
        index=train_num.index,
    )
    num_medians = train_num_imputed.median()

    for column in QUALITY_NUM_COLS:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(num_medians[column])
    for column in QUALITY_CAT_COLS:
        enriched[column] = enriched[column].astype("object").fillna("missing")

    from sklearn.compose import ColumnTransformer

    quality_preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                QUALITY_NUM_COLS,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                QUALITY_CAT_COLS,
            ),
        ]
    )

    quality_model = Pipeline(
        steps=[
            ("prep", quality_preprocess),
            ("reg", RidgeCV(alphas=np.logspace(-3, 3, 15))),
        ]
    )
    quality_model.fit(
        enriched.loc[train_mask, quality_features],
        enriched.loc[train_mask, "resid_after_geo"],
    )

    enriched["quality_score"] = quality_model.predict(enriched[quality_features])
    enriched["resid_after_quality"] = enriched["resid_after_geo"] - enriched["quality_score"]

    project_table = (
        enriched.loc[train_mask]
        .groupby("project_id", as_index=False)
        .agg(
            project_resid_mean=("resid_after_quality", "mean"),
            project_obs=("resid_after_quality", "size"),
            project_name=("project_name", mode_or_nan),
        )
    )
    project_table["project_premium"] = (
        project_table["project_obs"] / (project_table["project_obs"] + PROJECT_PREMIUM_TAU)
    ) * project_table["project_resid_mean"]

    enriched = enriched.merge(
        project_table[["project_id", "project_obs", "project_premium"]],
        on="project_id",
        how="left",
    )
    enriched["project_obs"] = enriched["project_obs"].fillna(0)
    enriched["project_premium"] = enriched["project_premium"].fillna(0.0)
    enriched["project_score"] = enriched["quality_score"] + enriched["project_premium"]
    enriched["market_target"] = enriched["log_price_sqm"] - enriched["geo_score"] - enriched["project_score"]
    return enriched, quality_model, project_table


def prepare_market_residual_inputs(
    base_dir: str | Path,
    allowed_quarters: list[str],
) -> dict[str, Any]:
    base_dir = Path(base_dir).resolve()
    deals_path = resolve_deals_path(base_dir)

    deals = pd.read_parquet(deals_path).copy()
    deals["quarter"] = deals["quarter"].astype("string")
    allowed_quarters = sorted(
        set(pd.Series(allowed_quarters).astype("string")) & set(deals["quarter"].dropna()),
        key=lambda q: pd.Period(q, freq="Q"),
    )
    deals = deals.loc[deals["quarter"].isin(allowed_quarters)].copy()

    train_quarters, test_quarters = build_model_split(allowed_quarters)
    deals["is_train"] = deals["quarter"].isin(train_quarters)
    deals["is_test"] = deals["quarter"].isin(test_quarters)

    deals = build_baseline_market_block(deals, train_quarters)
    deals, geo_model, geo_feature_cols = fit_geo_block(deals, train_quarters)
    deals, quality_model, project_table = fit_project_block(deals, train_quarters)

    diagnostics = {
        "deals_path": str(deals_path),
        "train_first": train_quarters[0],
        "train_last": train_quarters[-1],
        "test_first": test_quarters[0],
        "test_last": test_quarters[-1],
        "baseline_market_train_r2": float(
            1 - deals.loc[deals["is_train"], "baseline_market_residual"].var() / deals.loc[deals["is_train"], "log_price_sqm"].var()
        ),
        "geo_train_r2": float(
            r2_score(deals.loc[deals["is_train"], "baseline_market_residual"], deals.loc[deals["is_train"], "geo_score"])
        ),
        "geo_test_r2": float(
            r2_score(deals.loc[deals["is_test"], "baseline_market_residual"], deals.loc[deals["is_test"], "geo_score"])
        ),
        "quality_train_r2": float(
            r2_score(deals.loc[deals["is_train"], "resid_after_geo"], deals.loc[deals["is_train"], "quality_score"])
        ),
        "quality_test_r2": float(
            r2_score(deals.loc[deals["is_test"], "resid_after_geo"], deals.loc[deals["is_test"], "quality_score"])
        ),
        "projects_with_premium": int(project_table.shape[0]),
        "geo_feature_count": len(geo_feature_cols),
    }

    return {
        "deals": deals,
        "train_quarters": train_quarters,
        "test_quarters": test_quarters,
        "diagnostics": diagnostics,
        "project_table": project_table,
        "geo_model": geo_model,
        "quality_model": quality_model,
    }
