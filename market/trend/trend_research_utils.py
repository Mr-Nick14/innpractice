from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TREND_WINDOWS: tuple[int, ...] = (3, 6, 10)
DEFAULT_KEY_RATE_COLUMNS: tuple[str, ...] = (
    "key_rate_avg_q",
    "key_rate_eoq",
    "key_rate_min_q",
    "key_rate_max_q",
    "key_rate_log_return_q",
    "key_rate_qoq_change",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    feature_cols: tuple[str, ...]


def resolve_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "cashflow" / "market" / "hedonic_residualization.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from current working directory")


def quarter_to_idx(quarter: str) -> int:
    year = int(quarter[:4])
    q = int(quarter[-1])
    return year * 4 + q


def idx_to_quarter(idx: int) -> str:
    year, q = divmod(int(idx), 4)
    if q == 0:
        year -= 1
        q = 4
    return f"{year}Q{q}"


def generate_test_positions(n_quarters: int, *, test_frac: float = 0.3, n_positions: int = 10) -> list[tuple[int, int]]:
    if n_quarters <= 0:
        raise ValueError("n_quarters must be positive")
    test_size = max(1, int(round(n_quarters * test_frac)))
    max_start = max(0, n_quarters - test_size)
    starts = np.linspace(0, max_start, n_positions).round().astype(int)
    starts = np.unique(starts)
    return [(int(s), int(min(s + test_size, n_quarters))) for s in starts]


def load_trend_panel(root: Path | None = None) -> pd.DataFrame:
    root = resolve_repo_root(root)
    clean_path = root / "cashflow" / "market" / "data" / "quarterly_market_clean.parquet"
    macro_path = root / "cashflow" / "macro" / "data" / "macro_quarterly_Moscow.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean market series not found: {clean_path}")
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro file not found: {macro_path}")

    market = pd.read_parquet(clean_path).copy()
    macro = pd.read_csv(macro_path).copy()

    required_market = {"quarter", "market_log_level_q", "actual_log_price_median"}
    missing_market = required_market - set(market.columns)
    if missing_market:
        raise KeyError(f"Clean market series is missing columns: {sorted(missing_market)}")

    missing_macro = [col for col in DEFAULT_KEY_RATE_COLUMNS if col not in macro.columns]
    if missing_macro:
        raise KeyError(f"Macro data is missing key-rate columns: {missing_macro}")

    market["quarter_norm"] = market["quarter"].astype("string")
    market = market.dropna(subset=["quarter_norm", "market_log_level_q"]).copy()
    market["quarter_order"] = market["quarter_norm"].map(quarter_to_idx)
    market = market.sort_values("quarter_order").drop_duplicates("quarter_norm", keep="last").reset_index(drop=True)

    macro = macro[["quarter", *DEFAULT_KEY_RATE_COLUMNS]].copy()
    macro["quarter_norm"] = macro["quarter"].astype("string")
    macro["quarter_order"] = macro["quarter_norm"].map(quarter_to_idx)

    panel = market.merge(macro, on="quarter_norm", how="left", suffixes=("", "_macro"))
    panel = panel.sort_values("quarter_order").reset_index(drop=True)

    key_rate_cols = list(DEFAULT_KEY_RATE_COLUMNS)
    if panel[key_rate_cols].isna().any().any():
        missing_rows = int(panel[key_rate_cols].isna().any(axis=1).sum())
        raise ValueError(f"Key-rate columns contain missing values on {missing_rows} quarterly rows")

    panel["quarter_period"] = pd.PeriodIndex(panel["quarter_norm"], freq="Q")
    panel["idx"] = np.arange(len(panel), dtype=int)
    panel["market_log_ret_q"] = panel["market_log_level_q"].diff()
    return panel


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        level = float(y[-1]) if len(y) else 0.0
        return 0.0, level
    if np.allclose(np.var(x), 0.0) or np.allclose(np.var(y), 0.0):
        return 0.0, float(y[-1])
    slope, intercept = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(slope), float(intercept)


def _causal_base_series(y: np.ndarray, *, window: int, kind: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.empty(n, dtype=float)
    if n == 0:
        return out

    for t in range(n):
        hist = y[:t]
        if t == 0:
            out[t] = float(y[0])
            continue
        span = min(max(int(window), 1), t)
        recent = hist[-span:]

        if kind == "sma":
            out[t] = float(np.nanmean(recent))
        elif kind == "ema":
            out[t] = float(pd.Series(hist, dtype=float).ewm(span=span, adjust=False).mean().iloc[-1])
        elif kind == "lin":
            x = np.arange(t - span, t, dtype=float)
            slope, intercept = _fit_line(x, recent)
            out[t] = float(intercept + slope * t)
        elif kind == "poly":
            deg = 2 if span >= 3 else 1
            x = np.arange(t - span, t, dtype=float)
            if len(recent) <= deg:
                slope, intercept = _fit_line(x, recent)
                out[t] = float(intercept + slope * t)
            else:
                coeffs = np.polyfit(x, recent.astype(float), deg)
                out[t] = float(np.polyval(coeffs, t))
        else:
            raise ValueError(f"Unknown trend kind: {kind}")
    return out


def _extend_block(base: np.ndarray, *, test_start: int, test_end: int, fit_window: int) -> np.ndarray:
    out = np.asarray(base, dtype=float).copy()
    if test_start <= 0 or test_start >= len(out):
        return out
    anchor_start = max(0, test_start - min(test_start, fit_window))
    anchor_x = np.arange(anchor_start, test_start, dtype=float)
    anchor_y = out[anchor_start:test_start]
    slope, intercept = _fit_line(anchor_x, anchor_y)
    test_x = np.arange(test_start, min(test_end, len(out)), dtype=float)
    out[test_start:test_start + len(test_x)] = intercept + slope * test_x
    return out


def build_trend_feature_frame(
    y: Sequence[float] | np.ndarray,
    *,
    test_start: int,
    test_end: int,
    windows: Sequence[int] = DEFAULT_TREND_WINDOWS,
    kinds: Sequence[str] = ("lin", "sma", "ema", "poly"),
) -> pd.DataFrame:
    y_arr = np.asarray(y, dtype=float)
    feature_data: dict[str, np.ndarray] = {}
    for kind in kinds:
        for window in windows:
            base = _causal_base_series(y_arr, window=int(window), kind=kind)
            # Preserve the pre-test history and extrapolate the test block from it.
            extended = _extend_block(base, test_start=test_start, test_end=test_end, fit_window=max(3, int(window)))
            feature_data[f"{kind}_{int(window)}"] = extended
    return pd.DataFrame(feature_data)


def build_key_rate_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel[list(DEFAULT_KEY_RATE_COLUMNS)].copy()
    frame["key_rate_spread_q"] = frame["key_rate_max_q"] - frame["key_rate_min_q"]
    frame["key_rate_ma_4"] = frame["key_rate_avg_q"].rolling(4, min_periods=1).mean()
    frame["key_rate_ema_4"] = frame["key_rate_avg_q"].ewm(span=4, adjust=False).mean()
    frame["key_rate_ma_8"] = frame["key_rate_avg_q"].rolling(8, min_periods=1).mean()
    frame["key_rate_ema_8"] = frame["key_rate_avg_q"].ewm(span=8, adjust=False).mean()
    return frame


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float], *, name: str, scope: str) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return pd.DataFrame(
        [
            {
                "scope": scope,
                "model": name,
                "MAE_log": mean_absolute_error(y_true_arr, y_pred_arr),
                "RMSE_log": mean_squared_error(y_true_arr, y_pred_arr) ** 0.5,
                "R2": r2_score(y_true_arr, y_pred_arr),
            }
        ]
    )


def build_model_pipeline(kind: str) -> Pipeline:
    kind = kind.strip().lower()
    if kind == "linear":
        estimator = LinearRegression()
    elif kind == "ridge":
        estimator = RidgeCV(alphas=np.logspace(-4, 3, 25))
    elif kind == "lasso":
        estimator = LassoCV(alphas=np.logspace(-4, 1, 30), cv=3, max_iter=100_000, random_state=42)
    elif kind == "elasticnet":
        estimator = ElasticNetCV(alphas=np.logspace(-4, 1, 30), l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=3, max_iter=100_000, random_state=42)
    elif kind == "huber":
        estimator = HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=5000)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def extract_coefficients(model: Pipeline, feature_names: Sequence[str], *, model_name: str, scope: str) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame(columns=["scope", "model", "feature", "coef"])
    coefs = np.asarray(estimator.coef_, dtype=float).ravel()
    return pd.DataFrame(
        {
            "scope": scope,
            "model": model_name,
            "feature": list(feature_names),
            "coef": coefs[: len(feature_names)],
        }
    )
