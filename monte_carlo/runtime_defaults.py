from __future__ import annotations

import warnings
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE_ROOT / "cashflow"
MONTE_CARLO_ROOT = PROJECT_ROOT / "monte_carlo"
MONTE_CARLO_MODEL_ARTIFACTS_DIR = MONTE_CARLO_ROOT / "model_artifacts"
OLD_VERS_ROOT = WORKSPACE_ROOT / "old_vers" / "cashflow_project"
OLD_VERS_DATA_ROOT = OLD_VERS_ROOT / "data"
CURRENT_DATA_ROOT = PROJECT_ROOT / "data"

DEFAULT_REGION_KEY = "Moscow"
DEFAULT_DEALS_PARQUET = OLD_VERS_DATA_ROOT / "geo_enriched" / "deals_with_real_geo_Moscow.parquet"
DEFAULT_MODEL_ARTIFACTS_DIR = MONTE_CARLO_MODEL_ARTIFACTS_DIR
LEGACY_MODEL_ARTIFACTS_DIR = OLD_VERS_DATA_ROOT / "model_artifacts"


def default_moscow_deals_path() -> Path:
    candidates = [
        DEFAULT_DEALS_PARQUET,
        CURRENT_DATA_ROOT / "deals_with_real_geo_Moscow.parquet",
        WORKSPACE_ROOT / "data" / "deals_with_real_geo_Moscow.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_model_artifacts_dir() -> Path:
    candidates = [
        MONTE_CARLO_MODEL_ARTIFACTS_DIR,
        LEGACY_MODEL_ARTIFACTS_DIR,
        CURRENT_DATA_ROOT / "model_artifacts",
    ]
    for idx, candidate in enumerate(candidates):
        if candidate.exists():
            if idx > 0:
                warnings.warn(
                    f"Falling back to legacy model artifacts directory: {candidate}. "
                    f"Populate {MONTE_CARLO_MODEL_ARTIFACTS_DIR} to keep the new monte_carlo layout canonical.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return candidate
    return candidates[0]
