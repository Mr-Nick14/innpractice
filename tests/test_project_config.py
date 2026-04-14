"""Tests for ProjectConfig validation and computed properties."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo.mc_cashflow_engine import ProjectConfig


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------

class TestProjectConfigValid:
    def test_minimal_valid(self, tiny_config):
        assert tiny_config.name == "tiny_test"
        assert tiny_config.horizon_quarters == 4

    def test_total_project_cost(self, tiny_config):
        # land=5M + smr=8M + design=0
        assert tiny_config.total_project_cost == 13_000_000.0

    def test_debt_limit_from_ltc(self, tiny_config):
        # 13M * 0.80 = 10.4M
        assert tiny_config.resolved_debt_limit == pytest.approx(10_400_000.0)

    def test_initial_equity(self, tiny_config):
        # 13M - 10.4M = 2.6M
        assert tiny_config.resolved_initial_equity == pytest.approx(2_600_000.0)

    def test_explicit_debt_limit_overrides_ltc(self):
        cfg = ProjectConfig(
            name="x", start_year=2026, start_quarter=1,
            horizon_quarters=4, rns_quarter_index=0, rvz_quarter_index=3,
            sellable_area_sqm=100, avg_unit_area_sqm=50,
            initial_remaining_lots=2,
            key_rate_annual=0.1, full_rate_spread_before_rvz=0.01,
            full_rate_spread_after_rvz=0.01, privileged_rate_annual=0.02,
            reserve_fee_annual=0.005,
            debt_limit=5_000_000.0,
        )
        assert cfg.resolved_debt_limit == 5_000_000.0

    def test_explicit_equity_overrides_computed(self):
        cfg = ProjectConfig(
            name="x", start_year=2026, start_quarter=1,
            horizon_quarters=4, rns_quarter_index=0, rvz_quarter_index=3,
            sellable_area_sqm=100, avg_unit_area_sqm=50,
            initial_remaining_lots=2,
            key_rate_annual=0.1, full_rate_spread_before_rvz=0.01,
            full_rate_spread_after_rvz=0.01, privileged_rate_annual=0.02,
            reserve_fee_annual=0.005,
            initial_equity_contribution=1_000_000.0,
        )
        assert cfg.resolved_initial_equity == 1_000_000.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestProjectConfigValidation:
    def _base_kwargs(self):
        return dict(
            name="test", start_year=2026, start_quarter=1,
            horizon_quarters=4, rns_quarter_index=0, rvz_quarter_index=3,
            sellable_area_sqm=100, avg_unit_area_sqm=50,
            initial_remaining_lots=2,
            key_rate_annual=0.1, full_rate_spread_before_rvz=0.01,
            full_rate_spread_after_rvz=0.01, privileged_rate_annual=0.02,
            reserve_fee_annual=0.005,
        )

    def test_horizon_must_be_positive(self):
        kw = self._base_kwargs()
        kw["horizon_quarters"] = 0
        with pytest.raises(ValueError, match="horizon_quarters must be positive"):
            ProjectConfig(**kw)

    def test_negative_lots_rejected(self):
        kw = self._base_kwargs()
        kw["initial_remaining_lots"] = -1
        with pytest.raises(ValueError, match="initial_remaining_lots must be non-negative"):
            ProjectConfig(**kw)

    def test_avg_unit_area_must_be_positive(self):
        kw = self._base_kwargs()
        kw["avg_unit_area_sqm"] = 0
        with pytest.raises(ValueError, match="avg_unit_area_sqm must be positive"):
            ProjectConfig(**kw)

    def test_start_quarter_range(self):
        kw = self._base_kwargs()
        kw["start_quarter"] = 5
        with pytest.raises(ValueError, match="start_quarter must be in 1..4"):
            ProjectConfig(**kw)

    def test_rns_out_of_range(self):
        kw = self._base_kwargs()
        kw["rns_quarter_index"] = 5
        with pytest.raises(ValueError, match="rns_quarter_index out of range"):
            ProjectConfig(**kw)

    def test_rvz_before_rns_rejected(self):
        kw = self._base_kwargs()
        kw["rns_quarter_index"] = 2
        kw["rvz_quarter_index"] = 1
        with pytest.raises(ValueError, match="rvz_quarter_index must be >= rns_quarter_index"):
            ProjectConfig(**kw)
