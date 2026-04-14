"""Tests for CostSchedule validation and accessors."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo.mc_cashflow_engine import CostSchedule


class TestCostScheduleAccess:
    def test_land_returns_value(self, tiny_cost_schedule):
        assert tiny_cost_schedule.land(0) == 5_000_000.0
        assert tiny_cost_schedule.land(1) == 0.0

    def test_smr_returns_value(self, tiny_cost_schedule):
        for i in range(4):
            assert tiny_cost_schedule.smr(i) == 2_000_000.0

    def test_empty_sequence_returns_zero(self):
        cs = CostSchedule(land_by_quarter=[1.0], smr_by_quarter=[2.0])
        assert cs.design(0) == 0.0
        assert cs.other_opex(0) == 0.0
        assert cs.post_completion(0) == 0.0
        assert cs.property_tax(0) == 0.0
        assert cs.vat(0) == 0.0


class TestCostScheduleValidation:
    def test_valid_schedule_passes(self, tiny_cost_schedule):
        tiny_cost_schedule.validate(4)

    def test_wrong_length_raises(self):
        cs = CostSchedule(
            land_by_quarter=[1.0, 2.0],
            smr_by_quarter=[1.0, 2.0, 3.0],
        )
        with pytest.raises(ValueError, match="smr_by_quarter length must equal horizon"):
            cs.validate(2)

    def test_optional_empty_passes(self):
        cs = CostSchedule(
            land_by_quarter=[1.0, 2.0],
            smr_by_quarter=[1.0, 2.0],
        )
        cs.validate(2)
