"""Tests for stochastic driver generation and runtime cost adaptation."""

import math

import pytest

from monte_carlo.mc_cashflow_engine import Scenario
from monte_carlo.mc_stochastic_drivers import (
    DriverScenarioFactory,
    MonteCarloDriverConfig,
    RuntimeCostAdapter,
)


class TestDriverScenarioFactory:
    def test_reproducible_with_same_seed(self):
        config = MonteCarloDriverConfig(use_correlations=True)
        first = DriverScenarioFactory(config, seed=2026).make(name="a", horizon=4)
        second = DriverScenarioFactory(config, seed=2026).make(name="a", horizon=4)

        assert first.cache_key() == second.cache_key()

    def test_rejects_non_positive_horizon(self):
        factory = DriverScenarioFactory(seed=1)

        with pytest.raises(ValueError, match="horizon must be positive"):
            factory.make(name="bad", horizon=0)

    def test_trend_mode_disables_market_modulation(self):
        config = MonteCarloDriverConfig(
            market_path_mode="trend",
            market_regime_shift_std=1.0,
            market_shock_std=1.0,
        )
        scenario = DriverScenarioFactory(config, seed=42).make(name="trend", horizon=3)

        assert scenario.market_modulation_enabled() is False
        assert scenario.market_log_price_modulation(1) == 0.0


class TestRuntimeCostAdapter:
    def test_without_driver_keeps_base_schedule(self, tiny_cost_schedule):
        schedule = RuntimeCostAdapter.build_effective_schedule(
            base_schedule=tiny_cost_schedule,
            horizon=4,
            scenario=Scenario(),
        )

        assert schedule.land_by_quarter == (5_000_000.0, 0.0, 0.0, 0.0)
        assert schedule.smr_by_quarter == (2_000_000.0, 2_000_000.0, 2_000_000.0, 2_000_000.0)
        assert schedule.cost_index_by_quarter == (1.0, 1.0, 1.0, 1.0)

    def test_driver_applies_cost_timing_and_multiplier(self, tiny_cost_schedule):
        config = MonteCarloDriverConfig(
            overrun_log_std=0.0,
            cost_inflation_std=0.0,
            cost_quarterly_shock_std=0.0,
            rvz_delay_values=(1,),
            rvz_delay_probs=(1.0,),
            apply_cost_timing_shift=True,
        )
        driver = DriverScenarioFactory(config, seed=7).make(name="delay", horizon=4)
        scenario = Scenario(driver_scenario=driver)

        schedule = RuntimeCostAdapter.build_effective_schedule(
            base_schedule=tiny_cost_schedule,
            horizon=4,
            scenario=scenario,
        )

        assert schedule.cost_timing_shift_quarters == 1
        assert schedule.smr_by_quarter == pytest.approx((0.0, 2_000_000.0, 2_000_000.0, 2_000_000.0))
        assert all(math.isclose(value, 1.0) for value in schedule.cost_index_by_quarter)
