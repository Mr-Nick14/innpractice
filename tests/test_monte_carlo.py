"""Tests for MonteCarloRunner and ScenarioFactory."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo.mc_cashflow_engine import (
    MonteCarloRunner,
    RandomScenarioSpec,
    ScenarioFactory,
)


class TestScenarioFactory:
    def test_deterministic_with_seed(self):
        sf1 = ScenarioFactory(seed=42)
        sf2 = ScenarioFactory(seed=42)
        spec = RandomScenarioSpec(
            price_level_min=0.9, price_level_max=1.1,
            sales_level_min=0.8, sales_level_max=1.2,
        )
        s1 = sf1.make_random(name="a", horizon=4, spec=spec)
        s2 = sf2.make_random(name="a", horizon=4, spec=spec)
        assert s1.global_state.price_level_multiplier == s2.global_state.price_level_multiplier

    def test_scenario_within_bounds(self):
        sf = ScenarioFactory(seed=7)
        spec = RandomScenarioSpec(
            price_level_min=0.9, price_level_max=1.1,
            sales_level_min=0.8, sales_level_max=1.0,
            key_rate_shift_min=-0.01, key_rate_shift_max=0.03,
        )
        for _ in range(50):
            s = sf.make_random(name="t", horizon=4, spec=spec)
            assert 0.9 <= s.global_state.price_level_multiplier <= 1.1
            assert 0.8 <= s.global_state.sales_level_multiplier <= 1.0
            assert -0.01 <= s.global_state.key_rate_shift_annual <= 0.03


class TestMonteCarloRunner:
    def test_runs_correct_count(self, tiny_engine):
        sf = ScenarioFactory(seed=42)
        runner = MonteCarloRunner(tiny_engine, sf)
        mc = runner.run_random(
            n_runs=5,
            spec=RandomScenarioSpec(),
            name_prefix="test",
        )
        assert mc.summary.runs == 5
        assert len(mc.path_results) == 5

    def test_zero_runs_rejected(self, tiny_engine):
        sf = ScenarioFactory(seed=42)
        runner = MonteCarloRunner(tiny_engine, sf)
        with pytest.raises(ValueError, match="n_runs must be positive"):
            runner.run_random(n_runs=0, spec=RandomScenarioSpec())

    def test_summary_statistics_populated(self, tiny_engine):
        sf = ScenarioFactory(seed=42)
        runner = MonteCarloRunner(tiny_engine, sf)
        mc = runner.run_random(
            n_runs=10,
            spec=RandomScenarioSpec(
                price_level_min=0.95, price_level_max=1.05,
                sales_level_min=0.90, sales_level_max=1.10,
            ),
        )
        s = mc.summary
        assert s.runs == 10
        assert s.p05_min_dscr <= s.p50_min_dscr <= s.p95_min_dscr
        assert s.mean_peak_debt > 0
        assert 0 <= s.probability_dscr_below_1_0 <= 1.0
        assert 0 <= s.probability_dscr_below_1_2 <= 1.0

    def test_reproducible_with_same_seed(self, tiny_engine):
        spec = RandomScenarioSpec(
            price_level_min=0.9, price_level_max=1.1,
            sales_level_min=0.8, sales_level_max=1.2,
            local_price_shock_std=0.01,
        )
        sf1 = ScenarioFactory(seed=123)
        r1 = MonteCarloRunner(tiny_engine, sf1).run_random(n_runs=5, spec=spec)

        sf2 = ScenarioFactory(seed=123)
        r2 = MonteCarloRunner(tiny_engine, sf2).run_random(n_runs=5, spec=spec)

        assert r1.summary.mean_min_dscr == pytest.approx(r2.summary.mean_min_dscr)
        assert r1.summary.mean_peak_debt == pytest.approx(r2.summary.mean_peak_debt)
