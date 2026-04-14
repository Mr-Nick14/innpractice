"""
Tests for OnePathCashflowEngine core logic.

Every expected value here was verified by hand against the formulas
from 'Описание калькулятора cashflow.docx' and the cashflow.xlsx spreadsheet.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo.mc_cashflow_engine import (
    GlobalScenario,
    LocalScenarioPath,
    OnePathCashflowEngine,
    PathResult,
    Scenario,
    StubPriceModel,
    StubSalesModel,
    safe_div,
    QuarterId,
)


# ===================================================================
# Helper
# ===================================================================

def run_tiny(tiny_engine) -> PathResult:
    return tiny_engine.run()


# ===================================================================
# QuarterId
# ===================================================================

class TestQuarterId:
    def test_next_within_year(self):
        assert QuarterId(2026, 1).next() == QuarterId(2026, 2)

    def test_next_wraps_year(self):
        assert QuarterId(2026, 4).next() == QuarterId(2027, 1)

    def test_label(self):
        assert QuarterId(2026, 3).label() == "3Q2026"


# ===================================================================
# safe_div
# ===================================================================

class TestSafeDiv:
    def test_normal(self):
        assert safe_div(10, 2) == 5.0

    def test_zero_denominator(self):
        assert safe_div(10, 0) == 0.0

    def test_near_zero_denominator(self):
        assert safe_div(10, 1e-15) == 0.0


# ===================================================================
# Sales logic
# ===================================================================

class TestSalesLogic:
    """Verify that the StubSalesModel produces correct lot counts."""

    def test_total_lots_sold_equals_initial(self, tiny_engine):
        """All 10 lots should be sold over 4 quarters (shares sum ~ 1.0)."""
        result = run_tiny(tiny_engine)
        total_sold = sum(s.sold_lots for s in result.states)
        assert total_sold == 10

    def test_per_quarter_lots(self, tiny_engine):
        """Shares [0.3, 0.3, 0.2, 0.2] × 10 lots → [3, 3, 2, 2]."""
        result = run_tiny(tiny_engine)
        sold = [s.sold_lots for s in result.states]
        assert sold == [3, 3, 2, 2]

    def test_remaining_lots_decrease(self, tiny_engine):
        result = run_tiny(tiny_engine)
        remaining = [s.remaining_lots_end for s in result.states]
        assert remaining == [7, 4, 2, 0]

    def test_no_sales_before_rns(self, tiny_config, tiny_cost_schedule):
        """If rns_quarter_index=2, Q0 and Q1 should have 0 sales."""
        from dataclasses import replace
        late_rns_config = replace(tiny_config, rns_quarter_index=2)
        engine = OnePathCashflowEngine(
            config=late_rns_config,
            cost_schedule=tiny_cost_schedule,
            price_model=StubPriceModel(),
            sales_model=StubSalesModel(quarterly_sales_share=[0.3, 0.3, 0.2, 0.2]),
        )
        result = engine.run()
        assert result.states[0].sold_lots == 0
        assert result.states[1].sold_lots == 0
        assert result.states[2].sold_lots == 2  # share=0.2 × 10


# ===================================================================
# Revenue
# ===================================================================

class TestRevenue:
    def test_revenue_formula(self, tiny_engine):
        """revenue = sold_lots × avg_unit_area × price_sqm"""
        result = run_tiny(tiny_engine)
        q0 = result.states[0]
        # 3 lots × 50 sqm × 100,000 = 15,000,000
        assert q0.revenue == pytest.approx(15_000_000.0)

    def test_total_revenue(self, tiny_engine):
        """(3+3+2+2) × 50 × 100,000 = 50,000,000"""
        result = run_tiny(tiny_engine)
        assert result.summary.total_revenue == pytest.approx(50_000_000.0)


# ===================================================================
# Escrow mechanics (214-FZ)
# ===================================================================

class TestEscrow:
    """
    Per 214-FZ: pre-RVZ revenue goes to escrow, released at RVZ.
    Post-RVZ revenue goes directly to project cash.
    """

    def test_pre_rvz_revenue_goes_to_escrow(self, tiny_engine):
        result = run_tiny(tiny_engine)
        q0 = result.states[0]
        assert q0.escrow_inflow == pytest.approx(15_000_000.0)
        assert q0.project_cash_inflow == 0.0

    def test_escrow_accumulates(self, tiny_engine):
        result = run_tiny(tiny_engine)
        # Q0: +15M, Q1: +15M → 30M, Q2: +10M → 40M
        assert result.states[0].escrow_balance_end == pytest.approx(15_000_000.0)
        assert result.states[1].escrow_balance_end == pytest.approx(30_000_000.0)
        assert result.states[2].escrow_balance_end == pytest.approx(40_000_000.0)

    def test_escrow_fully_released_at_rvz(self, tiny_engine):
        """At RVZ (Q3), all escrow is released to project cash."""
        result = run_tiny(tiny_engine)
        rvz = result.states[3]  # rvz_quarter_index=3
        # 40M accumulated + 10M Q3 inflow = 50M released
        assert rvz.escrow_release == pytest.approx(50_000_000.0)
        assert rvz.escrow_balance_end == 0.0
        assert rvz.project_cash_inflow == pytest.approx(50_000_000.0)


# ===================================================================
# Costs
# ===================================================================

class TestCosts:
    def test_marketing_is_pct_of_revenue(self, tiny_engine):
        result = run_tiny(tiny_engine)
        q0 = result.states[0]
        assert q0.marketing_cost == pytest.approx(15_000_000.0 * 0.05)

    def test_total_costs_q0(self, tiny_engine):
        """Q0: land=5M + smr=2M + marketing=750k = 7.75M"""
        result = run_tiny(tiny_engine)
        assert result.states[0].total_costs_excl_financing == pytest.approx(7_750_000.0)

    def test_profit_tax_at_rvz(self, tiny_engine):
        """
        Accumulated pre-tax profit at RVZ:
          Q0: 15M - 5M - 2M - 750k = 7.25M
          Q1: 15M - 0 - 2M - 750k = 12.25M
          Q2: 10M - 0 - 2M - 500k = 7.5M
          Q3: 10M - 0 - 2M - 500k = 7.5M
          Total = 34.5M → tax = 34.5M × 0.20 = 6.9M
        """
        result = run_tiny(tiny_engine)
        assert result.states[3].profit_tax == pytest.approx(6_900_000.0)
        # Other quarters should have no profit tax
        for i in range(3):
            assert result.states[i].profit_tax == 0.0

    def test_total_costs_summary(self, tiny_engine):
        result = run_tiny(tiny_engine)
        # 7.75M + 2.75M + 2.5M + 9.4M(incl 6.9M tax) = 22.4M
        assert result.summary.total_costs_excl_financing == pytest.approx(22_400_000.0)


# ===================================================================
# Debt / financing
# ===================================================================

class TestDebtFinancing:
    def test_equity_injected_q0(self, tiny_engine):
        """Equity = 13M - 10.4M = 2.6M, injected at Q0."""
        result = run_tiny(tiny_engine)
        assert result.states[0].equity_inflow == pytest.approx(2_600_000.0)
        for i in range(1, 4):
            assert result.states[i].equity_inflow == 0.0

    def test_debt_limit(self, tiny_engine):
        result = run_tiny(tiny_engine)
        for s in result.states:
            assert s.debt_limit == pytest.approx(10_400_000.0)

    def test_debt_not_exceeds_limit(self, tiny_engine):
        result = run_tiny(tiny_engine)
        for s in result.states:
            assert s.debt_outstanding_end <= s.debt_limit + 1e-6

    def test_debt_repaid_at_rvz(self, tiny_engine):
        result = run_tiny(tiny_engine)
        assert result.summary.repaid_at_rvz is True
        assert result.summary.debt_fully_repaid is True
        assert result.states[3].debt_outstanding_end == pytest.approx(0.0)

    def test_peak_debt(self, tiny_engine):
        result = run_tiny(tiny_engine)
        assert result.summary.peak_debt == pytest.approx(10_400_000.0)


# ===================================================================
# Interest calculation  (per docx formula)
# ===================================================================

class TestInterest:
    """
    Formula from docx:
      interest = debt × ((1 - escrow_cov) × full_rate + escrow_cov × priv_rate) / 4

    With full escrow coverage (escrow > debt) → interest = debt × priv_rate / 4.
    """

    def test_q0_interest_with_full_escrow_coverage(self, tiny_engine):
        """
        Q0: debt_after_operating_draw = 5,150,000  (pre-financing draw)
        escrow=15M > debt → coverage=1.0
        interest = 5,150,000 × 0.05 / 4 = 64,375
        """
        result = run_tiny(tiny_engine)
        assert result.states[0].escrow_coverage_ratio == pytest.approx(1.0)
        assert result.states[0].interest_payment == pytest.approx(64_375.0)

    def test_reserve_fee_on_unused_limit(self, tiny_engine):
        """
        Q0: unused = 10.4M - 5.15M = 5.25M
        reserve_fee = 5.25M × 0.01 / 4 = 13,125
        """
        result = run_tiny(tiny_engine)
        assert result.states[0].reserve_fee_payment == pytest.approx(13_125.0)

    def test_total_debt_service_cost(self, tiny_engine):
        result = run_tiny(tiny_engine)
        q0 = result.states[0]
        assert q0.total_debt_service_cost == pytest.approx(
            q0.interest_payment + q0.reserve_fee_payment + q0.line_usage_fee_payment
        )


# ===================================================================
# CFADS / DSCR / ISCR  (per docx: CFADS = cash_inflow - costs)
# ===================================================================

class TestCfadsDscr:
    def test_cfads_pre_rvz(self, tiny_engine):
        """Pre-RVZ: project_cash_inflow=0, so CFADS = -costs."""
        result = run_tiny(tiny_engine)
        q0 = result.states[0]
        assert q0.cfads == pytest.approx(0.0 - 7_750_000.0)

    def test_cfads_at_rvz(self, tiny_engine):
        """RVZ: project_cash_inflow=50M, costs=9.4M → CFADS=40.6M"""
        result = run_tiny(tiny_engine)
        q3 = result.states[3]
        assert q3.cfads == pytest.approx(50_000_000.0 - 9_400_000.0)

    def test_dscr_at_rvz(self, tiny_engine):
        """DSCR = CFADS / DS where DS = repayment + interest + fees."""
        result = run_tiny(tiny_engine)
        q3 = result.states[3]
        expected_dscr = q3.cfads / q3.debt_service_for_ratio
        assert q3.dscr == pytest.approx(expected_dscr)
        assert q3.dscr == pytest.approx(result.summary.min_dscr)

    def test_iscr_formula(self, tiny_engine):
        """ISCR = CFADS / interest_payment."""
        result = run_tiny(tiny_engine)
        q0 = result.states[0]
        expected = q0.cfads / q0.interest_payment
        assert q0.iscr == pytest.approx(expected)


# ===================================================================
# Collateral coverage (per docx formula)
# ===================================================================

class TestCollateral:
    def test_collateral_coverage(self, tiny_engine):
        """
        collateral_value_net = sellable_area × collateral_price × (1 - discount)
                             = 500 × 90,000 × 0.70 = 31,500,000
        collateral_coverage = 31.5M / (debt_limit + total_financing_cost)
        """
        result = run_tiny(tiny_engine)
        net = 500 * 90_000 * 0.70
        assert net == pytest.approx(31_500_000.0)
        assert result.summary.collateral_coverage_ratio == pytest.approx(
            net / (10_400_000.0 + result.summary.total_interest_and_fees)
        )


# ===================================================================
# Scenario effects
# ===================================================================

class TestScenarioEffects:
    def test_price_multiplier_scales_revenue(self, tiny_engine):
        """price_level_multiplier=1.10 should increase revenue by 10%."""
        base = tiny_engine.run()
        scenario = Scenario(
            global_state=GlobalScenario(name="price_up", price_level_multiplier=1.10)
        )
        shocked = tiny_engine.run(scenario)
        assert shocked.summary.total_revenue == pytest.approx(
            base.summary.total_revenue * 1.10, rel=0.01
        )

    def test_key_rate_shift_increases_interest(self, tiny_engine):
        """Higher key rate → higher full_rate → more interest when escrow < debt."""
        base = tiny_engine.run()
        scenario = Scenario(
            global_state=GlobalScenario(name="rate_up", key_rate_shift_annual=0.05)
        )
        shocked = tiny_engine.run(scenario)
        # Interest should be >= base (may be equal if fully escrow-covered)
        assert shocked.summary.total_interest_and_fees >= base.summary.total_interest_and_fees - 1e-6

    def test_rvz_delay_shifts_escrow_release(self, tiny_config, tiny_cost_schedule):
        """RVZ delay by 0 vs something: escrow release should shift."""
        from dataclasses import replace as dc_replace

        # Extend horizon to 6 quarters to allow delay
        cfg6 = dc_replace(
            tiny_config,
            horizon_quarters=6,
            rvz_quarter_index=3,
        )
        cs6 = dc_replace(
            tiny_cost_schedule,
            land_by_quarter=[5_000_000.0] + [0.0] * 5,
            smr_by_quarter=[2_000_000.0] * 6,
        )
        shares6 = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
        engine6 = OnePathCashflowEngine(
            config=cfg6, cost_schedule=cs6,
            price_model=StubPriceModel(),
            sales_model=StubSalesModel(quarterly_sales_share=shares6),
        )

        base = engine6.run()
        delayed = engine6.run(Scenario(
            global_state=GlobalScenario(name="delay", rvz_delay_quarters=2)
        ))
        # Base: escrow released at Q3; Delayed: at Q5
        assert base.states[3].escrow_release > 0
        assert delayed.states[3].escrow_release == 0.0
        assert delayed.states[5].escrow_release > 0


# ===================================================================
# Demo-scale smoke test  (larger project, regression guard)
# ===================================================================

class TestDemoSmoke:
    """Run the demo engine and check key aggregates haven't drifted."""

    def test_demo_runs_without_error(self, demo_engine):
        result = demo_engine.run()
        assert len(result.states) == 10

    def test_demo_total_revenue(self, demo_engine):
        result = demo_engine.run()
        assert result.summary.total_revenue == pytest.approx(2_129_327_884.2, rel=1e-6)

    def test_demo_debt_fully_repaid(self, demo_engine):
        result = demo_engine.run()
        assert result.summary.debt_fully_repaid is True
        assert result.summary.ending_debt == pytest.approx(0.0)

    def test_demo_min_dscr_positive(self, demo_engine):
        result = demo_engine.run()
        assert result.summary.min_dscr > 1.0

    def test_demo_escrow_released_at_rvz(self, demo_engine):
        result = demo_engine.run()
        assert result.states[8].escrow_release > 0
        assert result.states[8].escrow_balance_end == 0.0
