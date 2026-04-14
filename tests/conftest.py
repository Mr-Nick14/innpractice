"""
Synthetic test fixtures for mc_cashflow_engine.

All data here is hand-crafted so that expected values can be verified
with pencil-and-paper arithmetic.  No real project data is used.
"""

import pytest
import sys
from pathlib import Path

# Ensure the cashflow package root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo.mc_cashflow_engine import (
    CostSchedule,
    GlobalScenario,
    LocalScenarioPath,
    OnePathCashflowEngine,
    ProjectConfig,
    Scenario,
    StubPriceModel,
    StubSalesModel,
)

# ---------------------------------------------------------------------------
# Tiny 4-quarter project  (easy to verify by hand)
# ---------------------------------------------------------------------------
#
#   Quarters:  Q1   Q2   Q3   Q4(=RVZ)
#   Sales:     3    3    2    2  lots  (out of 10)
#   Price:     100k / sqm,  area per lot = 50 sqm
#   Revenue:   15M  15M  10M  10M
#
#   Costs (land Q1 only, SMR every quarter):
#     land  = [5M, 0, 0, 0]
#     smr   = [2M, 2M, 2M, 2M]   total = 8M
#     marketing = 5% of revenue
#
#   Financing:
#     total_project_cost = 5M + 8M = 13M
#     LTC = 0.80  →  debt_limit = 10.4M
#     equity = 13M - 10.4M = 2.6M  (injected Q1)
#     key_rate = 0.20,  spread = 0.03 → full_rate = 0.23
#     privileged_rate = 0.05
#     reserve_fee = 0.01
# ---------------------------------------------------------------------------

TINY_HORIZON = 4
TINY_PRICE = 100_000.0
TINY_AVG_AREA = 50.0
TINY_LOTS = 10
TINY_SALES_SHARES = [0.3, 0.3, 0.2, 0.2]


@pytest.fixture
def tiny_config() -> ProjectConfig:
    return ProjectConfig(
        name="tiny_test",
        start_year=2026,
        start_quarter=1,
        horizon_quarters=TINY_HORIZON,
        rns_quarter_index=0,
        rvz_quarter_index=3,
        sellable_area_sqm=TINY_LOTS * TINY_AVG_AREA,
        avg_unit_area_sqm=TINY_AVG_AREA,
        initial_remaining_lots=TINY_LOTS,
        key_rate_annual=0.20,
        full_rate_spread_before_rvz=0.03,
        full_rate_spread_after_rvz=0.03,
        privileged_rate_annual=0.05,
        reserve_fee_annual=0.01,
        ltc=0.80,
        land_cost_total=5_000_000.0,
        smr_cost_total=8_000_000.0,
        design_cost_total=0.0,
        marketing_cost_ratio=0.05,
        profit_tax_rate=0.20,
        initial_price_sqm=TINY_PRICE,
        collateral_discount=0.30,
        collateral_price_sqm=90_000.0,
    )


@pytest.fixture
def tiny_cost_schedule() -> CostSchedule:
    return CostSchedule(
        land_by_quarter=[5_000_000.0, 0.0, 0.0, 0.0],
        smr_by_quarter=[2_000_000.0, 2_000_000.0, 2_000_000.0, 2_000_000.0],
    )


@pytest.fixture
def tiny_sales_model() -> StubSalesModel:
    return StubSalesModel(quarterly_sales_share=TINY_SALES_SHARES)


@pytest.fixture
def tiny_engine(tiny_config, tiny_cost_schedule, tiny_sales_model) -> OnePathCashflowEngine:
    return OnePathCashflowEngine(
        config=tiny_config,
        cost_schedule=tiny_cost_schedule,
        price_model=StubPriceModel(),
        sales_model=tiny_sales_model,
    )


# ---------------------------------------------------------------------------
# Demo project fixture (matches make_demo_inputs from the engine module)
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_config() -> ProjectConfig:
    return ProjectConfig(
        name="demo_test",
        start_year=2026,
        start_quarter=1,
        horizon_quarters=10,
        rns_quarter_index=0,
        rvz_quarter_index=8,
        sellable_area_sqm=11_113.0,
        avg_unit_area_sqm=50.29,
        initial_remaining_lots=221,
        key_rate_annual=0.21,
        full_rate_spread_before_rvz=0.035,
        full_rate_spread_after_rvz=0.035,
        privileged_rate_annual=0.037,
        reserve_fee_annual=0.005,
        ltc=0.926,
        land_cost_total=125_637_000.0,
        smr_cost_total=1_034_039_000.0,
        design_cost_total=0.0,
        marketing_cost_ratio=0.05,
        profit_tax_rate=0.25,
        initial_price_sqm=192_459.0,
        collateral_discount=0.30,
        collateral_price_sqm=168_000.0,
    )


@pytest.fixture
def demo_cost_schedule() -> CostSchedule:
    return CostSchedule(
        land_by_quarter=[125_637_000.0] + [0.0] * 9,
        smr_by_quarter=[
            0.0, 246_434_000.0, 112_515_000.0, 112_515_000.0,
            112_515_000.0, 112_515_000.0, 112_515_000.0, 112_515_000.0,
            112_515_000.0, 0.0,
        ],
    )


@pytest.fixture
def demo_sales_shares() -> list:
    return [0.118, 0.138, 0.144, 0.144, 0.138, 0.128, 0.110, 0.081, 0.0, 0.0]


@pytest.fixture
def demo_engine(demo_config, demo_cost_schedule, demo_sales_shares) -> OnePathCashflowEngine:
    return OnePathCashflowEngine(
        config=demo_config,
        cost_schedule=demo_cost_schedule,
        price_model=StubPriceModel(),
        sales_model=StubSalesModel(quarterly_sales_share=demo_sales_shares),
    )
