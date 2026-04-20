from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import random

import pandas as pd

from .hedonic_runtime import (
    HedonicPriceModel,
    ProjectCostStubConfig,
    ProjectInputAdapter,
)
from .mc_cashflow_engine import (
    FittedNBFeatureSalesModel,
    GlobalScenario,
    LocalScenarioPath,
    OnePathCashflowEngine,
    Scenario,
    StubSalesModel,
    build_equal_share_schedule,
)
from .mc_stochastic_drivers import DriverScenarioFactory, MonteCarloDriverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-project hedonic-integrated Monte Carlo cashflow path.")
    parser.add_argument("--project-id", type=str, default=None, help="Project id from deals parquet.")
    parser.add_argument("--project-name", type=str, default=None, help="Project name fallback if project_id is omitted.")
    parser.add_argument(
        "--deals-parquet",
        type=Path,
        default=Path(r"C:\proga\gazprom_tex\data\deals_with_real_geo_Moscow.parquet"),
        help="Reference deals parquet.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(r"C:\proga\gazprom_tex\data\model_artifacts"),
        help="Directory with saved hedonic artifacts.",
    )
    parser.add_argument("--region-key", type=str, default="Moscow")
    parser.add_argument("--market-mode", type=str, default="stochastic", choices=["replay", "trend", "stochastic", "scenario_stochastic", "macro_guided"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic market/sales.")
    parser.add_argument("--horizon-quarters", type=int, default=12)
    parser.add_argument("--price-level-multiplier", type=float, default=1.0)
    parser.add_argument("--sales-level-multiplier", type=float, default=1.0)
    parser.add_argument("--project-price-premium", type=float, default=0.0)
    parser.add_argument("--key-rate-shift", type=float, default=0.0)
    parser.add_argument("--local-price-shock-std", type=float, default=0.0)
    parser.add_argument("--local-sales-shock-std", type=float, default=0.0)
    parser.add_argument("--use-driver-layer", action="store_true", help="Enable the stochastic driver layer.")
    parser.add_argument("--driver-use-correlations", action="store_true", help="Enable correlated driver shocks.")
    parser.add_argument("--use-stub-sales", action="store_true", help="Force StubSalesModel instead of NB inference model.")
    parser.add_argument("--save-states-csv", type=Path, default=None, help="Optional output CSV for quarter states.")
    parser.add_argument("--print-top-projects", type=int, default=0, help="If >0 prints top projects and exits.")
    return parser.parse_args()


def _region_to_nb_key(region_key: str) -> str:
    key = region_key.strip().lower()
    mapping = {
        "moscow": "msk",
        "москва": "msk",
        "spb": "spb",
        "saint petersburg": "spb",
        "sankt-peterburg": "spb",
        "санкт-петербург": "spb",
        "krasnodar": "krd",
        "краснодар": "krd",
    }
    return mapping.get(key, "msk")


def _build_scenario(args: argparse.Namespace, horizon: int) -> Scenario:
    rng = random.Random(args.seed)
    if args.local_price_shock_std > 0:
        price_shocks = [rng.gauss(0.0, args.local_price_shock_std) for _ in range(horizon)]
    else:
        price_shocks = [0.0] * horizon
    if args.local_sales_shock_std > 0:
        sales_shocks = [rng.gauss(0.0, args.local_sales_shock_std) for _ in range(horizon)]
    else:
        sales_shocks = [0.0] * horizon

    global_state = GlobalScenario(
        name="hedonic_run",
        key_rate_shift_annual=args.key_rate_shift,
        price_level_multiplier=args.price_level_multiplier,
        sales_level_multiplier=args.sales_level_multiplier,
        project_price_premium=args.project_price_premium,
    )
    return Scenario(
        global_state=global_state,
        local_path=LocalScenarioPath(price_shocks=price_shocks, sales_shocks=sales_shocks),
    )


def main() -> None:
    args = parse_args()
    adapter = ProjectInputAdapter(args.deals_parquet)

    if args.print_top_projects > 0:
        top = adapter.list_top_projects(args.print_top_projects)
        print(top.to_string(index=False))
        return

    if not args.project_id and not args.project_name:
        raise SystemExit("Provide --project-id or --project-name (or use --print-top-projects N).")

    project_input = adapter.build_project_input(
        project_id=args.project_id,
        project_name=args.project_name,
        region_key=args.region_key,
    )
    cost_stub = ProjectCostStubConfig(horizon_quarters=args.horizon_quarters)
    config, cost_schedule = adapter.build_stub_project_config_and_schedule(
        project_input=project_input,
        cost_stub=cost_stub,
    )

    price_model = HedonicPriceModel.from_artifacts(
        project_input=project_input,
        artifact_dir=args.artifacts_dir,
        reference_deals_path=args.deals_parquet,
        market_mode=args.market_mode,
        market_seed=args.seed,
    )

    sales_model = None
    if not args.use_stub_sales:
        try:
            sales_model = FittedNBFeatureSalesModel.from_csv(
                region=_region_to_nb_key(args.region_key),
                class_group=project_input.class_group or "Комфорт",
                market_price_sqm=config.initial_price_sqm,
                stochastic=True,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"[WARN] Failed to load NB sales model, fallback to StubSalesModel: {exc}")

    if sales_model is None:
        sales_model = StubSalesModel(
            quarterly_sales_share=build_equal_share_schedule(config.horizon_quarters, total_share=0.95),
        )

    scenario = _build_scenario(args, config.horizon_quarters)
    if args.use_driver_layer:
        driver_factory = DriverScenarioFactory(
            MonteCarloDriverConfig(
                market_path_mode=args.market_mode,
                use_correlations=args.driver_use_correlations,
            ),
            seed=args.seed,
        )
        scenario = replace(
            scenario,
            driver_scenario=driver_factory.make(
                name="hedonic_run_drivers",
                horizon=config.horizon_quarters,
            ),
        )

    engine = OnePathCashflowEngine(
        config=config,
        cost_schedule=cost_schedule,
        price_model=price_model,
        sales_model=sales_model,
    )
    result = engine.run(scenario)

    print("=== Project Input ===")
    print(f"project_id: {project_input.project_id}")
    print(f"project_name: {project_input.project_name}")
    print(f"class_group (for NB sales): {project_input.class_group}")
    print(f"sellable_area_sqm: {project_input.sellable_area_sqm:,.0f}")
    print(f"avg_unit_area_sqm: {project_input.avg_unit_area_sqm:.2f}")
    print()

    print("=== Path Summary ===")
    print(f"total_revenue: {result.summary.total_revenue:,.0f}")
    print(f"peak_debt: {result.summary.peak_debt:,.0f}")
    print(f"ending_debt: {result.summary.ending_debt:,.0f}")
    print(f"min_dscr: {result.summary.min_dscr:.3f}")
    print(f"ending_project_cash_balance: {result.summary.ending_project_cash_balance:,.0f}")
    print()
    if result.summary.driver_summary:
        print("=== Driver Summary ===")
        for key, value in result.summary.driver_summary.items():
            print(f"{key}: {value:.4f}")
        print()

    print("=== Quarterly Price Components ===")
    print(
        "quarter | price_sqm | market_log | geo | quality | premium | lgb | sold_lots | revenue"
    )
    for st in result.states:
        print(
            f"{st.quarter_id.label():>7} | "
            f"{st.price_sqm:>10.0f} | "
            f"{st.market_log_price:>10.4f} | "
            f"{st.geo_score:>7.4f} | "
            f"{st.quality_score:>7.4f} | "
            f"{st.project_premium_component:>7.4f} | "
            f"{st.lgb_boost:>7.4f} | "
            f"{st.sold_lots:>9d} | "
            f"{st.revenue:>12.0f}"
        )

    if args.save_states_csv is not None:
        rows = []
        for st in result.states:
            row = asdict(st)
            row["quarter_label"] = st.quarter_id.label()
            row["quarter_year"] = st.quarter_id.year
            row["quarter_number"] = st.quarter_id.quarter
            row.pop("quarter_id", None)
            rows.append(row)
        out = args.save_states_csv.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
        print()
        print(f"Saved states to: {out}")

    if price_model.artifacts.warnings_log:
        print()
        print("=== Artifact / Runtime Warnings ===")
        for w in price_model.artifacts.warnings_log:
            print(f"- {w}")
        if price_model.last_missing_lgb_columns:
            print(f"- Missing LGB columns filled with fallback: {', '.join(price_model.last_missing_lgb_columns[:20])}")


if __name__ == "__main__":
    main()
