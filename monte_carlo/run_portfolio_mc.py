from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .hedonic_runtime import (
    HedonicPriceModel,
    ProjectCostStubConfig,
    ProjectInputAdapter,
)
from .mc_cashflow_engine import (
    MonteCarloRunner,
    RandomScenarioSpec,
    ScenarioFactory,
    StubSalesModel,
    FittedNBFeatureSalesModel,
    OnePathCashflowEngine,
    build_equal_share_schedule,
)
from .mc_stochastic_drivers import DriverScenarioFactory, MonteCarloDriverConfig


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monte Carlo cashflow for multiple projects with hedonic runtime integration.")
    parser.add_argument(
        "--deals-parquet",
        type=Path,
        default=Path(r"C:\proga\gazprom_tex\data\deals_with_real_geo_Moscow.parquet"),
        help="Deals parquet used as project source and reference dataset.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(r"C:\proga\gazprom_tex\data\model_artifacts"),
        help="Directory with persisted hedonic artifacts.",
    )
    parser.add_argument("--region-key", type=str, default="Moscow")
    parser.add_argument("--top-n", type=int, default=5, help="Run top-N projects by number of deals.")
    parser.add_argument(
        "--project-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of project_id values. If set, --top-n is ignored.",
    )
    parser.add_argument("--horizon-quarters", type=int, default=10)
    parser.add_argument("--mc-runs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--market-mode", type=str, default="stochastic", choices=["replay", "trend", "stochastic", "scenario_stochastic", "macro_guided"])
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output folder for CSV summary.")
    parser.add_argument("--use-stub-sales", action="store_true", help="Force StubSalesModel for all projects.")
    parser.add_argument("--use-driver-layer", action="store_true", help="Enable the stochastic driver layer.")
    parser.add_argument("--driver-use-correlations", action="store_true", help="Enable correlated shocks inside the driver layer.")
    return parser.parse_args()


def _default_mc_spec() -> RandomScenarioSpec:
    return RandomScenarioSpec(
        key_rate_shift_min=-0.01,
        key_rate_shift_max=0.02,
        price_level_min=0.95,
        price_level_max=1.08,
        sales_level_min=0.85,
        sales_level_max=1.15,
        project_price_premium_min=-0.02,
        project_price_premium_max=0.03,
        rvz_delay_choices=(0, 1),
        local_price_shock_std=0.01,
        local_sales_shock_std=0.05,
    )


def _prepare_output_dir(path: Optional[Path]) -> Path:
    if path is not None:
        out = path.resolve()
    else:
        out = Path(__file__).resolve().parent / "portfolio_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_sales_model(
    *,
    region_key: str,
    class_group: Optional[str],
    initial_price_sqm: float,
    seed: int,
    use_stub_sales: bool,
    horizon_quarters: int,
) -> tuple[Any, bool]:
    if use_stub_sales:
        return StubSalesModel(quarterly_sales_share=build_equal_share_schedule(horizon_quarters, total_share=0.95)), True
    try:
        model = FittedNBFeatureSalesModel.from_csv(
            region=_region_to_nb_key(region_key),
            class_group=class_group or "Комфорт",
            market_price_sqm=initial_price_sqm,
            stochastic=True,
            seed=seed,
        )
        return model, False
    except Exception:
        return StubSalesModel(quarterly_sales_share=build_equal_share_schedule(horizon_quarters, total_share=0.95)), True


def run_portfolio_mc(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    adapter = ProjectInputAdapter(args.deals_parquet)
    if args.project_ids:
        selected_project_ids = [str(p) for p in args.project_ids]
    else:
        top = adapter.list_top_projects(args.top_n)
        selected_project_ids = top["project_id"].astype(str).tolist()

    mc_spec = _default_mc_spec()
    summary_rows: List[Dict[str, Any]] = []

    for idx, pid in enumerate(selected_project_ids):
        project_input = adapter.build_project_input(project_id=pid, region_key=args.region_key)
        cfg, cost_schedule = adapter.build_stub_project_config_and_schedule(
            project_input=project_input,
            cost_stub=ProjectCostStubConfig(horizon_quarters=args.horizon_quarters),
        )
        price_model = HedonicPriceModel.from_artifacts(
            project_input=project_input,
            artifact_dir=args.artifacts_dir,
            reference_deals_path=args.deals_parquet,
            market_mode=args.market_mode,
            market_seed=args.seed + idx,
        )
        sales_model, used_stub_sales = _build_sales_model(
            region_key=args.region_key,
            class_group=project_input.class_group,
            initial_price_sqm=cfg.initial_price_sqm,
            seed=args.seed + idx,
            use_stub_sales=args.use_stub_sales,
            horizon_quarters=cfg.horizon_quarters,
        )
        driver_factory = None
        if args.use_driver_layer:
            driver_factory = DriverScenarioFactory(
                MonteCarloDriverConfig(
                    market_path_mode=args.market_mode,
                    use_correlations=args.driver_use_correlations,
                ),
                seed=args.seed + idx,
            )
        engine = OnePathCashflowEngine(
            config=cfg,
            cost_schedule=cost_schedule,
            price_model=price_model,
            sales_model=sales_model,
        )
        base = engine.run()
        mc = MonteCarloRunner(engine, ScenarioFactory(seed=args.seed + idx)).run_random(
            n_runs=args.mc_runs,
            spec=mc_spec,
            name_prefix=f"pid_{pid[:8]}",
            driver_factory=driver_factory,
        )

        # Missing data / synthetic fallback diagnostics.
        missing_items: List[str] = []
        if project_input.building_lat is None or project_input.building_lon is None:
            missing_items.append("building_lat_lon")
        if price_model.artifacts.geo_model is None:
            missing_items.append("geo_model_artifact")
        if price_model.artifacts.quality_model is None:
            missing_items.append("quality_model_artifact")
        if price_model.artifacts.h3_encoder is None:
            missing_items.append("h3_encoder_artifact")
        if price_model.artifacts.lgb_residual_model is None:
            missing_items.append("lgb_residual_model_artifact")
        if price_model.artifacts.final_model_pipeline is None:
            missing_items.append("final_model_artifact")
        if getattr(price_model, "_final_model_disabled", False):
            missing_items.append("final_model_inference_incompatible")
        if used_stub_sales:
            missing_items.append("nb_sales_model_or_class_mapping")

        premium_found = False
        premium_table = price_model.artifacts.project_premium_table
        if premium_table is not None and "project_id" in premium_table.columns:
            premium_found = bool((premium_table["project_id"].astype(str) == str(project_input.project_id)).any())
        if not premium_found:
            missing_items.append("project_premium_row")

        summary_rows.append(
            {
                "project_id": project_input.project_id,
                "project_name": project_input.project_name,
                "class_group": project_input.class_group,
                "horizon_quarters": cfg.horizon_quarters,
                "mc_runs": mc.summary.runs,
                "base_total_revenue": base.summary.total_revenue,
                "base_peak_debt": base.summary.peak_debt,
                "base_ending_debt": base.summary.ending_debt,
                "base_min_dscr": base.summary.min_dscr,
                "base_ending_cash": base.summary.ending_project_cash_balance,
                "mc_mean_min_dscr": mc.summary.mean_min_dscr,
                "mc_p05_min_dscr": mc.summary.p05_min_dscr,
                "mc_p50_min_dscr": mc.summary.p50_min_dscr,
                "mc_p95_min_dscr": mc.summary.p95_min_dscr,
                "mc_prob_dscr_lt_1_0": mc.summary.probability_dscr_below_1_0,
                "mc_prob_dscr_lt_1_2": mc.summary.probability_dscr_below_1_2,
                "mc_mean_peak_debt": mc.summary.mean_peak_debt,
                "mc_mean_ending_debt": mc.summary.mean_ending_debt,
                "mc_mean_final_effective_rate_annual": mc.summary.mean_final_effective_rate_annual,
                "driver_layer_enabled": bool(args.use_driver_layer),
                "driver_use_correlations": bool(args.driver_use_correlations),
                "missing_data_or_artifacts": ";".join(missing_items),
                "artifact_warnings_count": len(price_model.artifacts.warnings_log),
                "artifact_warnings": " | ".join(price_model.artifacts.warnings_log[:8]),
            }
        )

    out_dir = _prepare_output_dir(args.output_dir)
    out_csv = out_dir / "portfolio_mc_summary.csv"
    out_df = pd.DataFrame(summary_rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    return out_df, out_dir


def main() -> None:
    args = parse_args()
    out_df, out_dir = run_portfolio_mc(args)
    print("=== Portfolio MC Summary ===")
    print(out_df.to_string(index=False))
    print()
    print(f"Saved: {out_dir / 'portfolio_mc_summary.csv'}")
    print()
    print("Missing data / synthetic fallbacks legend:")
    print("  building_lat_lon: coordinates are absent in source data; H3/kNN fallback defaults were used.")
    print("  quality_model_artifact: quality block disabled -> quality_score=0 fallback.")
    print("  h3_encoder_artifact: H3 block disabled -> H3 fallback defaults.")
    print("  lgb_residual_model_artifact/final_model_artifact: residual boost disabled or fallback failed -> lgb_boost=0.")
    print("  project_premium_row: project not found in premium table -> premium=0.")
    print("  nb_sales_model_or_class_mapping: NB sales model not loaded; StubSalesModel used.")
    if args.use_driver_layer:
        print("  driver_layer_enabled: stochastic path-level and quarter-level drivers were applied.")


if __name__ == "__main__":
    main()
