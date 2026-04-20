from __future__ import annotations

import argparse
from pathlib import Path

from .hedonic_runtime import demo_run_hedonic_integration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hedonic + Monte Carlo integration demo.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(r"C:\proga\gazprom_tex\data\model_artifacts"),
        help="Directory with persisted hedonic artifacts.",
    )
    parser.add_argument(
        "--reference-deals",
        type=Path,
        default=Path(r"C:\proga\gazprom_tex\data\deals_with_real_geo_Moscow.parquet"),
        help="Reference deals parquet for feature defaults and spatial context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = demo_run_hedonic_integration(
        artifacts_dir=args.artifacts_dir,
        reference_deals_path=args.reference_deals,
    )
    summary = output["summary"]
    print("=== Hedonic Integration Demo ===")
    print(f"Project: {summary.project_name}")
    print(f"Runs one deterministic path. Ending debt: {summary.ending_debt:,.0f}")
    print(f"Min DSCR: {summary.min_dscr:.3f}")
    print(f"Warnings: {len(output['artifact_warnings'])}")
    if output["artifact_warnings"]:
        print("Artifact warnings:")
        for w in output["artifact_warnings"]:
            print(f"  - {w}")
    print("\nQuarterly price component snapshot:")
    for row in output["price_components"][:5]:
        print(
            f"{row['quarter']}: market={row['market_log_price']:.4f}, "
            f"geo={row['geo_score']:.4f}, quality={row['quality_score']:.4f}, "
            f"premium={row['project_premium_component']:.4f}, "
            f"lgb={row['lgb_boost']:.4f}, price={row['price_sqm']:.0f}"
        )


if __name__ == "__main__":
    main()

