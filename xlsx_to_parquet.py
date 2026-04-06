from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "data"
SOURCE_DIRS = {"deals", "projects"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected XLSX files from deals/projects to parquet."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help=(
            "XLSX files to convert. You can pass full paths, "
            "or just file names from data/deals or data/projects."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root directory with deals/projects folders (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers. 0 = auto.",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression (default: snappy).",
    )
    return parser.parse_args()


def resolve_input_path(file_arg: str, data_root: Path) -> Path:
    candidate = Path(file_arg).expanduser()
    if not candidate.suffix:
        candidate = candidate.with_suffix(".xlsx")

    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.append(Path.cwd() / candidate)
        search_paths.extend(data_root / src / candidate.name for src in SOURCE_DIRS)
        search_paths.extend(data_root / src / candidate for src in SOURCE_DIRS)

    for path in search_paths:
        if path.exists() and path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"File not found: {file_arg}")


def detect_source_group(input_path: Path, data_root: Path) -> str:
    try:
        rel = input_path.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"File is outside data root ({data_root}): {input_path}"
        ) from exc

    if not rel.parts:
        raise ValueError(f"Invalid file location: {input_path}")

    source_group = rel.parts[0]
    if source_group not in SOURCE_DIRS:
        raise ValueError(
            f"File must be inside one of {sorted(SOURCE_DIRS)}: {input_path}"
        )
    return source_group


def _to_datetime_series(series: pd.Series) -> pd.Series:
    x = series.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    parsed = pd.to_datetime(x, errors="coerce", dayfirst=True, utc=True)
    if hasattr(parsed, "dt"):
        parsed = parsed.dt.tz_localize(None)
    num = pd.to_numeric(x, errors="coerce")
    excel_dt = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return parsed.fillna(excel_dt)


def prepare_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    # Parquet + pyarrow can fail on object columns with mixed runtime types.
    # Normalize date-like columns first, then keep the rest as nullable strings.
    object_cols = prepared.select_dtypes(include=["object"]).columns
    for col in object_cols:
        col_name = str(col).lower()
        if "дата" in col_name or "date" in col_name:
            parsed = _to_datetime_series(prepared[col])
            non_null = prepared[col].notna().sum()
            if non_null > 0 and parsed.notna().sum() / non_null >= 0.6:
                prepared[col] = parsed
                continue

        prepared[col] = prepared[col].astype("string")

    return prepared


def convert_file(input_path: Path, data_root: Path, compression: str) -> tuple[Path, Path]:
    source_group = detect_source_group(input_path, data_root)
    output_dir = data_root / f"{source_group}_parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.parquet"

    df = pd.read_excel(input_path, dtype=object)
    df_ready = prepare_dataframe_for_parquet(df)
    df_ready.to_parquet(output_path, index=False, engine="pyarrow", compression=compression)
    return input_path, output_path


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()

    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        print("[ERROR] Missing dependency: pyarrow")
        print("Install it with: pip install pyarrow openpyxl")
        return 1

    try:
        resolved_inputs = [resolve_input_path(file_arg, data_root) for file_arg in args.files]
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    max_auto_workers = min(4, len(resolved_inputs), os.cpu_count() or 1)
    workers = args.workers if args.workers and args.workers > 0 else max_auto_workers
    workers = max(1, workers)

    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(convert_file, input_path, data_root, args.compression): input_path
            for input_path in resolved_inputs
        }
        for future in as_completed(futures):
            input_path = futures[future]
            try:
                src, dst = future.result()
                print(f"[OK] {src} -> {dst}")
            except Exception as exc:
                errors += 1
                print(f"[ERROR] {input_path}: {exc}")

    if errors:
        print(f"\nCompleted with {errors} error(s).")
        return 1

    print("\nAll files converted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
