from __future__ import annotations

import re
import warnings
from io import BytesIO
from typing import Callable, Optional
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", message="Workbook contains no default style")

RETRO_MORTGAGE_TABLE_URL = "https://www.cbr.ru/Queries/StatTable/Excel/4-6?lang=ru-RU"
CURRENT_MORTGAGE_BULLETIN_PAGE_URL = "https://www.cbr.ru/statistics/bank_sector/mortgage/mortgage_lending_market/"
RETRO_STAT_COLLECTION_PDF_URL = "https://www.cbr.ru/Collection/Collection/File/15723/Stat_digest_mortgage_05.pdf"

EXPECTED_MONTHLY_COLUMNS = [
    "date",
    "region",
    "mortgage_total_count_monthly",
    "mortgage_total_volume_mln_rub_monthly",
    "mortgage_total_rate_pct_monthly",
    "mortgage_ddu_count_monthly",
    "mortgage_ddu_volume_mln_rub_monthly",
    "mortgage_ddu_rate_pct_monthly",
    "source",
]

QUARTERLY_AGG_SPECS = [
    ("mortgage_total_count_monthly", "mortgage_total_count_q", "sum"),
    ("mortgage_total_volume_mln_rub_monthly", "mortgage_total_volume_mln_rub_q", "sum"),
    ("mortgage_total_rate_pct_monthly", "mortgage_total_rate_pct_q", "mean"),
    ("mortgage_ddu_count_monthly", "mortgage_ddu_count_q", "sum"),
    ("mortgage_ddu_volume_mln_rub_monthly", "mortgage_ddu_volume_mln_rub_q", "sum"),
    ("mortgage_ddu_rate_pct_monthly", "mortgage_ddu_rate_pct_q", "mean"),
]

SOURCE_PRIORITY = {
    "modern_monthly": 0,
    "retro_xlsx": 1,
    "retro_pdf_fallback": 2,
}

RU_MONTHS = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}

Logger = Callable[[str], None]


def _default_logger(message: str) -> None:
    print(message)


def empty_mortgage_monthly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_MONTHLY_COLUMNS)


def normalize_text(value: object) -> str:
    s = "" if pd.isna(value) else str(value).strip().lower().replace("ё", "е")
    s = s.replace("\xa0", " ").replace(" ", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_region_name(value: object) -> str:
    s = normalize_text(value)
    s = s.replace("город ", "г. ")
    s = s.replace("г.москва", "г. москва")
    if s == "москва":
        s = "г. москва"
    return s


def is_target_region(value: object, region_name: str = "г. Москва") -> bool:
    s = normalize_region_name(value)
    target = normalize_region_name(region_name)
    return s == target or ("москва" in s and "область" not in s and "санкт" not in s)


def parse_ru_month_year(text: object) -> pd.Timestamp:
    if pd.isna(text):
        return pd.NaT
    s = normalize_text(text)
    m = re.match(r"(.+?)\s+(\d{4})$", s)
    if not m:
        return pd.NaT
    month = RU_MONTHS.get(m.group(1).strip())
    if month is None:
        return pd.NaT
    return pd.Timestamp(year=int(m.group(2)), month=month, day=1)


def to_numeric_safe(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" ", " ", regex=False)
        .str.strip()
    )
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    out = pd.to_numeric(s, errors="coerce")
    return out.where(~(out > 100), out / 100.0)


def clip_date_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: Optional[str],
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date)
    return out[(out[date_col] >= start_ts) & (out[date_col] <= end_ts)].copy()


def _pick_modern_sheet_name(sheet_names: list[str], needs_rate: bool) -> str:
    norms = {name: normalize_text(name) for name in sheet_names}
    if needs_rate:
        for name, norm in norms.items():
            if "ставка" in norm and "руб" in norm:
                return name
    else:
        for name, norm in norms.items():
            if "руб" in norm and "ставк" not in norm and "срок" not in norm:
                return name
    return sheet_names[0]


def _find_header_row_with_months(raw: pd.DataFrame) -> int:
    for i in range(min(12, len(raw))):
        parsed = pd.Series(raw.iloc[i, 1:]).apply(parse_ru_month_year)
        if parsed.notna().sum() >= 6:
            return i
    raise RuntimeError("Не удалось найти строку с month-year заголовками")


def fetch_modern_mortgage_tables(
    cbr_mortgage_urls: dict[str, dict[str, str]],
    session: Optional[requests.Session] = None,
    logger: Optional[Logger] = None,
) -> dict[str, bytes]:
    session = session or requests.Session()
    logger = logger or _default_logger
    payloads: dict[str, bytes] = {}
    for key, meta in cbr_mortgage_urls.items():
        url = meta["url"]
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        payloads[key] = resp.content
        logger(f"[modern] {key}: downloaded {url}")
    return payloads


def parse_mortgage_table_modern(
    content: bytes,
    value_name: str,
    region_name: str,
    needs_rate: bool,
    start_date: str,
    end_date: Optional[str],
    logger: Optional[Logger] = None,
    numeric_parser: Callable[[pd.Series], pd.Series] = to_numeric_safe,
) -> pd.DataFrame:
    logger = logger or _default_logger
    excel = pd.ExcelFile(BytesIO(content))
    sheet = _pick_modern_sheet_name(excel.sheet_names, needs_rate=needs_rate)
    raw = excel.parse(sheet_name=sheet, header=None)
    header_row = _find_header_row_with_months(raw)
    months = pd.Series(raw.iloc[header_row, 1:]).apply(parse_ru_month_year)

    data = raw.iloc[header_row + 1 :].copy()
    data.columns = ["region"] + list(months)
    data = data.dropna(subset=["region"]).copy()

    row = data.loc[data["region"].apply(lambda x: is_target_region(x, region_name))]
    if row.empty:
        candidates = [x for x in data["region"].tolist() if "моск" in normalize_text(x)]
        raise RuntimeError(
            f"Регион {region_name!r} не найден в modern table. Кандидаты: {candidates[:10]}"
        )

    out = row.iloc[[0]].drop(columns=["region"]).T.reset_index()
    out.columns = ["date", value_name]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_name] = numeric_parser(out[value_name])
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out = clip_date_range(out, start_date=start_date, end_date=end_date)
    logger(
        f"[modern] {value_name}: sheet={sheet!r}, first={out['date'].min().date()}, "
        f"last={out['date'].max().date()}, rows={len(out)}"
    )
    return out


def build_mortgage_monthly_current(
    cbr_mortgage_urls: dict[str, dict[str, str]],
    region_name: str,
    start_date: str,
    end_date: Optional[str],
    session: Optional[requests.Session] = None,
    logger: Optional[Logger] = None,
    numeric_parser: Callable[[pd.Series], pd.Series] = to_numeric_safe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    logger = logger or _default_logger
    payloads = fetch_modern_mortgage_tables(cbr_mortgage_urls, session=session, logger=logger)
    parsed_frames: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, object] = {"series": {}}

    for key, meta in cbr_mortgage_urls.items():
        value_name = meta["value_name"]
        needs_rate = "rate" in key
        frame = parse_mortgage_table_modern(
            payloads[key],
            value_name=value_name,
            region_name=region_name,
            needs_rate=needs_rate,
            start_date=start_date,
            end_date=end_date,
            logger=logger,
            numeric_parser=numeric_parser,
        )
        parsed_frames[key] = frame
        diagnostics["series"][key] = {
            "first_date": frame["date"].min(),
            "last_date": frame["date"].max(),
            "rows": len(frame),
        }

    monthly = None
    for frame in parsed_frames.values():
        if monthly is None:
            monthly = frame.copy()
        else:
            monthly = monthly.merge(frame, on="date", how="outer")

    if monthly is None:
        return empty_mortgage_monthly_frame(), diagnostics

    monthly["region"] = region_name
    monthly["source"] = "modern_monthly"
    monthly = monthly.sort_values("date").reset_index(drop=True)
    diagnostics["first_date"] = monthly["date"].min()
    diagnostics["last_date"] = monthly["date"].max()
    diagnostics["rows"] = len(monthly)
    return monthly, diagnostics


def discover_retro_mortgage_sources(
    session: Optional[requests.Session] = None,
    logger: Optional[Logger] = None,
) -> dict[str, object]:
    session = session or requests.Session()
    logger = logger or _default_logger

    guessed_stat_urls = []
    for idx in range(1, 21):
        for template in [
            "https://www.cbr.ru/vfs/statistics/BankSector/Mortgage/stat_morgage_tables_{idx:02d}.xlsx",
            "https://www.cbr.ru/StaticHtml/File/{idx}/stat_morgage_tables_{idx:02d}.xlsx",
            "https://www.cbr.ru/Collection/Collection/File/{idx}/stat_morgage_tables_{idx:02d}.xlsx",
        ]:
            url = template.format(idx=idx)
            try:
                resp = session.head(url, allow_redirects=True, timeout=10)
                if resp.status_code == 200:
                    guessed_stat_urls.append(resp.url)
            except Exception:
                continue

    guessed_stat_urls = sorted(set(guessed_stat_urls))
    if guessed_stat_urls:
        logger(f"[retro] discovered stat_morgage_tables URLs: {guessed_stat_urls}")
    else:
        logger("[retro] stat_morgage_tables_XX.xlsx direct probe returned no 200 URLs")

    current_xlsx_links: list[str] = []
    current_pdf_links: list[str] = []
    try:
        resp = session.get(CURRENT_MORTGAGE_BULLETIN_PAGE_URL, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(CURRENT_MORTGAGE_BULLETIN_PAGE_URL, a["href"])
            href_low = href.lower()
            if href_low.endswith(".xlsx"):
                current_xlsx_links.append(href)
            elif href_low.endswith(".pdf"):
                current_pdf_links.append(href)
    except Exception as exc:
        logger(f"[retro] failed to scrape mortgage bulletin page: {exc!r}")

    current_xlsx_links = sorted(set(current_xlsx_links))
    current_pdf_links = sorted(set(current_pdf_links))
    if current_xlsx_links:
        logger(f"[retro] latest bulletin XLSX candidate: {current_xlsx_links[0]}")

    return {
        "stat_morgage_tables": guessed_stat_urls,
        "retro_excel_url": RETRO_MORTGAGE_TABLE_URL,
        "retro_pdf_candidates": [RETRO_STAT_COLLECTION_PDF_URL],
        "current_bulletin_xlsx": current_xlsx_links,
        "current_bulletin_pdf": current_pdf_links,
    }


def download_retro_mortgage_xlsx(
    url: str,
    session: Optional[requests.Session] = None,
    logger: Optional[Logger] = None,
) -> bytes:
    session = session or requests.Session()
    logger = logger or _default_logger
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    logger(f"[retro] downloaded Excel bundle: {url}")
    return resp.content


def _build_retro_header_map(raw: pd.DataFrame) -> dict[int, str]:
    header = (
        raw.iloc[:9, :]
        .copy()
        .apply(lambda col: col.map(normalize_text))
        .replace({"": pd.NA, "nan": pd.NA})
        .ffill(axis=1)
    )
    out: dict[int, str] = {}
    for j in range(header.shape[1]):
        parts = []
        for i in range(header.shape[0]):
            val = header.iat[i, j]
            if pd.isna(val):
                continue
            if not parts or parts[-1] != val:
                parts.append(val)
        out[j] = " | ".join(parts)
    return out


def _find_retro_region_row(raw: pd.DataFrame, region_name: str) -> int | None:
    for i in range(min(len(raw), 150)):
        if is_target_region(raw.iat[i, 0], region_name):
            return i
    return None


def _retro_metric_map(header_map: dict[int, str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for j, blob in header_map.items():
        if "ипотечные жилищные кредиты" in blob and "количество" in blob and "под залог прав требования" not in blob:
            mapping["mortgage_total_count_cum"] = j
        elif "ипотечные жилищные кредиты" in blob and "объем" in blob and "под залог прав требования" not in blob:
            mapping["mortgage_total_volume_cum"] = j
        elif (
            "ипотечные жилищные кредиты" in blob
            and "средневзвешенная ставка" in blob
            and "в течение месяца" in blob
            and "под залог прав требования" not in blob
        ):
            mapping["mortgage_total_rate_pct_monthly"] = j
        elif "под залог прав требования" in blob and "количество" in blob:
            mapping["mortgage_ddu_count_cum"] = j
        elif "под залог прав требования" in blob and "объем" in blob:
            mapping["mortgage_ddu_volume_cum"] = j
        elif (
            "под залог прав требования" in blob
            and "средневзвешенная ставка" in blob
            and "в течение месяца" in blob
        ):
            mapping["mortgage_ddu_rate_pct_monthly"] = j
    return mapping


def _reconstruct_monthly_from_cumulative(df: pd.DataFrame, cum_col: str, out_col: str) -> pd.Series:
    out = df.groupby(df["date"].dt.year)[cum_col].diff()
    first_mask = df.groupby(df["date"].dt.year).cumcount().eq(0)
    out.loc[first_mask] = df.loc[first_mask, cum_col]
    return out


def parse_retro_xlsx_mortgage(
    source: bytes | str,
    region_name: str,
    start_date: str,
    end_date: Optional[str],
    logger: Optional[Logger] = None,
    numeric_parser: Callable[[pd.Series], pd.Series] = to_numeric_safe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    logger = logger or _default_logger
    content = source if isinstance(source, bytes) else requests.get(source, timeout=120).content
    excel = pd.ExcelFile(BytesIO(content))
    records = []
    sheets_missing_ddu: list[str] = []

    for sheet in excel.sheet_names:
        match = re.search(r"(\d{2}\.\d{2}\.\d{4})", sheet)
        if not match:
            continue

        as_of = pd.to_datetime(match.group(1), dayfirst=True)
        month_date = as_of - pd.offsets.MonthBegin(1)
        raw = excel.parse(sheet_name=sheet, header=None)
        target_row = _find_retro_region_row(raw, region_name=region_name)
        if target_row is None:
            continue

        header_map = _build_retro_header_map(raw)
        metric_map = _retro_metric_map(header_map)
        record = {"date": month_date}
        for field in [
            "mortgage_total_count_cum",
            "mortgage_total_volume_cum",
            "mortgage_total_rate_pct_monthly",
            "mortgage_ddu_count_cum",
            "mortgage_ddu_volume_cum",
            "mortgage_ddu_rate_pct_monthly",
        ]:
            record[field] = raw.iat[target_row, metric_map[field]] if field in metric_map else np.nan
        if "mortgage_ddu_count_cum" not in metric_map:
            sheets_missing_ddu.append(sheet)
        records.append(record)

    if not records:
        logger("[retro] no usable sheets were parsed from retro Excel")
        return empty_mortgage_monthly_frame(), {"sheets_missing_ddu": []}

    retro = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    for col in retro.columns:
        if col != "date":
            retro[col] = numeric_parser(retro[col])

    retro["mortgage_total_count_monthly"] = _reconstruct_monthly_from_cumulative(
        retro, "mortgage_total_count_cum", "mortgage_total_count_monthly"
    )
    retro["mortgage_total_volume_mln_rub_monthly"] = _reconstruct_monthly_from_cumulative(
        retro, "mortgage_total_volume_cum", "mortgage_total_volume_mln_rub_monthly"
    )
    retro["mortgage_ddu_count_monthly"] = _reconstruct_monthly_from_cumulative(
        retro, "mortgage_ddu_count_cum", "mortgage_ddu_count_monthly"
    )
    retro["mortgage_ddu_volume_mln_rub_monthly"] = _reconstruct_monthly_from_cumulative(
        retro, "mortgage_ddu_volume_cum", "mortgage_ddu_volume_mln_rub_monthly"
    )

    retro = retro[
        [
            "date",
            "mortgage_total_count_monthly",
            "mortgage_total_volume_mln_rub_monthly",
            "mortgage_total_rate_pct_monthly",
            "mortgage_ddu_count_monthly",
            "mortgage_ddu_volume_mln_rub_monthly",
            "mortgage_ddu_rate_pct_monthly",
        ]
    ].copy()
    retro["region"] = region_name
    retro["source"] = "retro_xlsx"
    retro = clip_date_range(retro, start_date=start_date, end_date=end_date).reset_index(drop=True)

    diagnostics = {
        "rows": len(retro),
        "first_date": retro["date"].min() if not retro.empty else None,
        "last_date": retro["date"].max() if not retro.empty else None,
        "sheets_missing_ddu": sheets_missing_ddu,
        "metric_first_dates": {
            col: retro.loc[retro[col].notna(), "date"].min() if col in retro.columns else None
            for col in retro.columns
            if col.endswith("_monthly")
        },
    }
    logger(
        f"[retro] parsed Excel bundle: first={diagnostics['first_date']}, "
        f"last={diagnostics['last_date']}, rows={diagnostics['rows']}"
    )
    return retro, diagnostics


def parse_retro_pdf_mortgage_fallback(
    pdf_urls: list[str],
    region_name: str,
    start_date: str,
    end_date: Optional[str],
    logger: Optional[Logger] = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    logger = logger or _default_logger
    try:
        import importlib.util

        has_pdf_tools = any(
            importlib.util.find_spec(mod) is not None for mod in ("pdfplumber", "camelot", "tabula")
        )
    except Exception:
        has_pdf_tools = False

    if not pdf_urls:
        logger("[retro-pdf] no PDF URLs were provided for fallback")
        return empty_mortgage_monthly_frame(), {"used": False, "reason": "no_urls"}

    if not has_pdf_tools:
        logger(
            "[retro-pdf] PDF fallback requested, but pdfplumber/camelot/tabula are unavailable. "
            f"Skipped URLs: {pdf_urls}"
        )
        return empty_mortgage_monthly_frame(), {"used": False, "reason": "no_pdf_tools", "urls": pdf_urls}

    logger(
        "[retro-pdf] PDF fallback was requested, but conservative parser is not enabled in this environment. "
        f"Skipped URLs: {pdf_urls}"
    )
    return empty_mortgage_monthly_frame(), {"used": False, "reason": "not_implemented", "urls": pdf_urls}


def merge_mortgage_monthly_layers(
    modern_monthly: pd.DataFrame,
    retro_xlsx: Optional[pd.DataFrame] = None,
    retro_pdf_fallback: Optional[pd.DataFrame] = None,
    region_name: str = "г. Москва",
) -> pd.DataFrame:
    frames = []
    for df in [modern_monthly, retro_xlsx, retro_pdf_fallback]:
        if df is not None and not df.empty:
            frames.append(df.copy())
    if not frames:
        return empty_mortgage_monthly_frame()

    full = pd.concat(frames, ignore_index=True, sort=False)
    for col in EXPECTED_MONTHLY_COLUMNS:
        if col not in full.columns:
            full[col] = np.nan
    full["date"] = pd.to_datetime(full["date"])
    full["region"] = full["region"].fillna(region_name)
    full["priority"] = full["source"].map(SOURCE_PRIORITY).fillna(99)

    metric_cols = [c for c in EXPECTED_MONTHLY_COLUMNS if c.endswith("_monthly")]
    rows = []
    for dt, grp in full.sort_values(["date", "priority"]).groupby("date", sort=True):
        row = {"date": dt, "region": region_name}
        for col in metric_cols:
            vals = grp[col].dropna()
            row[col] = vals.iloc[0] if len(vals) else np.nan
        non_null_sources = grp.loc[grp[metric_cols].notna().any(axis=1), "source"]
        row["source"] = non_null_sources.iloc[0] if len(non_null_sources) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in EXPECTED_MONTHLY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[EXPECTED_MONTHLY_COLUMNS].sort_values("date").reset_index(drop=True)


def build_mortgage_monthly_backfilled(
    cbr_mortgage_urls: dict[str, dict[str, str]],
    region_name: str,
    start_date: str,
    end_date: Optional[str],
    session: Optional[requests.Session] = None,
    logger: Optional[Logger] = None,
    numeric_parser: Callable[[pd.Series], pd.Series] = to_numeric_safe,
) -> tuple[pd.DataFrame, dict[str, object]]:
    logger = logger or _default_logger
    current_monthly, current_diag = build_mortgage_monthly_current(
        cbr_mortgage_urls=cbr_mortgage_urls,
        region_name=region_name,
        start_date=start_date,
        end_date=end_date,
        session=session,
        logger=logger,
        numeric_parser=numeric_parser,
    )

    sources = discover_retro_mortgage_sources(session=session, logger=logger)
    retro_xlsx = empty_mortgage_monthly_frame()
    retro_xlsx_diag: dict[str, object] = {}
    if sources.get("retro_excel_url"):
        retro_bytes = download_retro_mortgage_xlsx(
            sources["retro_excel_url"],
            session=session,
            logger=logger,
        )
        retro_xlsx, retro_xlsx_diag = parse_retro_xlsx_mortgage(
            retro_bytes,
            region_name=region_name,
            start_date=start_date,
            end_date=end_date,
            logger=logger,
            numeric_parser=numeric_parser,
        )

    gap_metrics = []
    for metric in [
        "mortgage_total_count_monthly",
        "mortgage_total_volume_mln_rub_monthly",
        "mortgage_total_rate_pct_monthly",
        "mortgage_ddu_count_monthly",
        "mortgage_ddu_volume_mln_rub_monthly",
        "mortgage_ddu_rate_pct_monthly",
    ]:
        if retro_xlsx.empty:
            gap_metrics.append(metric)
            continue
        first_valid = retro_xlsx.loc[retro_xlsx[metric].notna(), "date"].min()
        if pd.isna(first_valid) or first_valid > pd.Timestamp(start_date):
            gap_metrics.append(metric)

    retro_pdf = empty_mortgage_monthly_frame()
    retro_pdf_diag: dict[str, object] = {"used": False}
    if gap_metrics:
        logger(f"[retro-pdf] remaining pre-{start_date} gaps after XLSX backfill: {gap_metrics}")
        retro_pdf, retro_pdf_diag = parse_retro_pdf_mortgage_fallback(
            pdf_urls=sources.get("retro_pdf_candidates", []),
            region_name=region_name,
            start_date=start_date,
            end_date=end_date,
            logger=logger,
        )

    combined = merge_mortgage_monthly_layers(
        modern_monthly=current_monthly,
        retro_xlsx=retro_xlsx,
        retro_pdf_fallback=retro_pdf,
        region_name=region_name,
    )
    diagnostics = {
        "sources": sources,
        "current": current_diag,
        "retro_xlsx": retro_xlsx_diag,
        "retro_pdf": retro_pdf_diag,
        "gap_metrics_after_retro_xlsx": gap_metrics,
        "combined_first_date": combined["date"].min() if not combined.empty else None,
        "combined_last_date": combined["date"].max() if not combined.empty else None,
        "combined_rows": len(combined),
    }
    return combined, diagnostics


def build_mortgage_quarterly_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(columns=["quarter"])

    tmp = monthly_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["quarter"] = tmp["date"].dt.to_period("Q").astype(str)

    quarter = pd.DataFrame({"quarter": sorted(tmp["quarter"].unique())})
    for monthly_col, quarterly_col, how in QUARTERLY_AGG_SPECS:
        if monthly_col not in tmp.columns:
            quarter[quarterly_col] = np.nan
            continue
        if how == "sum":
            agg = tmp.groupby("quarter")[monthly_col].sum(min_count=1)
        elif how == "mean":
            agg = tmp.groupby("quarter")[monthly_col].mean()
        else:
            raise ValueError(f"Unsupported aggregation: {how}")
        quarter = quarter.merge(agg.rename(quarterly_col), on="quarter", how="left")

        prev = quarter[quarterly_col].shift(1)
        quarter[f"{quarterly_col}_qoq_change"] = quarter[quarterly_col] - prev
        quarter[f"{quarterly_col}_log_return_q"] = np.where(
            (quarter[quarterly_col] > 0) & (prev > 0),
            np.log(quarter[quarterly_col]) - np.log(prev),
            np.nan,
        )

    if {"mortgage_ddu_count_q", "mortgage_total_count_q"}.issubset(quarter.columns):
        quarter["mortgage_ddu_count_share_q"] = (
            quarter["mortgage_ddu_count_q"] / quarter["mortgage_total_count_q"].replace(0, np.nan)
        )
    if {"mortgage_ddu_volume_mln_rub_q", "mortgage_total_volume_mln_rub_q"}.issubset(quarter.columns):
        quarter["mortgage_ddu_volume_share_q"] = (
            quarter["mortgage_ddu_volume_mln_rub_q"]
            / quarter["mortgage_total_volume_mln_rub_q"].replace(0, np.nan)
        )
    if {"mortgage_total_volume_mln_rub_q", "mortgage_total_count_q"}.issubset(quarter.columns):
        quarter["mortgage_avg_ticket_mln_rub_q"] = (
            quarter["mortgage_total_volume_mln_rub_q"]
            / quarter["mortgage_total_count_q"].replace(0, np.nan)
        )
    if {"mortgage_ddu_volume_mln_rub_q", "mortgage_ddu_count_q"}.issubset(quarter.columns):
        quarter["mortgage_ddu_avg_ticket_mln_rub_q"] = (
            quarter["mortgage_ddu_volume_mln_rub_q"]
            / quarter["mortgage_ddu_count_q"].replace(0, np.nan)
        )
    return quarter.sort_values("quarter").reset_index(drop=True)


def compare_coverage(
    before_coverage: pd.DataFrame,
    after_coverage: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    before = before_coverage.loc[before_coverage["feature"].isin(features)].copy()
    after = after_coverage.loc[after_coverage["feature"].isin(features)].copy()
    merged = before.merge(
        after,
        on="feature",
        how="outer",
        suffixes=("_before", "_after"),
    )
    merged["history_extended_backward"] = (
        merged["first_quarter_before"].fillna("NA") != merged["first_quarter_after"].fillna("NA")
    )
    return merged.sort_values("feature").reset_index(drop=True)
