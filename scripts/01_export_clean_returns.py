#!/usr/bin/env python3
"""Export cleaned return data from a Numbers file to CSV.

Usage:
    python scripts/01_export_clean_returns.py \
        --input cleaned_returns.numbers \
        --output data/clean_returns.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean return table to CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cleaned_returns.numbers"),
        help="Path to the Numbers workbook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clean_returns.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--sheet-index",
        type=int,
        default=0,
        help="Sheet index in Numbers workbook (0-based).",
    )
    parser.add_argument(
        "--table-index",
        type=int,
        default=0,
        help="Table index in selected sheet (0-based).",
    )
    return parser.parse_args()


def load_numbers_table(path: Path, sheet_index: int, table_index: int) -> pd.DataFrame:
    try:
        from numbers_parser import Document
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "numbers-parser is required. Install with: pip install numbers-parser"
        ) from exc

    document = Document(str(path))
    try:
        table = document.sheets[sheet_index].tables[table_index]
    except IndexError as exc:
        raise SystemExit(
            f"Invalid sheet/table index: sheet={sheet_index}, table={table_index}"
        ) from exc

    rows = table.rows(values_only=True)
    if len(rows) < 2:
        raise SystemExit("Selected table does not contain data rows.")

    header = rows[0]
    body = rows[1:]
    data = pd.DataFrame(body, columns=header)
    return data


def clean_returns_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Date", "Close", "SimpleReturn", "LogReturn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ("Close", "SimpleReturn", "LogReturn"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Date", "Close", "SimpleReturn", "LogReturn"])
    out = out.sort_values("Date").drop_duplicates(subset="Date", keep="first")

    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    df = load_numbers_table(args.input, args.sheet_index, args.table_index)
    clean_df = clean_returns_dataframe(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, index=False)

    print(f"Exported rows: {len(clean_df)}")
    print(f"Columns: {', '.join(clean_df.columns)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
