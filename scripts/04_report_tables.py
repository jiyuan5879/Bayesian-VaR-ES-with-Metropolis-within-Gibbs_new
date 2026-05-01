#!/usr/bin/env python3
"""Build report-ready tables and markdown from project outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready summary tables.")
    parser.add_argument("--returns", type=Path, default=Path("data/clean_returns.csv"))
    parser.add_argument(
        "--posterior-summary", type=Path, default=Path("outputs/posterior_summary.csv")
    )
    parser.add_argument("--risk", type=Path, default=Path("outputs/var_es_summary.csv"))
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("outputs/diagnostics/diagnostics_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report"))
    return parser.parse_args()


def load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Return file not found: {path}")
    df = pd.read_csv(path)
    required = {"Date", "LogReturn"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in returns file: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["LogReturn"] = pd.to_numeric(df["LogReturn"], errors="coerce")
    df = df.dropna(subset=["Date", "LogReturn"]).sort_values("Date")
    return df


def data_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    r = df["LogReturn"].to_numpy()
    n = len(r)
    annual_factor = 252
    out = pd.DataFrame(
        [
            {
                "n_obs": n,
                "start_date": df["Date"].min().strftime("%Y-%m-%d"),
                "end_date": df["Date"].max().strftime("%Y-%m-%d"),
                "mean_daily_log_return": float(np.mean(r)),
                "sd_daily_log_return": float(np.std(r, ddof=1)),
                "annualized_mean_log_return": float(np.mean(r) * annual_factor),
                "annualized_volatility": float(np.std(r, ddof=1) * np.sqrt(annual_factor)),
                "min_daily_log_return": float(np.min(r)),
                "max_daily_log_return": float(np.max(r)),
            }
        ]
    )
    return out


def markdown_table(df: pd.DataFrame, float_digits: int = 6) -> str:
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: f"{x:.{float_digits}f}")
    headers = [str(c) for c in show.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in show.iterrows():
        vals = [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ret_df = load_returns(args.returns)
    data_tbl = data_summary_table(ret_df)

    if not args.posterior_summary.exists():
        raise SystemExit(f"Posterior summary not found: {args.posterior_summary}")
    post_tbl = pd.read_csv(args.posterior_summary)

    if not args.risk.exists():
        raise SystemExit(f"Risk summary not found: {args.risk}")
    risk_tbl = pd.read_csv(args.risk)

    diag_tbl = None
    if args.diagnostics.exists():
        diag_tbl = pd.read_csv(args.diagnostics)

    data_csv = args.output_dir / "table_data_summary.csv"
    post_csv = args.output_dir / "table_posterior_summary.csv"
    risk_csv = args.output_dir / "table_var_es.csv"
    data_tbl.to_csv(data_csv, index=False)
    post_tbl.to_csv(post_csv, index=False)
    risk_tbl.to_csv(risk_csv, index=False)

    md_lines = [
        "# STATS211 Project Result Tables",
        "",
        "## 1) Data Summary (Log Returns)",
        "",
        markdown_table(data_tbl, float_digits=6),
        "",
        "## 2) Posterior Parameter Summary",
        "",
        markdown_table(post_tbl, float_digits=6),
        "",
        "## 3) VaR and ES (Loss Scale)",
        "",
        markdown_table(risk_tbl, float_digits=6),
        "",
    ]

    if diag_tbl is not None and not diag_tbl.empty:
        md_lines.extend(
            [
                "## 4) Convergence Diagnostics",
                "",
                markdown_table(diag_tbl, float_digits=4),
                "",
                "Diagnostic note: ideally split-Rhat should be close to 1, "
                "and ESS should be reasonably large.",
                "",
            ]
        )

    md_path = args.output_dir / "report_tables.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {data_csv}")
    print(f"Saved: {post_csv}")
    print(f"Saved: {risk_csv}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
