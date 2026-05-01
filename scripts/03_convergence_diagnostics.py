#!/usr/bin/env python3
"""Convergence diagnostics for MWG posterior samples.

This script is dependency-light (numpy/pandas only) and produces:
1) parameter-level diagnostics CSV
2) lag autocorrelation CSV (for easy plotting later)
3) split-chain R-hat CSV (single chain split into two halves)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute convergence diagnostics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/posterior_samples.csv"),
        help="Posterior samples CSV from 02_mwg_sampler.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics"),
        help="Where diagnostics files are saved.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=40,
        help="Maximum autocorrelation lag.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots.",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default="",
        help="Optional suffix appended to output filenames, e.g. '_new'.",
    )
    return parser.parse_args()


def autocorr(x: np.ndarray, lag: int) -> float:
    if lag == 0:
        return 1.0
    x0 = x[:-lag]
    x1 = x[lag:]
    x0c = x0 - x0.mean()
    x1c = x1 - x1.mean()
    denom = np.sqrt(np.sum(x0c * x0c) * np.sum(x1c * x1c))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x0c * x1c) / denom)


def ess_initial_positive_sequence(x: np.ndarray, max_lag: int) -> float:
    n = len(x)
    if n < 10:
        return float(n)
    acf = [autocorr(x, lag) for lag in range(1, max_lag + 1)]
    tau = 1.0
    for k in range(0, len(acf) - 1, 2):
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum <= 0:
            break
        tau += 2.0 * pair_sum
    return float(n / tau)


def split_rhat(x: np.ndarray) -> float:
    n = len(x)
    if n < 20:
        return np.nan
    half = n // 2
    x1 = x[:half]
    x2 = x[half : 2 * half]
    m = 2
    n_chain = half
    chain_means = np.array([x1.mean(), x2.mean()])
    grand_mean = chain_means.mean()
    b = n_chain * np.sum((chain_means - grand_mean) ** 2) / (m - 1)
    w = (x1.var(ddof=1) + x2.var(ddof=1)) / m
    var_hat = ((n_chain - 1) / n_chain) * w + (1 / n_chain) * b
    if w <= 0:
        return np.nan
    return float(np.sqrt(var_hat / w))


def multi_chain_rhat(chains: list[np.ndarray]) -> float:
    if len(chains) < 2:
        return np.nan
    min_len = min(len(c) for c in chains)
    if min_len < 20:
        return np.nan
    trimmed = np.array([c[:min_len] for c in chains], dtype=float)
    m, n = trimmed.shape
    chain_means = trimmed.mean(axis=1)
    grand_mean = chain_means.mean()
    b = n * np.sum((chain_means - grand_mean) ** 2) / (m - 1)
    w = trimmed.var(axis=1, ddof=1).mean()
    var_hat = ((n - 1) / n) * w + (1 / n) * b
    if w <= 0:
        return np.nan
    return float(np.sqrt(var_hat / w))


def geweke_z(x: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5) -> float:
    n = len(x)
    n_a = int(first_frac * n)
    n_b = int(last_frac * n)
    if n_a < 5 or n_b < 5:
        return np.nan
    a = x[:n_a]
    b = x[-n_b:]
    num = a.mean() - b.mean()
    den = np.sqrt(a.var(ddof=1) / n_a + b.var(ddof=1) / n_b)
    if den == 0:
        return np.nan
    return float(num / den)


def maybe_generate_plots(
    series_by_param: dict[str, np.ndarray],
    acf_df: pd.DataFrame,
    output_dir: Path,
    max_lag: int,
    file_suffix: str,
    chain_series_by_param: dict[str, dict[int, np.ndarray]] | None = None,
) -> None:
    # Keep matplotlib cache writable in sandboxed environments.
    mpl_cache = output_dir / ".mplcache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot generation skipped (matplotlib unavailable): {exc}")
        return

    # User-selected palette for multi-chain traces.
    chain_palette = ["#F7DD77", "#481E66", "#22948F", "#344E9B"]
    acf_color = "#344E9B"

    for param, x in series_by_param.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))

        # Trace plot
        chain_map = None if chain_series_by_param is None else chain_series_by_param.get(param)
        if chain_map:
            color_iter = cycle(chain_palette)
            for chain_id in sorted(chain_map):
                xc = chain_map[chain_id]
                axes[0].plot(
                    np.arange(1, len(xc) + 1),
                    xc,
                    lw=0.8,
                    alpha=0.9,
                    color=next(color_iter),
                    label=f"chain {chain_id}",
                )
            axes[0].legend(fontsize=7, ncol=2)
        else:
            axes[0].plot(np.arange(1, len(x) + 1), x, lw=0.8)
        axes[0].set_title(f"Trace: {param}")
        axes[0].set_xlabel("Saved draw")
        axes[0].set_ylabel(param)

        # Histogram (density)
        axes[1].hist(x, bins=40, density=True, alpha=0.8)
        # Add a smooth KDE-like curve (Gaussian kernel, Silverman's bandwidth).
        n = len(x)
        x_std = float(np.std(x, ddof=1))
        if n > 1 and x_std > 0:
            bw = 1.06 * x_std * (n ** (-1 / 5))
            bw = max(bw, 1e-12)
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
            z = (grid[:, None] - x[None, :]) / bw
            dens = np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))
            axes[1].plot(grid, dens, color="#481E66", lw=2.0, alpha=0.95)
        axes[1].set_title(f"Posterior density: {param}")
        axes[1].set_xlabel(param)
        axes[1].set_ylabel("Density")

        # ACF
        sub = acf_df[acf_df["parameter"] == param].copy()
        lags = sub["lag"].to_numpy(dtype=int)
        vals = sub["autocorr"].to_numpy(dtype=float)
        conf = 1.96 / np.sqrt(max(len(x), 1))
        axes[2].fill_between(
            [0, max_lag],
            [-conf, -conf],
            [conf, conf],
            color="#D9E4FF",
            alpha=0.8,
            zorder=0,
            label="95% band",
        )
        axes[2].axhline(0.0, color="black", lw=0.8)
        axes[2].vlines(lags, 0.0, vals, lw=1.2, color=acf_color)
        axes[2].plot(lags, vals, color=acf_color, lw=0.8, marker="o", ms=2.5, alpha=0.9)
        axes[2].set_xlim(0, max_lag)
        axes[2].set_ylim(min(-0.15, vals.min() - 0.05), max(1.0, vals.max() + 0.05))
        axes[2].set_title(f"ACF: {param}")
        axes[2].set_xlabel("Lag")
        axes[2].set_ylabel("Autocorrelation")

        fig.tight_layout()
        out = output_dir / f"{param}_diagnostics{file_suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")

    # Also save a dedicated ACF overview figure for cleaner presentation.
    n_params = len(series_by_param)
    fig, axes = plt.subplots(1, n_params, figsize=(5.2 * n_params, 3.8))
    if n_params == 1:
        axes = [axes]
    for ax, (param, x) in zip(axes, series_by_param.items()):
        sub = acf_df[acf_df["parameter"] == param].copy()
        lags = sub["lag"].to_numpy(dtype=int)
        vals = sub["autocorr"].to_numpy(dtype=float)
        conf = 1.96 / np.sqrt(max(len(x), 1))
        ax.fill_between([0, max_lag], [-conf, -conf], [conf, conf], color="#D9E4FF", alpha=0.8, zorder=0)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.vlines(lags, 0.0, vals, lw=1.2, color=acf_color)
        ax.plot(lags, vals, color=acf_color, lw=0.8, marker="o", ms=2.5, alpha=0.9)
        ax.set_xlim(0, max_lag)
        ax.set_ylim(min(-0.15, vals.min() - 0.05), max(1.0, vals.max() + 0.05))
        ax.set_title(f"ACF: {param}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
    fig.tight_layout()
    acf_overview = output_dir / f"autocorr_overview{file_suffix}.png"
    fig.savefig(acf_overview, dpi=160)
    plt.close(fig)
    print(f"Saved: {acf_overview}")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Posterior sample file not found: {args.input}")

    df = pd.read_csv(args.input)
    params = [c for c in df.columns if c != "iter"]
    has_chain = "chain" in df.columns
    if has_chain:
        params = [c for c in params if c != "chain"]
    if not params:
        raise SystemExit("No parameter columns found in posterior sample file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diag_rows = []
    acf_rows = []
    series_by_param: dict[str, np.ndarray] = {}
    chain_series_by_param: dict[str, dict[int, np.ndarray]] = {}
    for p in params:
        x = pd.to_numeric(df[p], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) < 20:
            continue
        series_by_param[p] = x
        ess = ess_initial_positive_sequence(x, args.max_lag)
        if has_chain:
            chain_groups = []
            chain_map: dict[int, np.ndarray] = {}
            for chain_id, g in df.groupby("chain", sort=True):
                xc = pd.to_numeric(g[p], errors="coerce").dropna().to_numpy(dtype=float)
                if len(xc) >= 20:
                    chain_groups.append(xc)
                    chain_map[int(chain_id)] = xc
            chain_series_by_param[p] = chain_map
            rhat = multi_chain_rhat(chain_groups)
        else:
            rhat = split_rhat(x)
        gz = geweke_z(x)
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        mcse = float(sd / np.sqrt(max(ess, 1.0)))

        diag_rows.append(
            {
                "parameter": p,
                "draws": len(x),
                "mean": mean,
                "sd": sd,
                "mcse": mcse,
                "ess": ess,
                "ess_per_draw": ess / len(x),
                "split_rhat": rhat,
                "geweke_z": gz,
            }
        )

        for lag in range(args.max_lag + 1):
            acf_rows.append({"parameter": p, "lag": lag, "autocorr": autocorr(x, lag)})

    diag_df = pd.DataFrame(diag_rows).sort_values("parameter")
    acf_df = pd.DataFrame(acf_rows).sort_values(["parameter", "lag"])

    diag_path = args.output_dir / f"diagnostics_summary{args.file_suffix}.csv"
    acf_path = args.output_dir / f"autocorr_by_lag{args.file_suffix}.csv"
    diag_df.to_csv(diag_path, index=False)
    acf_df.to_csv(acf_path, index=False)

    print(f"Saved: {diag_path}")
    print(f"Saved: {acf_path}")
    if not args.no_plots:
        maybe_generate_plots(
            series_by_param=series_by_param,
            acf_df=acf_df,
            output_dir=args.output_dir,
            max_lag=args.max_lag,
            file_suffix=args.file_suffix,
            chain_series_by_param=chain_series_by_param if has_chain else None,
        )
    print(diag_df.to_string(index=False))


if __name__ == "__main__":
    main()
