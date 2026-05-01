#!/usr/bin/env python3
"""Metropolis within Gibbs sampler for Student-t returns.

Model:
    y_i | mu, sigma2, lambda_i ~ Normal(mu, sigma2 / lambda_i)
    lambda_i | nu ~ Gamma(nu/2, nu/2)

Priors:
    mu | sigma2 ~ Normal(mu0, sigma2 / kappa0)
    sigma2 ~ InvGamma(a0, b0)
    nu - 2 ~ Exponential(rate_nu), nu > 2

Outputs:
    - posterior samples CSV
    - posterior summary CSV
    - VaR/ES summary CSV (posterior predictive simulation)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MWG sampler on clean log returns.")
    parser.add_argument("--input", type=Path, default=Path("data/clean_returns.csv"))
    parser.add_argument("--return-col", type=str, default="LogReturn")
    parser.add_argument("--n-iter", type=int, default=120000)
    parser.add_argument("--burn-in", type=int, default=20000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--chain-seed-stride", type=int, default=1000)
    parser.add_argument("--proposal-sd", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Tail probability alpha for VaR/ES (e.g. 0.01 for 99% VaR).",
    )
    parser.add_argument(
        "--risk-ci",
        type=float,
        default=0.95,
        help="Credible interval level for VaR/ES uncertainty summary.",
    )
    parser.add_argument("--pred-samples-per-draw", type=int, default=200)
    return parser.parse_args()


def load_log_returns(path: Path, col: str) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if col not in df.columns:
        raise SystemExit(f"Column '{col}' not found. Available: {list(df.columns)}")
    y = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if y.size < 100:
        raise SystemExit("Too few valid return observations for stable sampling.")
    return y


def log_post_nu_given_lambda(nu: float, lambdas: np.ndarray, rate_nu: float) -> float:
    if nu <= 2.0:
        return -np.inf
    n = lambdas.size
    half_nu = 0.5 * nu
    term1 = n * (half_nu * math.log(half_nu) - math.lgamma(half_nu))
    term2 = (half_nu - 1.0) * np.log(lambdas).sum()
    term3 = -half_nu * lambdas.sum()
    log_prior = math.log(rate_nu) - rate_nu * (nu - 2.0)
    return term1 + term2 + term3 + log_prior


def run_mwg(
    y: np.ndarray,
    n_iter: int,
    burn_in: int,
    thin: int,
    seed: int,
    proposal_sd: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if burn_in >= n_iter:
        raise SystemExit("burn-in must be smaller than n-iter.")

    rng = np.random.default_rng(seed)
    n = y.size

    # Weakly informative priors
    mu0 = float(np.mean(y))
    kappa0 = 0.01
    a0 = 0.01
    b0 = float(np.var(y, ddof=1))
    rate_nu = 0.1

    mu = float(np.mean(y))
    sigma2 = float(np.var(y, ddof=1))
    nu = 8.0
    lambdas = np.ones(n, dtype=float)

    saved = []
    accepted = 0
    mh_steps = 0

    for it in range(1, n_iter + 1):
        # 1) Gibbs: sample lambda_i | rest
        sq = (y - mu) ** 2 / sigma2
        shape = 0.5 * (nu + 1.0)
        rate = 0.5 * (nu + sq)
        lambdas = rng.gamma(shape=shape, scale=1.0 / rate)

        # 2) Gibbs: sample mu | rest
        kappa_n = kappa0 + lambdas.sum()
        mean_n = (kappa0 * mu0 + np.sum(lambdas * y)) / kappa_n
        var_n = sigma2 / kappa_n
        mu = rng.normal(loc=mean_n, scale=math.sqrt(var_n))

        # 3) Gibbs: sample sigma2 | rest
        a_n = a0 + 0.5 * (n + 1.0)
        weighted_ss = np.sum(lambdas * (y - mu) ** 2)
        prior_ss = kappa0 * (mu - mu0) ** 2
        b_n = b0 + 0.5 * (weighted_ss + prior_ss)
        gamma_draw = rng.gamma(shape=a_n, scale=1.0 / b_n)
        sigma2 = 1.0 / gamma_draw

        # 4) MH-within-Gibbs for nu via eta = log(nu - 2)
        mh_steps += 1
        eta_curr = math.log(nu - 2.0)
        eta_prop = eta_curr + rng.normal(0.0, proposal_sd)
        nu_prop = 2.0 + math.exp(eta_prop)

        logp_curr = log_post_nu_given_lambda(nu, lambdas, rate_nu) + eta_curr
        logp_prop = log_post_nu_given_lambda(nu_prop, lambdas, rate_nu) + eta_prop
        log_alpha = logp_prop - logp_curr
        if math.log(rng.uniform()) < log_alpha:
            nu = nu_prop
            accepted += 1

        if it > burn_in and (it - burn_in) % thin == 0:
            saved.append((it, mu, sigma2, nu))

    samples = pd.DataFrame(saved, columns=["iter", "mu", "sigma2", "nu"])
    acceptance_rate = accepted / mh_steps if mh_steps else np.nan
    meta = {
        "n_obs": float(n),
        "n_iter": float(n_iter),
        "burn_in": float(burn_in),
        "thin": float(thin),
        "saved_draws": float(len(samples)),
        "mh_acceptance_rate_nu": float(acceptance_rate),
    }
    return samples, meta


def summarise_samples(samples: pd.DataFrame, meta: dict[str, float]) -> pd.DataFrame:
    rows = []
    for col in ["mu", "sigma2", "nu"]:
        s = samples[col]
        rows.append(
            {
                "parameter": col,
                "mean": s.mean(),
                "sd": s.std(ddof=1),
                "q2.5": s.quantile(0.025),
                "q50": s.quantile(0.50),
                "q97.5": s.quantile(0.975),
            }
        )
    summary = pd.DataFrame(rows)
    summary.attrs["meta"] = meta
    return summary


def posterior_predictive_var_es_summary(
    samples: pd.DataFrame,
    alpha: float,
    ci_level: float,
    pred_samples_per_draw: int,
    seed: int,
) -> pd.DataFrame:
    if not (0.0 < alpha < 1.0):
        raise SystemExit(f"Invalid alpha: {alpha}. Must be in (0,1).")
    if not (0.0 < ci_level < 1.0):
        raise SystemExit(f"Invalid risk-ci: {ci_level}. Must be in (0,1).")

    rng = np.random.default_rng(seed + 999)

    mu = samples["mu"].to_numpy()
    sigma = np.sqrt(samples["sigma2"].to_numpy())
    nu = samples["nu"].to_numpy()

    # Draw posterior predictive returns.
    # Shape: (n_draws, pred_samples_per_draw)
    t_draws = rng.standard_t(df=np.repeat(nu, pred_samples_per_draw))
    t_draws = t_draws.reshape(len(samples), pred_samples_per_draw)
    pred_returns = mu[:, None] + sigma[:, None] * t_draws
    pred_losses = -pred_returns

    var_level = 1.0 - alpha
    var_per_draw = np.quantile(pred_losses, var_level, axis=1)
    es_per_draw = np.empty(len(samples), dtype=float)
    for i in range(len(samples)):
        tail = pred_losses[i, pred_losses[i] >= var_per_draw[i]]
        es_per_draw[i] = float(np.mean(tail))

    q_low = (1.0 - ci_level) / 2.0
    q_high = 1.0 - q_low
    out = pd.DataFrame(
        [
            {
                "alpha": alpha,
                "confidence_level": var_level,
                "risk_ci_level": ci_level,
                "VaR_mean": float(np.mean(var_per_draw)),
                "VaR_ci_lower": float(np.quantile(var_per_draw, q_low)),
                "VaR_ci_upper": float(np.quantile(var_per_draw, q_high)),
                "ES_mean": float(np.mean(es_per_draw)),
                "ES_ci_lower": float(np.quantile(es_per_draw, q_low)),
                "ES_ci_upper": float(np.quantile(es_per_draw, q_high)),
                "n_posterior_draws": int(len(samples)),
                "pred_samples_per_draw": int(pred_samples_per_draw),
            }
        ]
    )
    return out


def main() -> None:
    args = parse_args()
    if args.n_chains < 1:
        raise SystemExit("--n-chains must be >= 1")

    y = load_log_returns(args.input, args.return_col)
    chain_samples = []
    chain_meta_rows: list[dict[str, float]] = []
    for chain_id in range(1, args.n_chains + 1):
        chain_seed = args.seed + (chain_id - 1) * args.chain_seed_stride
        samples_i, meta_i = run_mwg(
            y=y,
            n_iter=args.n_iter,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=chain_seed,
            proposal_sd=args.proposal_sd,
        )
        if samples_i.empty:
            raise SystemExit(
                "No posterior samples saved. Adjust n-iter/burn-in/thin."
            )
        samples_i["chain"] = chain_id
        chain_samples.append(samples_i)

        row = dict(meta_i)
        row["chain"] = float(chain_id)
        row["seed"] = float(chain_seed)
        chain_meta_rows.append(row)

    samples = pd.concat(chain_samples, ignore_index=True)
    if samples.empty:
        raise SystemExit("No posterior samples saved. Adjust n-iter/burn-in/thin.")

    meta_df = pd.DataFrame(chain_meta_rows).sort_values("chain")
    meta = {
        "n_obs": float(len(y)),
        "n_chains": float(args.n_chains),
        "n_iter": float(args.n_iter),
        "burn_in": float(args.burn_in),
        "thin": float(args.thin),
        "saved_draws_total": float(len(samples)),
        "saved_draws_per_chain": float(len(samples) / args.n_chains),
        "mh_acceptance_rate_nu_mean": float(meta_df["mh_acceptance_rate_nu"].mean()),
        "mh_acceptance_rate_nu_min": float(meta_df["mh_acceptance_rate_nu"].min()),
        "mh_acceptance_rate_nu_max": float(meta_df["mh_acceptance_rate_nu"].max()),
    }
    summary = summarise_samples(samples, meta)
    var_es = posterior_predictive_var_es_summary(
        samples=samples,
        alpha=args.alpha,
        ci_level=args.risk_ci,
        pred_samples_per_draw=args.pred_samples_per_draw,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = args.output_dir / "posterior_samples.csv"
    summary_path = args.output_dir / "posterior_summary.csv"
    meta_path = args.output_dir / "sampler_meta.csv"
    chain_meta_path = args.output_dir / "sampler_meta_by_chain.csv"
    risk_path = args.output_dir / "var_es_summary.csv"

    samples.to_csv(posterior_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    meta_df.to_csv(chain_meta_path, index=False)
    var_es.to_csv(risk_path, index=False)

    print(f"Saved: {posterior_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {chain_meta_path}")
    print(f"Saved: {risk_path}")
    print(
        "MH acceptance rate (nu), mean/min/max: "
        f"{meta['mh_acceptance_rate_nu_mean']:.3f}/"
        f"{meta['mh_acceptance_rate_nu_min']:.3f}/"
        f"{meta['mh_acceptance_rate_nu_max']:.3f}"
    )
    print(var_es.to_string(index=False))


if __name__ == "__main__":
    main()
