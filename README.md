# Estimation of Value-at-Risk and Expected Shortfall  
### Using Bayesian Student-t Modeling and Metropolis-within-Gibbs Sampling

## Overview

This project develops a Bayesian framework for modeling financial return distributions and estimating extreme downside risk. Using daily log returns of the S&P 500 index, we apply a Student-t model to capture heavy-tailed behavior commonly observed in financial markets.

Posterior inference is conducted via a Metropolis-within-Gibbs (MWG) sampler, allowing us to estimate both model parameters and predictive risk measures. Based on the posterior predictive distribution, we compute Value-at-Risk (VaR) and Expected Shortfall (ES) at the 99% risk level.

The results show clear evidence of heavy tails in return distributions and highlight the importance of modeling tail risk beyond standard normal assumptions.

---

## Key Results

- Posterior mean of degrees-of-freedom parameter:  
  **ν ≈ 3.96** → strong heavy-tail evidence  

- 99% Risk Measures (daily loss scale):
  - **VaR ≈ 2.20%**  
  - **ES ≈ 3.16%**

- ES significantly exceeds VaR, indicating substantial tail risk.

---

## Methodology

### Model

We model daily log returns using a Student-t distribution:

- Captures heavy tails better than normal distribution  
- Implemented via a scale-mixture (Gaussian + Gamma latent variables)

### Bayesian Framework

- Weakly informative priors  
- Full posterior inference using MCMC  

### Sampling

- Metropolis-within-Gibbs (MWG)
  - Gibbs updates: μ, σ², λ  
  - Metropolis step: ν  

- Multi-chain setup:
  - 4 chains  
  - Burn-in: 20,000  
  - Total posterior samples: 20,000  

### Diagnostics

- Trace plots  
- Autocorrelation (ACF)  
- $\hat{R}$ and Effective Sample Size (ESS)  

All diagnostics indicate good convergence.

---

## Repository Structure

```bash
.
├── data/                      # cleaned return data
├── outputs/                   # intermediate outputs
├── outputs_new/               # final results (used in report)
├── outputs/diagnostics/       # diagnostic plots
├── figures/                   # figures used in report
├── scripts/                   # data processing and MCMC code
├── main.tex                   # final report (LaTeX)
└── README.md
