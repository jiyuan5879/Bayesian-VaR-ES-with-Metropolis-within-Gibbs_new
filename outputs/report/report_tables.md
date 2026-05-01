# STATS211 Project Result Tables

## 1) Data Summary (Log Returns)

| n_obs | start_date | end_date | mean_daily_log_return | sd_daily_log_return | annualized_mean_log_return | annualized_volatility | min_daily_log_return | max_daily_log_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 751 | 2023-04-03 | 2026-03-31 | 0.000616 | 0.009346 | 0.155336 | 0.148363 | -0.061609 | 0.090895 |

## 2) Posterior Parameter Summary

| parameter | mean | sd | q2.5 | q50 | q97.5 |
| --- | --- | --- | --- | --- | --- |
| mu | 0.000990 | 0.000270 | 0.000468 | 0.000983 | 0.001516 |
| sigma2 | 0.000040 | 0.000004 | 0.000033 | 0.000040 | 0.000047 |
| nu | 3.858463 | 0.569346 | 2.909019 | 3.800164 | 5.071844 |

## 3) VaR and ES (Loss Scale)

| confidence_level | VaR_loss | ES_loss | tail_count |
| --- | --- | --- | --- |
| 0.950000 | 0.012654 | 0.019919 | 20000 |
| 0.990000 | 0.023242 | 0.034176 | 4000 |

## 4) Convergence Diagnostics

| parameter | draws | mean | sd | mcse | ess | ess_per_draw | split_rhat | geweke_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mu | 2000 | 0.0010 | 0.0003 | 0.0000 | 1441.9390 | 0.7210 | 0.9995 | 0.5152 |
| nu | 2000 | 3.8585 | 0.5693 | 0.0443 | 165.0934 | 0.0825 | 1.0010 | 2.9605 |
| sigma2 | 2000 | 0.0000 | 0.0000 | 0.0000 | 357.3048 | 0.1787 | 1.0004 | 0.6306 |

Diagnostic note: ideally split-Rhat should be close to 1, and ESS should be reasonably large.
