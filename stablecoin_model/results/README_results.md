# Latest quantitative results

This folder records the latest quantitative outputs discussed in the paper development.

## Full solver (medium grid) reference run

Interpretation:
- aligned with the paper's formal model
- preserves the core qualitative results
- useful as the benchmark numerical representation of the theory

Reference summary from the latest run:
- average crisis, cash-only: 0.000
- average crisis, with stablecoins: 0.185
- monotone stablecoin demand: 1.000
- monotone total FX demand: 1.000
- max welfare gain at low-to-mid overvaluation: 0.044
- min welfare difference at high overvaluation: -0.459

Reference calibration used in that run:
- rho = 4.0
- alpha = 0.86
- delta = 0.08
- rbar = 0.18
- lambda_loss = 0.35
- xi_coord = 0.35
- a_s = 0.025
- b_s = 0.055
- kappa_s = 0.06
- nu_share = 6.0
- phi_share = 0.9
- run_weight_normal = 0.3

## Balanced fast calibration (milder magnitudes)

Interpretation:
- same economic structure as the exact model
- used for screening and calibration tuning
- produces a milder welfare sign switch and less dramatic crisis differences

Representative summary:
- average crisis, cash-only: 0.000
- average crisis, with stablecoins: 0.111
- monotone stablecoin demand: 1.000
- monotone total FX demand: 1.000
- max welfare gain at low-to-mid overvaluation: 0.043
- min welfare difference at high overvaluation: -0.019

## Paper-code alignment

The paper's formal model should be interpreted as aligned with the exact / fully nested global-games solver.
The fast solver preserves the same economic structure and is used for screening, calibration, and tractable quantitative illustration.
