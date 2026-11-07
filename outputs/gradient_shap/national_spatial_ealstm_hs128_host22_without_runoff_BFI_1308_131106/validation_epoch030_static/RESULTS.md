# Conditional static GradientSHAP: EA-LSTM epoch 30

## Experimental setup

- Run: `runs/national_spatial_ealstm_hs128_host22_without_runoff_BFI_1308_131106`
- Evaluation period: `validation`
- Explained catchments: 33
- Explained prediction dates: 396
- Dates per catchment: up to 12, evenly spaced
- Training-catchment background size: 32
- Expected-gradient draws per prediction: 64
- Dynamic precipitation/PET histories are fixed for each prediction; only static attributes vary.
- SHAP values are expressed in specific-discharge prediction units (mm d^-1).

## Top static attributes

| Rank | Attribute | Mean absolute SHAP | Median absolute SHAP | Mean SHAP |
|---:|---|---:|---:|---:|
| 1 | HostClass29 | 0.210257 | 0.057171 | 0.066419 |
| 2 | PE_6190_mm | 0.176316 | 0.058430 | -0.000126 |
| 3 | HostClass17 | 0.129342 | 0.053139 | 0.034549 |
| 4 | SAAR_6190_mm | 0.121450 | 0.045758 | 0.031750 |
| 5 | HostClass24 | 0.113993 | 0.054499 | 0.003576 |
| 6 | Area_km2 | 0.100076 | 0.037563 | 0.020914 |
| 7 | HostClass2 | 0.096420 | 0.032493 | 0.016023 |
| 8 | HostClass1 | 0.091516 | 0.043449 | -0.027490 |
| 9 | HostClass4 | 0.081882 | 0.042061 | 0.020247 |
| 10 | HostClass21 | 0.075897 | 0.034331 | 0.025549 |

## Comparison with permutation importance

The 24-feature rank Spearman correlation is ρ=0.5670 (p=0.003866).

## Approximation diagnostics

Median absolute additivity residual: 0.100107 mm d^-1.
95th percentile absolute additivity residual: 0.688195 mm d^-1.

These are conditional static Expected-Gradient/GradientSHAP attributions, not causal effects. Individual HOST proportions are compositional and correlated, so the grouped HOST result remains the more physically defensible primary interpretation.
