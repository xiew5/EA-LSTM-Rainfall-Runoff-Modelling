# Conditional static GradientSHAP: EA-LSTM epoch 30

## Experimental setup

- Run: `runs/national_spatial_ealstm_hs128_host22_without_runoff_BFI_1308_131106`
- Evaluation period: `validation`
- Explained catchments: 33
- Explained prediction dates: 396
- Dates per catchment: up to 12, evenly spaced
- Training-catchment background size: 32
- Expected-gradient draws per prediction: 256
- Dynamic precipitation/PET histories are fixed for each prediction; only static attributes vary.
- SHAP values are expressed in specific-discharge prediction units (mm d^-1).

## Top static attributes

| Rank | Attribute | Mean absolute SHAP | Median absolute SHAP | Mean SHAP |
|---:|---|---:|---:|---:|
| 1 | HostClass29 | 0.207286 | 0.060921 | 0.067810 |
| 2 | PE_6190_mm | 0.160922 | 0.046052 | -0.004360 |
| 3 | HostClass17 | 0.126170 | 0.052817 | 0.035805 |
| 4 | SAAR_6190_mm | 0.123221 | 0.046735 | 0.039913 |
| 5 | HostClass24 | 0.113970 | 0.052836 | 0.002847 |
| 6 | Area_km2 | 0.096358 | 0.034198 | 0.019420 |
| 7 | HostClass2 | 0.094002 | 0.036013 | 0.014153 |
| 8 | HostClass1 | 0.092670 | 0.051556 | -0.027403 |
| 9 | HostClass4 | 0.077366 | 0.039595 | 0.015747 |
| 10 | HostClass21 | 0.069531 | 0.032600 | 0.024891 |

## Comparison with permutation importance

The 24-feature rank Spearman correlation is ρ=0.5843 (p=0.002713).

## Approximation diagnostics

Median absolute additivity residual: 0.053032 mm d^-1.
95th percentile absolute additivity residual: 0.341110 mm d^-1.

These are conditional static Expected-Gradient/GradientSHAP attributions, not causal effects. Individual HOST proportions are compositional and correlated, so the grouped HOST result remains the more physically defensible primary interpretation.
