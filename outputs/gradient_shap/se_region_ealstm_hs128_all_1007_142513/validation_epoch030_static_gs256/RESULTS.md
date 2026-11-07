# Conditional static GradientSHAP: EA-LSTM epoch 30

## Experimental setup

- Run: `runs/se_region_ealstm_hs128_all_1007_142513`
- Evaluation period: `validation`
- Explained catchments: 146
- Explained prediction dates: 1752
- Dates per catchment: up to 12, evenly spaced
- Training-catchment background size: 32
- Expected-gradient draws per prediction: 256
- Dynamic precipitation/PET histories are fixed for each prediction; only static attributes vary.
- SHAP values are expressed in specific-discharge prediction units (mm d^-1).

## Top static attributes

| Rank | Attribute | Mean absolute SHAP | Median absolute SHAP | Mean SHAP |
|---:|---|---:|---:|---:|
| 1 | HostClass1 | 0.069122 | 0.029115 | 0.000124 |
| 2 | Runoff_mm | 0.056364 | 0.026718 | 0.005782 |
| 3 | PE_6190_mm | 0.046233 | 0.025317 | 0.008981 |
| 4 | Area_km2 | 0.043473 | 0.020658 | 0.002013 |
| 5 | HostClass21 | 0.037885 | 0.015408 | -0.013912 |
| 6 | HostClass10 | 0.035402 | 0.021385 | 0.000320 |
| 7 | HostClass2 | 0.034317 | 0.013804 | 0.009363 |
| 8 | HostClass11 | 0.033237 | 0.016178 | 0.000086 |
| 9 | HostClass6 | 0.030714 | 0.013537 | 0.006387 |
| 10 | HostClass5 | 0.028875 | 0.012102 | 0.000587 |

## Comparison with permutation importance

The 27-feature rank Spearman correlation is ρ=0.7167 (p=2.603e-05).

## Approximation diagnostics

Median absolute additivity residual: 0.034031 mm d^-1.
95th percentile absolute additivity residual: 0.236793 mm d^-1.

These are conditional static Expected-Gradient/GradientSHAP attributions, not causal effects. Individual HOST proportions are compositional and correlated, so the grouped HOST result remains the more physically defensible primary interpretation.
