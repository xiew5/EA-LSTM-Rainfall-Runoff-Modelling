# Conditional static GradientSHAP: EA-LSTM epoch 30

## Experimental setup

- Run: `runs/nw_region_ealstm_hs128_all_0707_111531`
- Evaluation period: `validation`
- Explained catchments: 31
- Explained prediction dates: 372
- Dates per catchment: up to 12, evenly spaced
- Training-catchment background size: 29
- Expected-gradient draws per prediction: 256
- Dynamic precipitation/PET histories are fixed for each prediction; only static attributes vary.
- SHAP values are expressed in specific-discharge prediction units (mm d^-1).

## Top static attributes

| Rank | Attribute | Mean absolute SHAP | Median absolute SHAP | Mean SHAP |
|---:|---|---:|---:|---:|
| 1 | HostClass30 | 0.108967 | 0.040738 | 0.000898 |
| 2 | HostClass29 | 0.100029 | 0.046372 | 0.013661 |
| 3 | HostClass24 | 0.090077 | 0.047070 | -0.007970 |
| 4 | BFIHOST | 0.088726 | 0.038976 | -0.001666 |
| 5 | Runoff_mm | 0.074176 | 0.030574 | 0.003576 |
| 6 | HostClass8 | 0.069602 | 0.027064 | 0.004260 |
| 7 | HostClass9 | 0.069431 | 0.017010 | -0.001552 |
| 8 | Area_km2 | 0.063917 | 0.025435 | 0.002179 |
| 9 | HostClass26 | 0.056210 | 0.024478 | 0.002285 |
| 10 | HostClass21 | 0.049612 | 0.023713 | -0.002286 |

## Comparison with permutation importance

The 26-feature rank Spearman correlation is ρ=0.7258 (p=2.71e-05).

## Approximation diagnostics

Median absolute additivity residual: 0.062606 mm d^-1.
95th percentile absolute additivity residual: 0.249442 mm d^-1.

These are conditional static Expected-Gradient/GradientSHAP attributions, not causal effects. Individual HOST proportions are compositional and correlated, so the grouped HOST result remains the more physically defensible primary interpretation.
