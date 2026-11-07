# Conditional static GradientSHAP: EA-LSTM epoch 30

## Experimental setup

- Run: `runs/national_spatial_ealstm_hs128_all_without_runoff_BFI_0908_030304`
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
| 1 | HostClass29 | 0.169142 | 0.039093 | 0.062384 |
| 2 | SAAR_6190_mm | 0.161151 | 0.054204 | 0.086051 |
| 3 | HostClass24 | 0.102889 | 0.044819 | 0.001429 |
| 4 | HostClass17 | 0.101602 | 0.030644 | 0.011602 |
| 5 | PE_6190_mm | 0.088352 | 0.034034 | -0.016562 |
| 6 | HostClass26 | 0.086768 | 0.033856 | -0.019168 |
| 7 | HostClass21 | 0.077126 | 0.042039 | 0.023482 |
| 8 | HostClass1 | 0.067397 | 0.029318 | 0.005292 |
| 9 | Area_km2 | 0.066053 | 0.020938 | 0.021107 |
| 10 | HostClass2 | 0.065794 | 0.020668 | 0.004372 |

## Comparison with permutation importance

The 32-feature rank Spearman correlation is ρ=0.4355 (p=0.01273).

## Approximation diagnostics

Median absolute additivity residual: 0.083412 mm d^-1.
95th percentile absolute additivity residual: 0.629373 mm d^-1.

These are conditional static Expected-Gradient/GradientSHAP attributions, not causal effects. Individual HOST proportions are compositional and correlated, so the grouped HOST result remains the more physically defensible primary interpretation.
