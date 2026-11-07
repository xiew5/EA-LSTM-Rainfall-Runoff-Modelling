# Catchment embedding analysis

- Run: `experimentNSE30tun0hs128_3006_170741`
- Checkpoint: `model_epoch030.pt`
- Model / hidden size: `ealstm` / `128`
- Catchments: 219
- Static inputs: 34
- Similarity representation: 128-dimensional sigmoid EA-LSTM input gate
- Neighbour ranking: Euclidean distance in input-gate space
- PCA variance (PC1 / PC2): 0.5215 / 0.1234
- Spearman correlation between pairwise gate distance and standardized static-attribute distance: 0.3125

## Cluster sizes

- Cluster 0: 42
- Cluster 1: 38
- Cluster 2: 34
- Cluster 3: 32
- Cluster 4: 32
- Cluster 5: 41

Embeddings from different trained runs are not directly comparable. Run this analysis separately for each checkpoint and compare within-run geometry or aggregate diagnostics.



Nearest catchments to 22007:
 rank neighbour_basin neighbour_split  gate_euclidean_distance
    1           28030           train                 1.390229
    2           24007            test                 1.826121
    3           24002           train                 1.834974
    4           23004           train                 1.972407
    5           76008           train                 2.003761

Nearest catchments to 29003:
 rank neighbour_basin neighbour_split  gate_euclidean_distance
    1           47018           train                 1.560858
    2           23018      validation                 1.759471
    3           27084           train                 1.824733
    4           28041           train                 1.841393
    5           27051           train                 1.855395

The learned EA-LSTM embedding placed both test catchments close to catchments represented during model development. Catchment 22007 combined a close training analogue with strong predictive performance, suggesting successful transfer of learned hydrological behaviour. In contrast, catchment 29003 also had several close training analogues but performed poorly, indicating that proximity in the static gate-embedding space alone is insufficient to guarantee reliable temporal predictions.