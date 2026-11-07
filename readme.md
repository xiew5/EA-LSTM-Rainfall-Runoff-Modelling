# EA-LSTM Rainfall–Runoff Modelling

This repository is a hydrological modelling and interpretation project built around a local copy of the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework. It trains entity-aware LSTM (EA-LSTM) rainfall–runoff models on CAMELS-GB-style data, compares national and regional basin splits, tests static-catchment feature ablations, and interprets trained models using catchment embeddings, permutation importance, and conditional GradientSHAP. A separate Q90 workflow screens static attributes with conventional machine-learning regressors.

The main daily inputs are precipitation and potential evapotranspiration (`peti`); the prediction target is specific discharge (`discharge_spec`, normally in mm/day). Static inputs include catchment area, climate and hydrological descriptors, and HOST soil-class proportions.

## Repository map

```text
FinalProject/
├── train.py                         # Main training launcher
├── result_analysis.py               # Evaluate one run and plot one basin
├── catchment_embedding_analysis.py  # Extract and compare EA-LSTM embeddings
├── permutation_importance_analysis.py
├── static_gradient_shap_analysis.py
├── prepare_national_host22_attributes.py
├── make_national_tuning_configs.py
├── make_static_ablation_configs.py
├── config/                          # Experiment YAML files
├── basins/                          # Train/validation/test and regional basin lists
├── NATIONAL_input/                  # Local licensed input (not tracked by Git)
├── NATIONAL_HOST22_input/           # Local derived HOST22 input (not tracked)
├── NW_REGION_CAMELS_input/          # Local regional input (not tracked)
├── SE_REGION_CAMELS_input/          # Local regional input (not tracked)
├── Q90/                             # Q90 script; input CSVs are not tracked
├── runs/                            # Local checkpoints/evaluations (not tracked)
├── outputs/                         # Generated tables, metadata, and figures
├── neuralhydrology/                 # Pinned upstream NeuralHydrology submodule
├── DATA.md                          # Input and model-artifact policy
├── data/                            # Notes about the original source data
└── evaluation/                      # Reserved, currently empty
```

Local/generated entries at the root are `.venv/` (a Python virtual environment), `__pycache__/` (compiled bytecode), `.DS_Store` (macOS metadata), and `FinalProject.code-workspace` (VS Code workspace settings). They are not part of the scientific workflow. Licensed input folders and `runs/` are also excluded; see [`DATA.md`](DATA.md) for the expected layout and publication policy. `neuralhydrology/` is pinned as a Git submodule, while its local `.venv/` remains untracked.

## Installation and quick start

NeuralHydrology declares Python 3.10 or newer and its core dependencies in `neuralhydrology/pyproject.toml`. From the project root:

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./neuralhydrology
```

The Q90 analysis additionally needs scikit-learn:

```bash
python -m pip install scikit-learn
```

Most supplied training configurations use Apple Metal (`device: mps`). Change this field to `cpu` or an available `cuda:N` device when necessary.

Start the default experiment or pass another YAML configuration:

```bash
python train.py
python train.py config/without_runoff/national_spatial_split_nse_hs128_host22_without_runoff_BFI.yml
```

Training creates a timestamped directory under `runs/`. A typical run contains the resolved `config.yml`, model checkpoints (`model_epochNNN.pt`), optimizer states, TensorBoard events, `output.log`, a fitted data scaler, validation figures, and test results/metrics.

## Project-specific Python programs

| File | Purpose |
|---|---|
| `train.py` | Minimal command-line entry point. It calls `neuralhydrology.nh_run.start_run()` with the YAML path supplied as its first argument, defaulting to `config/neuralhydrology_experment1.yml`. |
| `result_analysis.py` | Evaluates a hard-coded run if test results do not yet exist, selects the requested or first usable basin, calculates all NeuralHydrology metrics, plots observed against simulated discharge, and saves `test_plot_<basin>.png`. Edit `RUN_DIR`, `EPOCH`, and optional `BASIN_ID` before use. |
| `catchment_embedding_analysis.py` | Loads a trained EA-LSTM and its scaler, reconstructs standardized static inputs, and exports static-network outputs, input-gate logits, and sigmoid gate embeddings. It also computes cosine/euclidean similarity, nearest neighbours, NumPy PCA, k-means clusters, regional/split labels, plots, and metadata. |
| `permutation_importance_analysis.py` | Measures catchment-wise static-feature importance from the loss of NSE/KGE after permuting either individual attributes or the Area, Climate, Hydrology, and HOST groups. It provides a standard evaluator and a faster batched EA-LSTM implementation, supports resumable CSV output and sequence subsampling, and writes summaries, confidence intervals, plots, and metadata. |
| `static_gradient_shap_analysis.py` | Computes conditional static-feature Expected Gradients (the path-sampling formulation of GradientSHAP) for selected basin dates while holding dynamic inputs fixed. It uses training catchments as backgrounds, checks additivity, supports resuming, produces global/basin summaries and plots, and compares SHAP rankings with individual and grouped permutation results. |
| `prepare_national_host22_attributes.py` | Derives the published 22-class HOST scheme from the original 30 HOST columns, verifies that class coverage is conserved, removes `Runoff_mm` and `BFIHOST`, reports zero-variance fields, and writes `NATIONAL_HOST22_input/attributes/national_host22_attributes.csv`. |
| `make_national_tuning_configs.py` | Clones a completed run configuration and generates six national EA-LSTM tuning YAMLs spanning hidden sizes 96/128, all versus HOST-only statics, and standard versus lower learning-rate schedules. |
| `make_static_ablation_configs.py` | Generates six static-feature ablation configurations: all, no HOST classes, climate only, hydrology only, area only, and HOST classes only. |
| `Q90/q90_static_feature_analysis.py` | Runs a fast, non-LSTM Q90 screening study. It detects identifiers/metadata, handles numeric and categorical inputs, compares dummy, linear, and tree/boosting regressors with cross-validation, evaluates feature subsets and PCA, ranks raw features, reports MAE/RMSE/R²/NSE, and creates diagnostic CSVs and plots. `Runoff_mm` and `BFIHOST` are intentionally excluded. |

Common analysis commands are:

```bash
python catchment_embedding_analysis.py --run-dir runs/<run_name> --epoch 30

python permutation_importance_analysis.py \
  --run-dir runs/<run_name> --period validation --mode groups \
  --epoch 30 --repeats 30 --device mps

python static_gradient_shap_analysis.py \
  --run-dir runs/<run_name> --period validation --epoch 30 --device mps

python Q90/q90_static_feature_analysis.py \
  --data-path Q90/Q90flow_percentile.csv \
  --output-dir outputs/q90_static_feature_analysis
```

Use each script's `--help` option for its complete arguments. The interpretation scripts assume an EA-LSTM checkpoint and, in their optimized paths, one target, one frequency, and `predict_last_n: 1`.

## Configuration files

`config/` contains NeuralHydrology YAML files. They define basin lists and date ranges, device/seed, validation metrics, model and embedding architecture, optimizer and learning-rate schedule, sequence length, input variables, static attributes, and output location.

| Path | Contents |
|---|---|
| `config/neuralhydrology_experment1.yml` | Default national EA-LSTM experiment used by `train.py` (the filename's `experment` spelling is intentional and referenced by code). |
| `config/national_ealstm_hs128_c.yml` | National hidden-size-128 EA-LSTM configuration. |
| `config/nw_region_ealstm_hs128_c.yml` | North-west regional model. |
| `config/se_region_ealstm_hs128_all.yml` | South-east regional model. |
| `config/ablation/` | Six generated hidden-size-128 static-feature ablation experiments. |
| `config/national_tuning/` | Six generated national architecture/feature/learning-rate variants. |
| `config/holdout/` | Four transfer/holdout tests: national-to-NW missing-forcing, national-to-SE holdout, NW missing-forcing, and SE holdout. |
| `config/without_runoff/` | National spatial/time split experiments that omit runoff and BFI; includes the derived 22-class HOST variant. |

## Data and basin folders

The input directories below are expected locally but are not distributed in this repository. See [`DATA.md`](DATA.md) for licensing, schema, and setup details.

### `NATIONAL_input/`

- `attributes/national_attributes.csv` is the model-ready table keyed by `gauge_id`, with area, PE, runoff, rainfall, BFIHOST, and HOST class fractions.
- `attributes/Catchment extraction_...csv` is the wider source/export table from which modelling attributes were collated.
- `timeseries/` contains 219 catchment CSVs named `CAMELS_GB_hydromet_timeseries_<gauge>_<dates>.csv`, with `date`, `precipitation`, `peti`, and `discharge_spec` columns.

### `NATIONAL_HOST22_input/`

- `attributes/national_host22_attributes.csv` is generated by `prepare_national_host22_attributes.py` and contains Area, PE, SAAR, and 22 HOST fields but no runoff or BFIHOST.
- `timeseries` is a symbolic link to `../NATIONAL_input/timeseries`; the large daily data are not duplicated.
- `README.md` records the Griffin et al. (2019) HOST grouping and explains why the zero-variance `HostClass28` is omitted from training configurations.

### `NW_REGION_CAMELS_input/` and `SE_REGION_CAMELS_input/`

These are regional CAMELS-GB-compatible datasets. Each has an `attributes/` table and a `timeseries/` directory with the same daily schema as the national data. The NW folder also contains catchment notes, an attribute-variance summary, a preparation report, and station-availability data. The SE folder contains preparation, time-series availability, and failed-time-series reports.

### `basins/`

Plain-text gauge-ID lists define the experimental splits:

- `train_list.txt`, `validation_list.txt`, `test_list.txt`, and `basin_list.txt` define the national split and complete national set.
- `nw_region_*` and `se_region_*` define complete regional sets and their train/validation/test lists.
- `nw_missing_forcing_test_list.txt` and `se_holdout_test_list.txt` identify five-basin challenge sets.
- `*_excluding_missing_forcing.txt` and `*_excluding_holdout.txt` prevent challenge basins from entering the corresponding training or validation data.

### `Q90/`

Contains the static-feature screening script. The local working `Q90flow_percentile.csv` dataset (Q90 in mm/day) and the original-unit CSV are excluded pending redistribution approval.

### `data/`

Contains `readme.txt`, which documents the source gauged-flow and catchment-average meteorological data used for CERF2-HadUK, including the NRFA origin and licence context. The actual model-ready subsets live in the input folders above.

## Results folders

### `runs/`

Locally contains 28 timestamped NeuralHydrology experiments, but the directory is intentionally excluded from Git. Their names encode the experiment family and often the date/time, for example baseline experiments, NSE loss/epoch tuning, hidden-size/dropout tuning, static ablations, national/regional models, transfer/holdout tests, and no-runoff/BFI spatial or temporal splits. Each run is self-contained; use its saved `config.yml` to determine its exact data split and features rather than relying only on the directory name. See [`DATA.md`](DATA.md) for the recommended release bundle.

Common run artifacts are:

- `model_epochNNN.pt` and `optimizer_state_epochNNN.pt`: PyTorch checkpoints.
- `train_data/train_data_scaler.yml`: fitted feature centring/scaling values.
- `events.out.tfevents.*`, `output.log`, and `img_log/`: TensorBoard/logging artifacts.
- `validation/` and `test/model_epochNNN/`: serialized predictions and metric CSVs.
- `test_plot_<basin>.png`: observed/simulated hydrograph created by `result_analysis.py`.

### `outputs/`

- `all_runs_test_metrics_summary.csv`: cross-run test-metric summary.
- `catchment_embeddings/<run>/`: extracted embeddings, PCA coordinates/plots, similarity and distance matrices, nearest-neighbour tables, cluster attribute means, metadata, and a generated interpretation README.
- `permutation_importance/<run>/<period_epoch_mode>/`: basin-level perturbation results, baselines, repeat/feature summaries, plots, and run metadata.
- `gradient_shap/<run>/<period_epoch_static>/`: long-form SHAP values, sample/additivity checks, basin/global summaries, comparison tables, plots, and metadata.
- `q90_static_feature_analysis/`: model comparisons, predictions/residual plots, PCA variance, keyword matches, and feature-importance rankings.
- `static_ablation_summary/`: aggregate tables comparing static-feature ablations and national/regional runs.

`evaluation/` is currently empty and reserved for future evaluation material.

## Vendored NeuralHydrology package

`neuralhydrology/` is a pinned submodule of the independently packaged upstream NeuralHydrology project. Its `pyproject.toml` defines dependencies and the `nh-run`, `nh-schedule-runs`, and `nh-results-ensemble` console commands. `README.md`, `LICENSE`, `CITATION.cff`, and `CONTRIBUTING.rst` provide upstream project, licence, citation, and contribution information. `docs/` holds Sphinx documentation, `examples/` holds notebooks/sample configurations, and `test/` holds pytest coverage. The Python files are described below.

### Package entry points

| Python file | Purpose |
|---|---|
| `neuralhydrology/neuralhydrology/__init__.py` | Package initializer. |
| `neuralhydrology/neuralhydrology/__about__.py` | Package version metadata. |
| `neuralhydrology/neuralhydrology/nh_run.py` | CLI/programmatic API for starting, continuing, fine-tuning, and evaluating runs. |
| `neuralhydrology/neuralhydrology/nh_run_scheduler.py` | Distributes multiple configuration files across one or more GPUs and concurrent run slots. |

### `neuralhydrology/neuralhydrology/datasetzoo/`

Dataset adapters convert supported hydrological datasets to the common `BaseDataset` interface.

| Python file | Purpose |
|---|---|
| `__init__.py` | Dataset factory plus runtime dataset registration. |
| `basedataset.py` | Core PyTorch dataset: loads basin data/attributes, normalizes features, builds valid sequences, handles multiple frequencies and missing data, and caches validation/test data. |
| `camelsaus.py` | CAMELS-AUS adapter, loaders, and preprocessing. |
| `camelsbr.py` | CAMELS-BR adapter, loaders, and preprocessing. |
| `camelscl.py` | CAMELS-CL adapter, loaders, and preprocessing. |
| `camelsde.py` | CAMELS-DE time-series and attribute adapter. |
| `camelsgb.py` | CAMELS-GB adapter used by this project; loads the local attribute and per-basin time-series CSVs. |
| `camelsind.py` | CAMELS-IND adapter plus raw time-series/attribute preprocessing. |
| `camelsus.py` | CAMELS-US attributes, forcing, discharge, and basin-area loading. |
| `caravan.py` | Caravan NetCDF/CSV time-series and sub-dataset attribute loading. |
| `datasetregistry.py` | Registry that maps dataset names to adapter classes. |
| `genericdataset.py` | Generic CSV/NetCDF dataset adapter. |
| `hourlycamelsus.py` | Hourly CAMELS-US extension with forcing, discharge, stage, and NetCDF loaders. |
| `lamah.py` | LamaH dataset adapter for multiple sub-datasets and temporal resolutions. |
| `template.py` | Skeleton/example for implementing another dataset adapter. |

### `neuralhydrology/neuralhydrology/datautils/`

| Python file | Purpose |
|---|---|
| `__init__.py` | Data-utility package initializer. |
| `climateindices.py` | Calculates dynamic precipitation/climate indices, including optimized rolling computations. |
| `pet.py` | Priestley–Taylor potential evapotranspiration and supporting radiation/atmospheric formulae. |
| `utils.py` | Loads scalers, HydroATLAS attributes and basin lists; validates attributes; infers, sorts, and compares time-series frequencies. |

### `neuralhydrology/neuralhydrology/evaluation/`

| Python file | Purpose |
|---|---|
| `__init__.py` | Selects the appropriate regression or uncertainty tester. |
| `evaluate.py` | Starts evaluation for a run, epoch, and data period. |
| `metrics.py` | Hydrological metrics including NSE, MSE/RMSE, KGE components, Pearson correlation, flow-duration-curve errors, and peak-timing/error metrics. |
| `plots.py` | Percentile, regression, and predictive-uncertainty plots. |
| `signatures.py` | Hydrological signatures such as high/low/zero-flow frequency and duration, Q5/Q95/mean flow, BFI, FDC slope, runoff ratio, and streamflow elasticity. |
| `tester.py` | Evaluation engine: loads datasets/checkpoints, runs inference, calculates metrics, and serializes/plots regression or uncertainty results. |
| `utils.py` | Loads basin-ID encodings and converts nested evaluation results into metric data frames. |

### `neuralhydrology/neuralhydrology/modelzoo/`

| Python file | Purpose |
|---|---|
| `__init__.py` | Model factory mapping configuration names to implementations. |
| `basemodel.py` | Abstract neural-model interface and shared configuration checks. |
| `baseconceptualmodel.py` | Base class for differentiable conceptual hydrological models. |
| `arlstm.py` | Autoregressive LSTM that feeds earlier predictions back as inputs. |
| `cudalstm.py` | Standard high-performance PyTorch/CUDA-compatible LSTM model. |
| `customlstm.py` | Explicit LSTM-cell implementation that exposes gate states for inspection. |
| `ealstm.py` | Entity-aware LSTM; static catchment attributes determine a time-invariant input gate. This is the main project model. |
| `embcudalstm.py` | LSTM variant with learned basin-ID embeddings. |
| `fc.py` | Configurable fully connected embedding/network block. |
| `gru.py` | Gated recurrent unit model. |
| `handoff_forecast_lstm.py` | Forecast model that hands states from hindcast to forecast LSTMs. |
| `head.py` | Output heads for deterministic regression and GMM, CMAL, or UMAL probabilistic predictions. |
| `hybridmodel.py` | Couples a neural network with a conceptual hydrological model. |
| `inputlayer.py` | Static/dynamic input assembly, embedding, concatenation, and optional NaN handling. |
| `mamba.py` | Mamba state-space sequence model wrapper. |
| `mclstm.py` | Mass-conserving LSTM and its normalized gate implementations. |
| `mtslstm.py` | Multi-timescale LSTM with state transfer between temporal resolutions. |
| `multihead_forecast_lstm.py` | Shared hindcast LSTM with separate forecast heads. |
| `odelstm.py` | ODE-LSTM with several numerical state-evolution methods. |
| `positional_encoding.py` | Positional encoding used by sequence/attention models. |
| `sequential_forecast_lstm.py` | Sequential hindcast-to-forecast LSTM architecture. |
| `shm.py` | Simple Hydrological Model implementation. |
| `stacked_forecast_lstm.py` | Stacked hindcast and forecast LSTM architecture. |
| `template.py` | Skeleton/example for adding a model. |
| `transformer.py` | Transformer encoder model for hydrological sequences. |
| `x_lstm.py` | xLSTM-based sequence model integration. |

### `neuralhydrology/neuralhydrology/training/`

| Python file | Purpose |
|---|---|
| `__init__.py` | Factories for optimizers, losses, and regularization objects. |
| `basetrainer.py` | Main training loop, data loading, checkpointing, validation, device handling, and resume/fine-tune logic. |
| `earlystopper.py` | Tracks validation progress and signals early stopping. |
| `logger.py` | Console, file, TensorBoard scalar, and figure logging. |
| `loss.py` | Mask-aware MSE, RMSE, NSE, GMM, CMAL, and UMAL losses. |
| `regularization.py` | Multi-frequency consistency and overlapping-forecast regularizers. |
| `train.py` | Constructs the trainer and starts model training. |

### `neuralhydrology/neuralhydrology/utils/`

| Python file | Purpose |
|---|---|
| `__init__.py` | Utility package initializer. |
| `config.py` | Central typed configuration wrapper, validation, defaults, serialization, and random run-name generation. |
| `configutils.py` | Generates configuration grids from a base YAML and parameter modifications. |
| `errors.py` | Custom exceptions for missing train/evaluation data and all-NaN series. |
| `logging_utils.py` | Logging setup plus Git hash/diff capture for reproducibility. |
| `nh_results_ensemble.py` | CLI/utilities to select validation runs and ensemble their test predictions and metrics. |
| `ratingcurve.py` | Fits and evaluates stage–discharge rating curves. |
| `samplingutils.py` | Sampling for Monte Carlo dropout, Gaussian mixtures, CMAL, and UMAL predictive distributions, including negative-value handling. |

### Tests and documentation Python files

| Python file | Purpose |
|---|---|
| `neuralhydrology/test/__init__.py` | Test package initializer. |
| `neuralhydrology/test/conftest.py` | Shared pytest options and fixtures for configurations, datasets, forcings, and model families. |
| `neuralhydrology/test/test_config_runs.py` | End-to-end daily, multi-timescale, autoregressive, embedded, Transformer, extra-feature, and missing-target configuration tests. |
| `neuralhydrology/test/test_custom_lstm.py` | Checks numerical equivalence of the custom and standard LSTM implementations. |
| `neuralhydrology/test/test_datautils.py` | Tests frequency sorting, inference, comparison, and conversion factors. |
| `neuralhydrology/test/test_mclstm.py` | Verifies mass conservation in MC-LSTM. |
| `neuralhydrology/test/test_uncertainty.py` | Tests probabilistic heads, sampling, and negative-sample handling. |
| `neuralhydrology/docs/source/conf.py` | Sphinx documentation configuration. |

The empty `__init__.py` files intentionally mark package boundaries and do not implement additional behaviour.

## Reproducibility notes

- Run commands from the repository root because configurations and several scripts use project-relative paths.
- Treat `runs/` as immutable experiment records: analysis reads the saved configuration and scaler as well as the checkpoint.
- Basin IDs should be loaded as strings so leading zeros, if present, are not lost.
- Do not compare raw embedding coordinates from independently trained models directly; their latent spaces can be rotated or permuted.
- The Q90 feature selection is designed as fast screening. Its top-N selection is estimated on the full dataset and is therefore not a fully nested, leakage-free model-selection experiment.
- Large checkpoints, serialized results, virtual environments, cache files, and generated plots are artifacts rather than source code.

## Upstream licence and citation

The vendored NeuralHydrology package is distributed under the BSD 3-Clause licence; see `neuralhydrology/LICENSE`. If the framework is used in research, follow the citation guidance in `neuralhydrology/CITATION.cff` and `neuralhydrology/README.md`. Dataset licensing and provenance should be handled separately according to `data/readme.txt` and the original data provider's terms.
