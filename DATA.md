# Data and model artefact policy

The training data and full experiment directories are intentionally not stored in this Git repository.

## Input data

The local data originate from NRFA gauged-flow records and catchment-average rainfall and potential-evapotranspiration products assembled for CERF2-HadUK. The source bundle is covered by a separate EA data licence. Obtain permission and the original files from the data owner before distributing or publishing them.

After obtaining the data, arrange the model-ready files as follows:

```text
NATIONAL_input/
├── attributes/
│   └── national_attributes.csv
└── timeseries/
    └── CAMELS_GB_hydromet_timeseries_<gauge>_<dates>.csv

NATIONAL_HOST22_input/
├── attributes/
│   └── national_host22_attributes.csv
└── timeseries -> ../NATIONAL_input/timeseries

NW_REGION_CAMELS_input/
├── attributes/
└── timeseries/

SE_REGION_CAMELS_input/
├── attributes/
└── timeseries/
```

Each daily time-series CSV must contain `date`, `precipitation`, `peti`, and `discharge_spec`. The project configurations expect the input directories at the repository root. Generate the HOST22 attribute table with:

```bash
python prepare_national_host22_attributes.py
```

The Q90 workflow expects `Q90/Q90flow_percentile.csv`; that dataset is also excluded from Git pending confirmation that redistribution is permitted.

For collaborative or automated workflows, keep these folders under a versioned data store such as DVC backed by private object storage. Do not commit restricted source data to Git LFS: LFS changes storage mechanics but does not resolve licensing or access-control requirements.

## Training runs

NeuralHydrology writes checkpoints, optimizer states, TensorBoard events, logs, fitted scalers, and serialized predictions under `runs/`. This directory is ignored because it is large and mostly reproducible from the committed configuration files.

For a published model, archive only the files needed to reproduce inference and evaluation:

- the resolved `config.yml`;
- the selected `model_epochNNN.pt` checkpoint;
- `train_data/train_data_scaler.yml`;
- basin split files and the source commit hash;
- metrics and a checksum manifest.

Attach that compact archive to a versioned GitHub Release or deposit it in Zenodo and cite its DOI. Use DVC or private object storage for all intermediate runs. The smaller tables and figures under `outputs/` are kept in Git as the project results.
