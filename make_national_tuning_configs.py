from copy import deepcopy
from pathlib import Path

import yaml


BASE_RUN_CONFIG = Path("runs/experimentNSE30_1306_113739/config.yml")
OUT_DIR = Path("config/national_tuning")

ALL_STATIC = [
    "Area_km2",
    "PE_6190_mm",
    "Runoff_mm",
    "SAAR_6190_mm",
    "BFIHOST",
    *[f"HostClass{i}" for i in range(1, 28)],
    "HostClass29",
    "HostClass30",
]

HOSTCLASS_STATIC = [
    *[f"HostClass{i}" for i in range(1, 28)],
    "HostClass29",
    "HostClass30",
]

EXPERIMENTS = [
    {
        "name": "national_ealstm_hs128_all",
        "hidden_size": 128,
        "output_dropout": 0.4,
        "learning_rate": {0: 1e-3, 10: 5e-4, 20: 1e-4},
        "static_attributes": ALL_STATIC,
    },
    {
        "name": "national_ealstm_hs128_hostclass",
        "hidden_size": 128,
        "output_dropout": 0.4,
        "learning_rate": {0: 1e-3, 10: 5e-4, 20: 1e-4},
        "static_attributes": HOSTCLASS_STATIC,
    },
    {
        "name": "national_ealstm_hs96_all",
        "hidden_size": 96,
        "output_dropout": 0.4,
        "learning_rate": {0: 1e-3, 10: 5e-4, 20: 1e-4},
        "static_attributes": ALL_STATIC,
    },
    {
        "name": "national_ealstm_hs96_hostclass",
        "hidden_size": 96,
        "output_dropout": 0.4,
        "learning_rate": {0: 1e-3, 10: 5e-4, 20: 1e-4},
        "static_attributes": HOSTCLASS_STATIC,
    },
    {
        "name": "national_ealstm_hs128_all_lowlr",
        "hidden_size": 128,
        "output_dropout": 0.4,
        "learning_rate": {0: 7e-4, 15: 3e-4, 20: 1e-4},
        "static_attributes": ALL_STATIC,
    },
    {
        "name": "national_ealstm_hs128_hostclass_lowlr",
        "hidden_size": 128,
        "output_dropout": 0.4,
        "learning_rate": {0: 7e-4, 15: 3e-4, 20: 1e-4},
        "static_attributes": HOSTCLASS_STATIC,
    },
]


def main() -> None:
    with BASE_RUN_CONFIG.open() as fp:
        base = yaml.safe_load(fp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for exp in EXPERIMENTS:
        cfg = deepcopy(base)
        cfg.pop("run_dir", None)
        cfg.pop("train_dir", None)
        cfg.pop("img_log_dir", None)
        cfg.pop("commit_hash", None)
        cfg.pop("number_of_basins", None)
        cfg["experiment_name"] = exp["name"]
        cfg["data_dir"] = "./NATIONAL_input"
        cfg["train_basin_file"] = "./basins/basin_list.txt"
        cfg["validation_basin_file"] = "./basins/basin_list.txt"
        cfg["test_basin_file"] = "./basins/test_list.txt"
        cfg["hidden_size"] = exp["hidden_size"]
        cfg["output_dropout"] = exp["output_dropout"]
        cfg["learning_rate"] = exp["learning_rate"]
        cfg["static_attributes"] = exp["static_attributes"]

        out_file = OUT_DIR / f"{exp['name']}.yml"
        with out_file.open("w") as fp:
            yaml.safe_dump(cfg, fp, sort_keys=False)
        print(out_file)


if __name__ == "__main__":
    main()
