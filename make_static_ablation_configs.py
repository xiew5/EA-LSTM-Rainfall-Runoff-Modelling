from copy import deepcopy
from pathlib import Path

import yaml


BASE_CONFIG = Path("config/neuralhydrology_experment1.yml")
OUT_DIR = Path("config/ablation")

HOST_CLASSES = [f"HostClass{i}" for i in range(1, 28)] + ["HostClass29", "HostClass30"]

FEATURE_SETS = {
    "all": [
        "Area_km2",
        "PE_6190_mm",
        "Runoff_mm",
        "SAAR_6190_mm",
        "BFIHOST",
        *HOST_CLASSES,
    ],
    "no_hostclass": [
        "Area_km2",
        "PE_6190_mm",
        "Runoff_mm",
        "SAAR_6190_mm",
        "BFIHOST",
    ],
    "climate": [
        "PE_6190_mm",
        "SAAR_6190_mm",
    ],
    "hydrology": [
        "Runoff_mm",
        "BFIHOST",
    ],
    "area_only": [
        "Area_km2",
    ],
    "hostclass": HOST_CLASSES,
}


def main() -> None:
    with BASE_CONFIG.open() as fp:
        base = yaml.safe_load(fp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, static_attributes in FEATURE_SETS.items():
        cfg = deepcopy(base)
        cfg["experiment_name"] = f"experimentNSE30hs128_{name}"
        cfg["hidden_size"] = 128
        cfg["output_dropout"] = 0.4
        cfg["learning_rate"] = {0: 1e-3, 10: 5e-4, 20: 1e-4}
        cfg["static_attributes"] = static_attributes

        out_file = OUT_DIR / f"neuralhydrology_hs128_static_{name}.yml"
        with out_file.open("w") as fp:
            yaml.safe_dump(cfg, fp, sort_keys=False)

        print(out_file)


if __name__ == "__main__":
    main()
