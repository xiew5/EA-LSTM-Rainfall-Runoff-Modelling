from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("NATIONAL_input/attributes/national_attributes.csv")
OUTPUT = Path("NATIONAL_HOST22_input/attributes/national_host22_attributes.csv")

GROUPS = OrderedDict(
    {
        "HostClass22a": ["HostClass7", "HostClass8", "HostClass9", "HostClass10"],
        "HostClass22b": ["HostClass11", "HostClass12", "HostClass15"],
        "HostClass22c": ["HostClass16", "HostClass18"],
        "HostClass22d": ["HostClass20", "HostClass23"],
        "HostClass22e": ["HostClass26", "HostClass27"],
    }
)

UNMERGED_HOST_CLASSES = [
    "HostClass1",
    "HostClass2",
    "HostClass3",
    "HostClass4",
    "HostClass5",
    "HostClass6",
    "HostClass13",
    "HostClass14",
    "HostClass17",
    "HostClass19",
    "HostClass21",
    "HostClass22",
    "HostClass24",
    "HostClass25",
    "HostClass28",
    "HostClass29",
    "HostClass30",
]


def main() -> None:
    source = pd.read_csv(SOURCE, dtype={"gauge_id": str})
    original_host = [f"HostClass{number}" for number in range(1, 31)]
    required = [
        "gauge_id",
        "Area_km2",
        "PE_6190_mm",
        "SAAR_6190_mm",
        *original_host,
    ]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Missing required columns in {SOURCE}: {missing}")
    if source["gauge_id"].duplicated().any():
        duplicates = source.loc[source["gauge_id"].duplicated(), "gauge_id"].tolist()
        raise ValueError(f"Duplicate gauge IDs in {SOURCE}: {duplicates}")
    if source[required[1:]].isna().any().any():
        raise ValueError(f"Missing values found in required columns of {SOURCE}")

    output = source[["gauge_id", "Area_km2", "PE_6190_mm", "SAAR_6190_mm"]].copy()
    for column in UNMERGED_HOST_CLASSES:
        output[column] = source[column]
    for group_name, members in GROUPS.items():
        output[group_name] = source[members].sum(axis=1)

    derived_host = [*UNMERGED_HOST_CLASSES, *GROUPS.keys()]
    if len(derived_host) != 22 or len(set(derived_host)) != 22:
        raise RuntimeError("The derived HOST schema is not exactly 22 unique columns.")

    original_sum = source[original_host].sum(axis=1)
    derived_sum = output[derived_host].sum(axis=1)
    if not np.allclose(original_sum, derived_sum, rtol=0.0, atol=1e-12):
        max_difference = float((original_sum - derived_sum).abs().max())
        raise RuntimeError(f"HOST coverage was not preserved; max difference={max_difference}")

    forbidden = {"Runoff_mm", "Runoff_6190_mm", "BFIHOST"}
    present_forbidden = sorted(forbidden.intersection(output.columns))
    if present_forbidden:
        raise RuntimeError(f"Forbidden attributes present in output: {present_forbidden}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)

    zero_variance = output[derived_host].columns[output[derived_host].std() == 0].tolist()
    print(f"Wrote {len(output)} catchments to {OUTPUT}")
    print(f"HOST columns: {len(derived_host)}")
    print(f"Maximum HOST coverage-sum difference: {(original_sum - derived_sum).abs().max():.3g}")
    print(f"Zero-variance HOST columns: {zero_variance}")


if __name__ == "__main__":
    main()
