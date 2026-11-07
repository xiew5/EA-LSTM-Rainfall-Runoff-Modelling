import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt

from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import eval_run


RUN_DIR = Path("runs/national_spatial_ealstm_hs128_host22_without_runoff_BFI_1308_131106")
EPOCH = 30

BASIN_ID = None


def find_usable_basin(results):
    usable_basins = []

    for basin_id in results.keys():
        xr = results[basin_id]["1D"]["xr"]

        qobs = xr["discharge_spec_obs"].isel(time_step=-1)
        qsim = xr["discharge_spec_sim"].isel(time_step=-1)

        obs_count = qobs.count().item()
        sim_count = qsim.count().item()

        print(f"{basin_id}: obs valid = {obs_count}, sim valid = {sim_count}")

        if obs_count > 0 and sim_count > 0:
            usable_basins.append(basin_id)

    if not usable_basins:
        raise ValueError("No usable basin found. All basins have invalid observed or simulated values.")

    return usable_basins[0]


def main() -> None:
    results_file = RUN_DIR / "test" / f"model_epoch{EPOCH:03d}" / "test_results.p"

    if not results_file.exists():
        eval_run(run_dir=RUN_DIR, period="test")

    with open(results_file, "rb") as fp:
        results = pickle.load(fp)

    basin_id = BASIN_ID

    if basin_id is None:
        basin_id = find_usable_basin(results)
        print(f"Automatically selected basin: {basin_id}")

    if basin_id not in results:
        raise ValueError(
            f"Basin {basin_id} not found. Available basin IDs: {list(results.keys())}"
        )

    xr = results[basin_id]["1D"]["xr"]

    qobs = xr["discharge_spec_obs"].isel(time_step=-1)
    qsim = xr["discharge_spec_sim"].isel(time_step=-1)

    obs_count = qobs.count().item()
    sim_count = qsim.count().item()

    print(f"Selected basin: {basin_id}")
    print(f"Valid observed values: {obs_count}")
    print(f"Valid simulated values: {sim_count}")

    if obs_count == 0:
        raise ValueError(
            f"Basin {basin_id} has no valid observed values. "
            "Choose another basin or check its original timeseries file."
        )

    if sim_count == 0:
        raise ValueError(
            f"Basin {basin_id} has no valid simulated values."
        )

    values = metrics.calculate_all_metrics(qobs, qsim)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(qobs["date"], qobs)
    ax.plot(qsim["date"], qsim)

    ax.set_ylabel("Discharge (mm/d)")
    ax.set_title(f"Test period - Basin {basin_id} - NSE {values['NSE']:.3f}")
    ax.legend(["observed", "simulated"], loc="upper right")

    plot_file = RUN_DIR / f"test_plot_{basin_id}.png"
    fig.savefig(plot_file, bbox_inches="tight")
    print(f"Saved plot to: {plot_file}")

    for key, val in values.items():
        print(f"{key}: {val:.5f}")

    plt.show()


if __name__ == "__main__":
    main()
