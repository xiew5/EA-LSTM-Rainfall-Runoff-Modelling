from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/permutation_importance_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.utils import metrics_to_dataframe
from neuralhydrology.utils.config import Config


DEFAULT_RUN_DIR = Path("runs/experimentNSE30tun0hs128_3006_170741")
DEFAULT_EPOCH = 30
DEFAULT_SEED = 42

NON_HOST_GROUPS = {
    "Area": ["Area_km2"],
    "Climate": ["PE_6190_mm", "SAAR_6190_mm"],
    "Hydrology": ["Runoff_mm", "BFIHOST"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Catchment-wise permutation importance for a trained EA-LSTM."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument(
        "--period", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--mode", choices=("groups", "individual"), default="groups"
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--engine",
        choices=("batched", "standard"),
        default="batched",
        help="batched reuses dynamic embeddings and evaluates several perturbations together.",
    )
    parser.add_argument(
        "--scenario-batch-size",
        type=int,
        default=16,
        help="Number of permutation scenarios evaluated together by the batched engine.",
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=1,
        help=(
            "Number of feature sets evaluated in one shared batched pass. "
            "Values above 1 reduce repeated dynamic-input work in individual mode."
        ),
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Evaluate every Nth validation sequence. Use 1 for the full daily set, 7 for weekly sampling.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, or cuda:X. auto uses the run config with a safe fallback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/permutation_importance/<run>/<period>_epochNNN.",
    )
    parser.add_argument(
        "--max-basins",
        type=int,
        default=None,
        help="Development/smoke-test option. Do not use for final results.",
    )
    return parser.parse_args()


def resolve_device(cfg: Config, requested: str) -> str:
    device = cfg.device if requested == "auto" else requested
    if device is None:
        device = "cpu"

    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS is unavailable in this process; falling back to CPU.")
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable in this process; falling back to CPU.")
        return "cpu"
    return device


def feature_sets(mode: str, attribute_names: list[str]) -> dict[str, list[str]]:
    if mode == "individual":
        return {name: [name] for name in attribute_names}

    configured = set(attribute_names)
    # Some trained configurations intentionally omit one or more complete
    # feature groups (for example Runoff_mm and BFIHOST). Build grouped
    # permutation scenarios only from attributes that the model actually uses.
    groups = {
        name: [feature for feature in features if feature in configured]
        for name, features in NON_HOST_GROUPS.items()
    }
    groups = {name: features for name, features in groups.items() if features}
    groups["HOST"] = [name for name in attribute_names if name.startswith("HostClass")]

    grouped = {name for features in groups.values() for name in features}
    ungrouped = configured - grouped
    if ungrouped:
        raise ValueError(f"Static attributes are not assigned to a group: {sorted(ungrouped)}")
    return groups


def metric_frame(results: dict, metrics: Iterable[str]) -> pd.DataFrame:
    frame = metrics_to_dataframe(results, metrics, ["discharge_spec"])
    frame.index = frame.index.astype(str)
    return frame.sort_index()


def numpy_nse_kge(obs: np.ndarray, simulations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized NSE/KGE for one observed series and many simulated series."""
    valid_obs = np.isfinite(obs)
    obs = obs[valid_obs].astype(np.float64, copy=False)
    simulations = simulations[:, valid_obs].astype(np.float64, copy=False)
    if len(obs) < 2:
        nan_values = np.full(simulations.shape[0], np.nan)
        return nan_values, nan_values.copy()

    # The trained regression model should not produce NaNs. Retain a slower,
    # scenario-wise fallback so the result remains well-defined if it does.
    if not np.isfinite(simulations).all():
        nse_values = np.full(simulations.shape[0], np.nan)
        kge_values = np.full(simulations.shape[0], np.nan)
        for index, sim in enumerate(simulations):
            valid = np.isfinite(sim)
            if valid.sum() < 2:
                continue
            nse_values[index], kge_values[index] = numpy_nse_kge(
                obs[valid], sim[valid][None, :]
            )
        return nse_values, kge_values

    obs_mean = obs.mean()
    obs_centered = obs - obs_mean
    obs_sum_squares = np.sum(obs_centered**2)
    obs_std = obs.std()

    errors = simulations - obs[None, :]
    nse_values = 1.0 - np.sum(errors**2, axis=1) / obs_sum_squares

    sim_means = simulations.mean(axis=1)
    sim_centered = simulations - sim_means[:, None]
    sim_stds = simulations.std(axis=1)
    correlations = np.sum(sim_centered * obs_centered[None, :], axis=1) / np.sqrt(
        np.sum(sim_centered**2, axis=1) * obs_sum_squares
    )
    alpha = sim_stds / obs_std
    beta = sim_means / obs_mean
    kge_values = 1.0 - np.sqrt(
        (correlations - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
    )
    return nse_values, kge_values


def fast_ealstm_predictions(
    tester,
    dataset,
    static_vectors: torch.Tensor,
    scenario_batch_size: int,
    sample_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict many static perturbations while sharing each dynamic embedding.

    This is mathematically equivalent to separate EA-LSTM forward passes in
    evaluation mode. It avoids recomputing the dynamic embedding and, because
    only ``predict_last_n=1`` is needed, avoids storing all recurrent states.
    """
    model = tester.model
    cfg = tester.cfg
    if cfg.model.lower() != "ealstm" or cfg.predict_last_n != 1:
        raise ValueError("The batched engine requires EA-LSTM with predict_last_n=1.")
    if len(dataset.frequencies) != 1:
        raise ValueError("The batched engine currently supports one frequency.")
    if model.embedding_net.nan_handling_method is not None:
        raise ValueError("The batched engine does not support a configured NaN handling method.")
    if len(cfg.target_variables) != 1:
        raise ValueError("The batched engine currently supports one target variable.")

    device = tester.device
    static_vectors = static_vectors.to(device)
    n_scenarios = static_vectors.shape[0]
    dynamic_features = list(itertools.chain(*model.embedding_net._dynamic_inputs))
    inference_dataset = dataset
    if sample_stride > 1:
        inference_dataset = Subset(dataset, range(0, len(dataset), sample_stride))
    loader = DataLoader(
        inference_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    observations = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            dynamic_data = {key: value.to(device) for key, value in data["x_d"].items()}
            raw_dynamic = torch.cat(
                [dynamic_data[key] for key in dynamic_features], dim=-1
            ).transpose(0, 1)
            embedded_dynamic = model.embedding_net.dynamics_embeddings[0](raw_dynamic)
            batch_size = embedded_dynamic.shape[1]

            batch_predictions = []
            for start in range(0, n_scenarios, scenario_batch_size):
                scenario_static = static_vectors[start : start + scenario_batch_size]
                scenario_count = scenario_static.shape[0]
                expanded_static = scenario_static[:, None, :].expand(
                    scenario_count, batch_size, -1
                )
                embedded_static = model.embedding_net.statics_embedding(expanded_static)
                input_gate = torch.sigmoid(model.input_gate(embedded_static)).reshape(
                    scenario_count * batch_size, model._hidden_size
                )

                hidden = embedded_dynamic.new_zeros(
                    scenario_count * batch_size, model._hidden_size
                )
                cell = embedded_dynamic.new_zeros(
                    scenario_count * batch_size, model._hidden_size
                )
                for dynamic_step in embedded_dynamic:
                    expanded_dynamic = dynamic_step.unsqueeze(0).expand(
                        scenario_count, batch_size, -1
                    ).reshape(scenario_count * batch_size, -1)
                    hidden, cell = model._cell(
                        expanded_dynamic, input_gate, (hidden, cell)
                    )

                normalized_prediction = model.head(model.dropout(hidden))["y_hat"]
                batch_predictions.append(
                    normalized_prediction.reshape(scenario_count, batch_size)
                )

            all_predictions.append(torch.cat(batch_predictions, dim=0).cpu().numpy())
            observations.append(data["y"][:, -1, 0].numpy())

    normalized_simulations = np.concatenate(all_predictions, axis=1)
    normalized_observations = np.concatenate(observations)
    target = cfg.target_variables[0]
    scale = float(
        tester.scaler["xarray_feature_scale"][[target]].to_array().values.squeeze()
    )
    center = float(
        tester.scaler["xarray_feature_center"][[target]].to_array().values.squeeze()
    )
    return (
        normalized_observations * scale + center,
        normalized_simulations * scale + center,
    )


def run_batched_permutations(
    tester,
    groups: dict[str, list[str]],
    feature_indices: dict[str, list[int]],
    original_attributes: dict[str, torch.Tensor],
    usable_basins: list[str],
    baseline: pd.DataFrame,
    repeats: int,
    base_seed: int,
    scenario_batch_size: int,
    feature_batch_size: int,
    sample_stride: int,
    basin_results_path: Path,
    analysis_baseline_path: Path,
) -> pd.DataFrame:
    """Evaluate and persist all not-yet-completed grouped/individual scenarios."""
    done = completed_scenarios(basin_results_path)
    total = len(groups) * repeats
    completed = sum(
        (name, repeat) in done for name in groups for repeat in range(repeats)
    )
    if analysis_baseline_path.exists():
        analysis_baseline = pd.read_csv(analysis_baseline_path, dtype={"basin": str}).set_index("basin")
    else:
        analysis_baseline = pd.DataFrame(columns=["NSE", "KGE"])

    pending_groups = []
    for group_number, (group_name, indices) in enumerate(feature_indices.items()):
        pending = [repeat for repeat in range(repeats) if (group_name, repeat) not in done]
        if pending:
            pending_groups.append((group_number, group_name, indices, pending))

    for block_start in range(0, len(pending_groups), feature_batch_size):
        group_block = pending_groups[block_start : block_start + feature_batch_size]
        scenarios = []
        for group_number, group_name, indices, pending in group_block:
            for repeat in pending:
                scenario_seed = base_seed + group_number * 100_000 + repeat
                scenarios.append(
                    {
                        "group_name": group_name,
                        "indices": indices,
                        "repeat": repeat,
                        "scenario_seed": scenario_seed,
                        "donors": dict(
                            zip(
                                usable_basins,
                                np.random.default_rng(scenario_seed)
                                .permutation(usable_basins)
                                .tolist(),
                            )
                        ),
                    }
                )

        accumulated = {
            (scenario["group_name"], scenario["repeat"]): []
            for scenario in scenarios
        }
        block_names = ", ".join(group_name for _, group_name, _, _ in group_block)
        block_start_time = time.monotonic()
        print(
            f"Batched evaluation: [{block_names}], {len(scenarios)} pending "
            f"permutations in scenario chunks of {scenario_batch_size}."
        )
        for basin_number, basin in enumerate(usable_basins, start=1):
            vectors = [original_attributes[basin]]
            for scenario in scenarios:
                donor = scenario["donors"][basin]
                vector = original_attributes[basin].clone()
                indices = scenario["indices"]
                vector[indices] = original_attributes[donor][indices]
                vectors.append(vector)

            obs, simulations = fast_ealstm_predictions(
                tester,
                tester.cached_datasets[basin],
                torch.stack(vectors),
                scenario_batch_size,
                sample_stride,
            )
            nse_values, kge_values = numpy_nse_kge(obs, simulations)

            # Scenario zero is the unpermuted vector. It must reproduce the
            # ordinary NeuralHydrology evaluator before perturbations are used.
            if sample_stride == 1:
                expected_nse = baseline.loc[basin, "NSE"]
                expected_kge = baseline.loc[basin, "KGE"]
            elif basin in analysis_baseline.index:
                expected_nse = analysis_baseline.loc[basin, "NSE"]
                expected_kge = analysis_baseline.loc[basin, "KGE"]
            else:
                expected_nse = nse_values[0]
                expected_kge = kge_values[0]
                analysis_baseline.loc[basin, ["NSE", "KGE"]] = [expected_nse, expected_kge]

            if not (
                np.isclose(nse_values[0], expected_nse, atol=5e-4, rtol=0)
                and np.isclose(kge_values[0], expected_kge, atol=5e-4, rtol=0)
            ):
                raise RuntimeError(
                    f"Batched inference mismatch for basin {basin}: "
                    f"NSE {nse_values[0]} vs {expected_nse}, "
                    f"KGE {kge_values[0]} vs {expected_kge}."
                )

            for scenario_index, scenario in enumerate(scenarios, start=1):
                group_name = scenario["group_name"]
                repeat = scenario["repeat"]
                donor = scenario["donors"][basin]
                accumulated[(group_name, repeat)].append(
                    {
                        "feature_set": group_name,
                        "repeat": repeat,
                        "scenario_seed": scenario["scenario_seed"],
                        "basin": basin,
                        "donor_basin": donor,
                        "baseline_NSE": expected_nse,
                        "permuted_NSE": nse_values[scenario_index],
                        "delta_NSE": expected_nse - nse_values[scenario_index],
                        "baseline_KGE": expected_kge,
                        "permuted_KGE": kge_values[scenario_index],
                        "delta_KGE": expected_kge - kge_values[scenario_index],
                    }
                )

            if basin_number == 1 or basin_number % 5 == 0 or basin_number == len(usable_basins):
                print(
                    f"  [{block_names}]: {basin_number}/{len(usable_basins)} basins "
                    f"({(time.monotonic() - block_start_time) / 60:.1f} min)"
                )

        analysis_baseline.sort_index().to_csv(analysis_baseline_path, index_label="basin")

        for scenario in scenarios:
            group_name = scenario["group_name"]
            repeat = scenario["repeat"]
            rows = pd.DataFrame(accumulated[(group_name, repeat)])
            append_csv(rows, basin_results_path)
            completed += 1
            print(
                f"[{completed}/{total}] {group_name} repeat {repeat + 1}/{repeats}: "
                f"mean ΔNSE={rows['delta_NSE'].mean():.5f}, "
                f"median ΔNSE={rows['delta_NSE'].median():.5f}"
            )
    if analysis_baseline.empty:
        raise RuntimeError("No analysis baseline metrics were produced or loaded.")
    return analysis_baseline.sort_index()


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_scenarios(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    existing = pd.read_csv(path, usecols=["feature_set", "repeat"])
    return set(zip(existing["feature_set"].astype(str), existing["repeat"].astype(int)))


def build_repeat_summary(basin_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, repeat), group in basin_results.groupby(
        ["feature_set", "repeat"], sort=False
    ):
        rows.append(
            {
                "feature_set": feature_set,
                "repeat": int(repeat),
                "n_basins": int(group["delta_NSE"].notna().sum()),
                "mean_delta_NSE": group["delta_NSE"].mean(),
                "median_delta_NSE": group["delta_NSE"].median(),
                "mean_delta_KGE": group["delta_KGE"].mean(),
                "median_delta_KGE": group["delta_KGE"].median(),
                "mean_permuted_NSE": group["permuted_NSE"].mean(),
                "median_permuted_NSE": group["permuted_NSE"].median(),
                "mean_permuted_KGE": group["permuted_KGE"].mean(),
                "median_permuted_KGE": group["permuted_KGE"].median(),
            }
        )
    return pd.DataFrame(rows)


def q025(series: pd.Series) -> float:
    return float(series.quantile(0.025))


def q975(series: pd.Series) -> float:
    return float(series.quantile(0.975))


def build_feature_summary(
    repeat_summary: pd.DataFrame, feature_groups: dict[str, list[str]], baseline: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for feature_set, group in repeat_summary.groupby("feature_set", sort=False):
        row = {
            "feature_set": feature_set,
            "features": ", ".join(feature_groups[feature_set]),
            "n_features": len(feature_groups[feature_set]),
            "n_repeats": group["repeat"].nunique(),
            "n_basins": int(group["n_basins"].max()),
            "baseline_mean_NSE": baseline["NSE"].mean(),
            "baseline_median_NSE": baseline["NSE"].median(),
            "baseline_mean_KGE": baseline["KGE"].mean(),
            "baseline_median_KGE": baseline["KGE"].median(),
        }
        for metric in ("NSE", "KGE"):
            for aggregation in ("mean", "median"):
                column = f"{aggregation}_delta_{metric}"
                row[f"{column}_across_repeats"] = group[column].mean()
                row[f"{column}_repeat_q025"] = q025(group[column])
                row[f"{column}_repeat_q975"] = q975(group[column])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "mean_delta_NSE_across_repeats", ascending=False
    )


def plot_importance(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("mean_delta_NSE_across_repeats")
    values = ordered["mean_delta_NSE_across_repeats"].to_numpy()
    lower = values - ordered["mean_delta_NSE_repeat_q025"].to_numpy()
    upper = ordered["mean_delta_NSE_repeat_q975"].to_numpy() - values

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(ordered) + 1.5)))
    ax.barh(
        ordered["feature_set"],
        values,
        xerr=np.vstack([lower, upper]),
        color="#3973AC",
        alpha=0.9,
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Permutation importance (mean paired ΔNSE)")
    ax.set_ylabel("")
    ax.set_title("EA-LSTM static-feature permutation importance")
    ax.text(
        0.99,
        0.02,
        "Error bars: 2.5th–97.5th percentiles across permutation repeats",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    cfg: Config,
    device: str,
    attributes: list[str],
    feature_groups: dict[str, list[str]],
    basins: list[str],
) -> None:
    metadata = {
        "run_dir": str(args.run_dir),
        "checkpoint": f"model_epoch{args.epoch:03d}.pt",
        "period": args.period,
        "mode": args.mode,
        "requested_repeats": args.repeats,
        "base_seed": args.seed,
        "device": device,
        "engine": args.engine,
        "scenario_batch_size": args.scenario_batch_size,
        "feature_batch_size": args.feature_batch_size,
        "sample_stride": args.sample_stride,
        "model": cfg.model,
        "loss": cfg.loss,
        "hidden_size": cfg.hidden_size,
        "attribute_order_used_by_dataset": attributes,
        "feature_sets": feature_groups,
        "basins": basins,
        "max_basins": args.max_basins,
        "importance_definition": "baseline metric minus permuted metric; positive means performance deteriorated",
        "permutation_unit": "catchment; one donor assignment is shared by all daily samples of a catchment",
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.max_basins is not None and args.max_basins < 2:
        raise ValueError("--max-basins must be at least 2")
    if args.scenario_batch_size < 1:
        raise ValueError("--scenario-batch-size must be at least 1")
    if args.feature_batch_size < 1:
        raise ValueError("--feature-batch-size must be at least 1")
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be at least 1")

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.yml"
    checkpoint = run_dir / f"model_epoch{args.epoch:03d}.pt"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    cfg = Config(config_path)
    device = resolve_device(cfg, args.device)
    cfg.device = device
    cfg.update_config({"verbose": 0})

    # Standalone validation would otherwise interpret zero as an empty random
    # subset in BaseTester.evaluate(). Explicitly request all listed basins.
    cfg.validate_n_random_basins = sys.maxsize

    if args.output_dir is None:
        output_dir = (
            Path("outputs")
            / "permutation_importance"
            / run_dir.name
            / f"{args.period}_epoch{args.epoch:03d}_{args.mode}"
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    basin_results_path = output_dir / "basin_results.csv"
    baseline_path = output_dir / "baseline_metrics.csv"
    full_baseline_path = output_dir / "baseline_full_daily_metrics.csv"
    repeat_summary_path = output_dir / "repeat_summary.csv"
    feature_summary_path = output_dir / "feature_summary.csv"

    tester = get_tester(cfg=cfg, run_dir=run_dir, period=args.period, init_model=True)
    if args.max_basins is not None:
        tester.basins = tester.basins[: args.max_basins]

    print(f"Loading epoch {args.epoch} and evaluating the unpermuted baseline...")
    start = time.monotonic()
    baseline_results = tester.evaluate(
        epoch=args.epoch,
        save_results=False,
        save_all_output=False,
        metrics=["NSE", "KGE"],
    )
    baseline = metric_frame(baseline_results, ["NSE", "KGE"])
    baseline.to_csv(full_baseline_path, index_label="basin")

    usable_basins = [
        basin
        for basin in baseline.index
        if basin in tester.cached_datasets
        and np.isfinite(baseline.loc[basin, "NSE"])
        and np.isfinite(baseline.loc[basin, "KGE"])
    ]
    if len(usable_basins) < 2:
        raise RuntimeError("Fewer than two usable catchments were available for permutation.")
    tester.basins = usable_basins

    # BaseDataset sorts static columns alphabetically before constructing x_s.
    # The training scaler keys preserve that exact tensor order.
    attribute_names = list(tester.scaler["attribute_means"].keys())
    groups = feature_sets(args.mode, attribute_names)
    feature_indices = {
        group: [attribute_names.index(feature) for feature in features]
        for group, features in groups.items()
    }

    original_attributes = {
        basin: tester.cached_datasets[basin]._attributes[basin].clone()
        for basin in usable_basins
    }
    if any(len(vector) != len(attribute_names) for vector in original_attributes.values()):
        raise RuntimeError("Static tensor size does not match the scaler attribute order.")

    baseline = baseline.loc[usable_basins]
    print(
        f"Baseline complete in {(time.monotonic() - start) / 60:.1f} min: "
        f"{len(usable_basins)} usable basins, mean/median NSE "
        f"{baseline['NSE'].mean():.5f}/{baseline['NSE'].median():.5f}."
    )
    write_metadata(
        output_dir / "metadata.json",
        args,
        cfg,
        device,
        attribute_names,
        groups,
        usable_basins,
    )

    if args.engine == "batched":
        if args.sample_stride == 1:
            baseline.to_csv(baseline_path, index_label="basin")
        baseline = run_batched_permutations(
            tester=tester,
            groups=groups,
            feature_indices=feature_indices,
            original_attributes=original_attributes,
            usable_basins=usable_basins,
            baseline=baseline,
            repeats=args.repeats,
            base_seed=args.seed,
            scenario_batch_size=args.scenario_batch_size,
            feature_batch_size=args.feature_batch_size,
            sample_stride=args.sample_stride,
            basin_results_path=basin_results_path,
            analysis_baseline_path=baseline_path,
        )
    else:
        if args.sample_stride != 1:
            raise ValueError("--sample-stride is only supported by the batched engine.")
        baseline.to_csv(baseline_path, index_label="basin")
        done = completed_scenarios(basin_results_path)
        total = len(groups) * args.repeats
        completed = sum(
            (name, repeat) in done for name in groups for repeat in range(args.repeats)
        )

        try:
            for group_number, (group_name, indices) in enumerate(feature_indices.items()):
                for repeat in range(args.repeats):
                    scenario = (group_name, repeat)
                    if scenario in done:
                        continue

                    scenario_seed = args.seed + group_number * 100_000 + repeat
                    rng = np.random.default_rng(scenario_seed)
                    donors = rng.permutation(usable_basins).tolist()

                    for recipient, donor in zip(usable_basins, donors):
                        vector = original_attributes[recipient].clone()
                        vector[indices] = original_attributes[donor][indices]
                        tester.cached_datasets[recipient]._attributes[recipient] = vector

                    scenario_start = time.monotonic()
                    permuted_results = tester.evaluate(
                        save_results=False,
                        save_all_output=False,
                        metrics=["NSE", "KGE"],
                        model=tester.model,
                    )
                    permuted = metric_frame(
                        permuted_results, ["NSE", "KGE"]
                    ).reindex(usable_basins)

                    rows = pd.DataFrame(
                        {
                            "feature_set": group_name,
                            "repeat": repeat,
                            "scenario_seed": scenario_seed,
                            "basin": usable_basins,
                            "donor_basin": donors,
                            "baseline_NSE": baseline["NSE"].to_numpy(),
                            "permuted_NSE": permuted["NSE"].to_numpy(),
                            "delta_NSE": baseline["NSE"].to_numpy()
                            - permuted["NSE"].to_numpy(),
                            "baseline_KGE": baseline["KGE"].to_numpy(),
                            "permuted_KGE": permuted["KGE"].to_numpy(),
                            "delta_KGE": baseline["KGE"].to_numpy()
                            - permuted["KGE"].to_numpy(),
                        }
                    )
                    append_csv(rows, basin_results_path)
                    completed += 1
                    elapsed = time.monotonic() - scenario_start
                    print(
                        f"[{completed}/{total}] {group_name} repeat {repeat + 1}/{args.repeats}: "
                        f"mean ΔNSE={rows['delta_NSE'].mean():.5f}, "
                        f"median ΔNSE={rows['delta_NSE'].median():.5f} "
                        f"({elapsed / 60:.1f} min)"
                    )
        finally:
            for basin, vector in original_attributes.items():
                tester.cached_datasets[basin]._attributes[basin] = vector

    basin_results = pd.read_csv(basin_results_path)
    repeat_summary = build_repeat_summary(basin_results)
    repeat_summary.to_csv(repeat_summary_path, index=False)
    summary = build_feature_summary(repeat_summary, groups, baseline)
    summary.to_csv(feature_summary_path, index=False)
    plot_importance(summary, output_dir / "permutation_importance_nse.png")

    print("\nPermutation importance complete.")
    print(summary[[
        "feature_set",
        "n_features",
        "mean_delta_NSE_across_repeats",
        "median_delta_NSE_across_repeats",
        "mean_delta_KGE_across_repeats",
    ]].to_string(index=False))
    print(f"\nResults saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
