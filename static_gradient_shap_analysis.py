from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/static_gradient_shap_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from neuralhydrology.datautils.utils import load_basin_file
from neuralhydrology.datasetzoo.camelsgb import load_camels_gb_attributes
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config


DEFAULT_RUN_DIR = Path(
    "runs/national_spatial_ealstm_hs128_all_without_runoff_BFI_0908_030304"
)
DEFAULT_PERMUTATION_DIR = Path(
    "outputs/permutation_importance/"
    "national_spatial_ealstm_hs128_all_without_runoff_BFI_0908_030304/"
    "validation_epoch030_individual"
)
DEFAULT_GROUPED_PERMUTATION_DIR = Path(
    "outputs/permutation_importance/"
    "national_spatial_ealstm_hs128_all_without_runoff_BFI_0908_030304/"
    "validation_epoch030_groups"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conditional static-feature GradientSHAP/Expected Gradients for a trained EA-LSTM. "
            "The 365-step dynamic input is fixed for each explained prediction."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--period", choices=("validation", "test"), default="validation")
    parser.add_argument("--samples-per-basin", type=int, default=12)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument(
        "--gradient-samples",
        type=int,
        default=64,
        help="Number of random baseline/interpolation draws per explained prediction.",
    )
    parser.add_argument("--gradient-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda:X")
    parser.add_argument("--max-basins", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--permutation-dir", type=Path, default=DEFAULT_PERMUTATION_DIR)
    parser.add_argument(
        "--grouped-permutation-dir", type=Path, default=DEFAULT_GROUPED_PERMUTATION_DIR
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


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def selected_indices(length: int, requested: int) -> list[int]:
    if length < 1:
        return []
    count = min(length, requested)
    return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))


def conditional_predictions(
    model: torch.nn.Module,
    embedded_dynamic: torch.Tensor,
    static_batch: torch.Tensor,
    target_scale: float,
    target_center: float,
) -> torch.Tensor:
    """Predict one date for many static vectors while holding its dynamics fixed."""
    embedded_static = model.embedding_net.statics_embedding(static_batch)
    input_gate = torch.sigmoid(model.input_gate(embedded_static))
    batch_size = static_batch.shape[0]
    hidden = embedded_dynamic.new_zeros(batch_size, model._hidden_size)
    cell = embedded_dynamic.new_zeros(batch_size, model._hidden_size)
    for dynamic_step in embedded_dynamic:
        expanded_dynamic = dynamic_step.expand(batch_size, -1)
        hidden, cell = model._cell(expanded_dynamic, input_gate, (hidden, cell))
    normalized = model.head(model.dropout(hidden))["y_hat"].reshape(batch_size, -1)[:, 0]
    return normalized * target_scale + target_center


def embed_dynamic(model: torch.nn.Module, sample: dict, device: str) -> torch.Tensor:
    dynamic_features = list(itertools.chain(*model.embedding_net._dynamic_inputs))
    raw_dynamic = torch.cat([sample["x_d"][key] for key in dynamic_features], dim=-1)
    raw_dynamic = raw_dynamic.to(device)
    with torch.no_grad():
        embedded = model.embedding_net.dynamics_embeddings[0](raw_dynamic)
    return embedded.detach()


def gradient_shap_values(
    model: torch.nn.Module,
    embedded_dynamic: torch.Tensor,
    explained_static: torch.Tensor,
    backgrounds: torch.Tensor,
    gradient_samples: int,
    gradient_batch_size: int,
    seed: int,
    target_scale: float,
    target_center: float,
) -> tuple[np.ndarray, float, float, float]:
    """Expected Gradients, the path-sampling formulation of GradientSHAP.

    Baselines are sampled from training catchments. For every sampled baseline,
    a point is sampled uniformly on the straight path to the explained static
    vector. Dynamic inputs remain fixed, so attributions are conditional static
    contributions for the selected basin-date prediction.
    """
    rng = np.random.default_rng(seed)
    background_count = len(backgrounds)
    selected_backgrounds = rng.integers(0, background_count, size=gradient_samples)
    alphas = rng.random(gradient_samples).astype(np.float32)
    attribution_sum = torch.zeros_like(explained_static)

    for start in range(0, gradient_samples, gradient_batch_size):
        stop = min(start + gradient_batch_size, gradient_samples)
        baseline = backgrounds[selected_backgrounds[start:stop]]
        alpha = torch.from_numpy(alphas[start:stop]).to(backgrounds.device).unsqueeze(1)
        difference = explained_static.unsqueeze(0) - baseline
        interpolated = (baseline + alpha * difference).detach().requires_grad_(True)
        predictions = conditional_predictions(
            model,
            embedded_dynamic,
            interpolated,
            target_scale,
            target_center,
        )
        gradients = torch.autograd.grad(predictions.sum(), interpolated)[0]
        attribution_sum += (difference * gradients).sum(dim=0).detach()

    shap_values = attribution_sum / gradient_samples
    with torch.no_grad():
        prediction = float(
            conditional_predictions(
                model,
                embedded_dynamic,
                explained_static.unsqueeze(0),
                target_scale,
                target_center,
            )[0].cpu()
        )
        baseline_predictions = []
        for start in range(0, background_count, gradient_batch_size):
            baseline_predictions.append(
                conditional_predictions(
                    model,
                    embedded_dynamic,
                    backgrounds[start : start + gradient_batch_size],
                    target_scale,
                    target_center,
                )
            )
        expected_value = float(torch.cat(baseline_predictions).mean().cpu())
    residual = prediction - expected_value - float(shap_values.sum().cpu())
    return shap_values.cpu().numpy(), prediction, expected_value, residual


def build_feature_summary(values: pd.DataFrame) -> pd.DataFrame:
    summary = (
        values.groupby("feature", sort=False)
        .agg(
            n_samples=("shap_value", "size"),
            mean_abs_shap=("abs_shap_value", "mean"),
            median_abs_shap=("abs_shap_value", "median"),
            mean_shap=("shap_value", "mean"),
            shap_q025=("shap_value", lambda x: x.quantile(0.025)),
            shap_q975=("shap_value", lambda x: x.quantile(0.975)),
        )
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
    )
    summary.insert(0, "shap_rank", range(1, len(summary) + 1))
    return summary


def plot_global_bar(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, max(6, 0.3 * len(ordered) + 1.5)))
    ax.barh(ordered["feature"], ordered["mean_abs_shap"], color="#7B3294")
    ax.set_xlabel("Mean |conditional static GradientSHAP| (mm d$^{-1}$)")
    ax.set_ylabel("")
    ax.set_title("EA-LSTM static-feature GradientSHAP importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_beeswarm(values: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    top_features = summary.head(20)["feature"].tolist()
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.42 * len(top_features) + 1.5)))
    scatter = None
    for row_number, feature in enumerate(reversed(top_features)):
        subset = values[values["feature"] == feature]
        standardized = subset["standardized_value"].to_numpy()
        color_values = np.clip(standardized, -2.5, 2.5)
        jitter = rng.normal(0, 0.08, size=len(subset))
        scatter = ax.scatter(
            subset["shap_value"],
            row_number + jitter,
            c=color_values,
            cmap="coolwarm",
            vmin=-2.5,
            vmax=2.5,
            s=12,
            alpha=0.65,
            linewidths=0,
        )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(list(reversed(top_features)))
    ax.set_xlabel("Conditional static GradientSHAP value (mm d$^{-1}$)")
    ax.set_title("Static-feature GradientSHAP distribution")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Standardized feature value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def permutation_comparison(
    shap_summary: pd.DataFrame, permutation_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame | None, float | None, float | None]:
    path = permutation_dir / "feature_summary.csv"
    if not path.exists():
        return None, None, None
    permutation = pd.read_csv(path).rename(columns={"feature_set": "feature"})
    permutation = permutation.sort_values(
        "mean_delta_NSE_across_repeats", ascending=False
    ).reset_index(drop=True)
    permutation["permutation_rank"] = np.arange(1, len(permutation) + 1)
    comparison = shap_summary.merge(permutation, on="feature", how="inner")
    rho, p_value = spearmanr(comparison["shap_rank"], comparison["permutation_rank"])
    comparison.to_csv(output_dir / "shap_permutation_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        comparison["mean_delta_NSE_across_repeats"],
        comparison["mean_abs_shap"],
        color="#2C7FB8",
        alpha=0.8,
    )
    for _, row in comparison.head(10).iterrows():
        ax.annotate(
            row["feature"],
            (row["mean_delta_NSE_across_repeats"], row["mean_abs_shap"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0, color="#777777", linewidth=0.7)
    ax.set_xlabel("Permutation importance (mean ΔNSE)")
    ax.set_ylabel("Mean |GradientSHAP| (mm d$^{-1}$)")
    ax.set_title(f"SHAP vs permutation importance (rank Spearman ρ={rho:.3f})")
    fig.tight_layout()
    fig.savefig(output_dir / "shap_vs_permutation.png", dpi=200)
    plt.close(fig)
    return comparison, float(rho), float(p_value)


def group_shap_comparison(
    values: pd.DataFrame, grouped_permutation_dir: Path, output_dir: Path
) -> pd.DataFrame:
    def feature_group(feature: str) -> str:
        if feature == "Area_km2":
            return "Area"
        if feature in {"PE_6190_mm", "SAAR_6190_mm"}:
            return "Climate"
        if feature in {"Runoff_mm", "BFIHOST"}:
            return "Hydrology"
        if feature.startswith("HostClass"):
            return "HOST"
        raise ValueError(f"No SHAP feature group for {feature}")

    grouped = values.copy()
    grouped["feature_group"] = grouped["feature"].map(feature_group)
    per_sample = (
        grouped.groupby(["basin", "date", "feature_group"], as_index=False)["shap_value"]
        .sum()
        .rename(columns={"shap_value": "group_shap_value"})
    )
    summary = (
        per_sample.assign(abs_group_shap=lambda x: x["group_shap_value"].abs())
        .groupby("feature_group", as_index=False)
        .agg(
            mean_abs_group_shap=("abs_group_shap", "mean"),
            median_abs_group_shap=("abs_group_shap", "median"),
            mean_group_shap=("group_shap_value", "mean"),
        )
        .sort_values("mean_abs_group_shap", ascending=False)
    )
    permutation_path = grouped_permutation_dir / "feature_summary.csv"
    if permutation_path.exists():
        permutation = pd.read_csv(permutation_path).rename(
            columns={"feature_set": "feature_group"}
        )
        summary = summary.merge(permutation, on="feature_group", how="left")
    per_sample.to_csv(output_dir / "group_shap_values.csv", index=False)
    summary.to_csv(output_dir / "group_shap_summary.csv", index=False)
    return summary


def write_results(
    path: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    sample_summary: pd.DataFrame,
    rho: float | None,
    p_value: float | None,
) -> None:
    top = summary.head(10)
    lines = [
        f"# Conditional static GradientSHAP: EA-LSTM epoch {args.epoch}",
        "",
        "## Experimental setup",
        "",
        f"- Run: `{args.run_dir}`",
        f"- Evaluation period: `{args.period}`",
        f"- Explained catchments: {sample_summary['basin'].nunique()}",
        f"- Explained prediction dates: {len(sample_summary)}",
        f"- Dates per catchment: up to {args.samples_per_basin}, evenly spaced",
        f"- Training-catchment background size: {args.background_size}",
        f"- Expected-gradient draws per prediction: {args.gradient_samples}",
        "- Dynamic precipitation/PET histories are fixed for each prediction; only static attributes vary.",
        "- SHAP values are expressed in specific-discharge prediction units (mm d^-1).",
        "",
        "## Top static attributes",
        "",
        "| Rank | Attribute | Mean absolute SHAP | Median absolute SHAP | Mean SHAP |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {int(row['shap_rank'])} | {row['feature']} | "
            f"{row['mean_abs_shap']:.6f} | {row['median_abs_shap']:.6f} | "
            f"{row['mean_shap']:.6f} |"
        )
    if rho is not None:
        lines.extend(
            [
                "",
                "## Comparison with permutation importance",
                "",
                f"The {len(summary)}-feature rank Spearman correlation is "
                f"ρ={rho:.4f} (p={p_value:.4g}).",
            ]
        )
    median_abs_residual = sample_summary["additivity_residual"].abs().median()
    q95_abs_residual = sample_summary["additivity_residual"].abs().quantile(0.95)
    lines.extend(
        [
            "",
            "## Approximation diagnostics",
            "",
            f"Median absolute additivity residual: {median_abs_residual:.6f} mm d^-1.",
            f"95th percentile absolute additivity residual: {q95_abs_residual:.6f} mm d^-1.",
            "",
            "These are conditional static Expected-Gradient/GradientSHAP attributions, not causal effects. "
            "Individual HOST proportions are compositional and correlated, so the grouped HOST result remains "
            "the more physically defensible primary interpretation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for name in (
        "samples_per_basin",
        "background_size",
        "gradient_samples",
        "gradient_batch_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_basins is not None and args.max_basins < 1:
        raise ValueError("--max-basins must be at least 1")

    run_dir = args.run_dir.resolve()
    cfg = Config(run_dir / "config.yml")
    device = resolve_device(cfg, args.device)
    cfg.device = device
    cfg.update_config({"verbose": 0})
    if cfg.model.lower() != "ealstm" or cfg.predict_last_n != 1:
        raise ValueError("This analysis requires EA-LSTM with predict_last_n=1.")

    if args.output_dir is None:
        output_dir = (
            Path("outputs")
            / "gradient_shap"
            / run_dir.name
            / f"{args.period}_epoch{args.epoch:03d}_static"
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "shap_values_long.csv"
    sample_path = output_dir / "sample_summary.csv"

    tester = get_tester(cfg=cfg, run_dir=run_dir, period=args.period, init_model=True)
    tester._load_weights(epoch=args.epoch)
    model = tester.model
    model.eval()
    basins = tester.basins[: args.max_basins] if args.max_basins else tester.basins

    feature_names = list(tester.scaler["attribute_means"].keys())
    means = tester.scaler["attribute_means"].reindex(feature_names)
    stds = tester.scaler["attribute_stds"].reindex(feature_names)
    train_basins = load_basin_file(cfg.train_basin_file)
    raw_attributes = load_camels_gb_attributes(cfg.data_dir)
    missing = [basin for basin in train_basins if basin not in raw_attributes.index]
    if missing:
        raise ValueError(f"Training basins missing raw attributes: {missing}")
    standardized_training = (
        raw_attributes.loc[train_basins, feature_names] - means
    ) / stds
    rng = np.random.default_rng(args.seed)
    background_count = min(args.background_size, len(standardized_training))
    background_positions = np.sort(
        rng.choice(len(standardized_training), size=background_count, replace=False)
    )
    background_basins = standardized_training.index[background_positions].astype(str).tolist()
    backgrounds = torch.tensor(
        standardized_training.iloc[background_positions].to_numpy(dtype=np.float32),
        device=device,
    )

    target = cfg.target_variables[0]
    target_scale = float(
        tester.scaler["xarray_feature_scale"][[target]].to_array().values.squeeze()
    )
    target_center = float(
        tester.scaler["xarray_feature_center"][[target]].to_array().values.squeeze()
    )

    completed = set()
    if long_path.exists():
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            existing_metadata = json.loads(metadata_path.read_text())
            resume_parameters = {
                "samples_per_basin": args.samples_per_basin,
                "background_size": background_count,
                "gradient_samples": args.gradient_samples,
                "seed": args.seed,
            }
            mismatches = {
                name: (existing_metadata.get(name), value)
                for name, value in resume_parameters.items()
                if existing_metadata.get(name) != value
            }
            if mismatches:
                details = ", ".join(
                    f"{name}: existing={old!r}, requested={new!r}"
                    for name, (old, new) in mismatches.items()
                )
                raise ValueError(
                    "Refusing to resume GradientSHAP with incompatible parameters "
                    f"({details}). Use a new --output-dir or the original parameters."
                )
        existing = pd.read_csv(long_path, usecols=["basin", "date", "feature"])
        counts = existing.groupby([existing["basin"].astype(str), "date"])["feature"].nunique()
        completed = {key for key, count in counts.items() if count == len(feature_names)}

    total = len(basins) * args.samples_per_basin
    processed = 0
    nonfinite_samples = 0
    start_time = time.monotonic()
    for basin_number, basin in enumerate(basins, start=1):
        basin = str(basin)
        dataset = tester._get_dataset(basin)
        indices = selected_indices(len(dataset), args.samples_per_basin)
        for sample_number, index in enumerate(indices, start=1):
            sample = dataset[index]
            date = pd.Timestamp(sample["date"][-1]).strftime("%Y-%m-%d")
            key = (basin, date)
            if key in completed:
                processed += 1
                continue
            embedded_dynamic = embed_dynamic(model, sample, device)
            explained_static = sample["x_s"].to(device)
            if len(explained_static) != len(feature_names):
                raise RuntimeError("Static tensor length does not match scaler feature order.")
            sample_seed = args.seed + int(basin) * 10_000 + pd.Timestamp(date).toordinal()
            shap_values, prediction, expected_value, residual = gradient_shap_values(
                model=model,
                embedded_dynamic=embedded_dynamic,
                explained_static=explained_static,
                backgrounds=backgrounds,
                gradient_samples=args.gradient_samples,
                gradient_batch_size=args.gradient_batch_size,
                seed=sample_seed,
                target_scale=target_scale,
                target_center=target_center,
            )
            finite_outputs = (
                np.isfinite(prediction)
                and np.isfinite(expected_value)
                and np.isfinite(residual)
                and np.isfinite(shap_values).all()
            )
            if not finite_outputs:
                print(
                    f"WARNING: skipping non-finite GradientSHAP sample for basin "
                    f"{basin} on {date}."
                )
                processed += 1
                nonfinite_samples += 1
                continue
            raw_values = raw_attributes.loc[basin, feature_names].to_numpy(dtype=float)
            rows = pd.DataFrame(
                {
                    "basin": basin,
                    "date": date,
                    "feature": feature_names,
                    "feature_value": raw_values,
                    "standardized_value": explained_static.detach().cpu().numpy(),
                    "shap_value": shap_values,
                    "abs_shap_value": np.abs(shap_values),
                }
            )
            append_csv(rows, long_path)
            append_csv(
                pd.DataFrame(
                    [
                        {
                            "basin": basin,
                            "date": date,
                            "prediction": prediction,
                            "conditional_expected_value": expected_value,
                            "sum_static_shap": float(shap_values.sum()),
                            "additivity_residual": residual,
                        }
                    ]
                ),
                sample_path,
            )
            processed += 1
            if processed == 1 or processed % 12 == 0 or processed == total:
                print(
                    f"[{processed}/{total}] basin {basin_number}/{len(basins)}, "
                    f"date {sample_number}/{len(indices)}; "
                    f"elapsed {(time.monotonic() - start_time) / 60:.1f} min; "
                    f"|additivity residual|={abs(residual):.5f}"
                )

    values = pd.read_csv(long_path, dtype={"basin": str})
    values = values.drop_duplicates(["basin", "date", "feature"], keep="last")
    values.to_csv(long_path, index=False)
    sample_summary = pd.read_csv(sample_path, dtype={"basin": str})
    sample_summary = sample_summary.drop_duplicates(["basin", "date"], keep="last")
    sample_summary.to_csv(sample_path, index=False)
    # Validation sequences are much longer than the requested sample count.  Reuse
    # the already-known total here instead of loading every basin dataset twice.
    expected_samples = total - nonfinite_samples
    if len(values) != expected_samples * len(feature_names):
        raise RuntimeError(
            f"Expected {expected_samples * len(feature_names)} SHAP rows, found {len(values)}."
        )
    if len(sample_summary) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} finite SHAP samples, found {len(sample_summary)}."
        )

    feature_summary = build_feature_summary(values)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
    basin_summary = (
        values.groupby(["basin", "feature"], as_index=False)
        .agg(
            mean_abs_shap=("abs_shap_value", "mean"),
            median_abs_shap=("abs_shap_value", "median"),
            mean_shap=("shap_value", "mean"),
        )
    )
    basin_summary.to_csv(output_dir / "basin_feature_summary.csv", index=False)
    plot_global_bar(feature_summary, output_dir / "gradient_shap_global_bar.png")
    plot_beeswarm(values, feature_summary, output_dir / "gradient_shap_beeswarm.png")
    _, rho, p_value = permutation_comparison(
        feature_summary, args.permutation_dir, output_dir
    )
    group_summary = group_shap_comparison(
        values, args.grouped_permutation_dir, output_dir
    )

    metadata = {
        "run_dir": str(args.run_dir),
        "checkpoint": f"model_epoch{args.epoch:03d}.pt",
        "period": args.period,
        "method": "conditional static Expected Gradients / GradientSHAP",
        "importance_unit": "specific discharge prediction, mm d-1",
        "dynamic_conditioning": "the original 365-step dynamic history is fixed per explained prediction",
        "samples_per_basin": args.samples_per_basin,
        "explained_samples": len(sample_summary),
        "skipped_nonfinite_samples": nonfinite_samples,
        "explained_basins": sorted(sample_summary["basin"].unique().tolist()),
        "background_size": background_count,
        "background_source": "training-catchment standardized static attributes",
        "background_basins": background_basins,
        "gradient_samples": args.gradient_samples,
        "gradient_batch_size": args.gradient_batch_size,
        "seed": args.seed,
        "device": device,
        "feature_order": feature_names,
        "permutation_rank_spearman_rho": rho,
        "permutation_rank_spearman_p": p_value,
        "median_absolute_additivity_residual": float(
            sample_summary["additivity_residual"].abs().median()
        ),
        "q95_absolute_additivity_residual": float(
            sample_summary["additivity_residual"].abs().quantile(0.95)
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_results(
        output_dir / "RESULTS.md",
        args,
        feature_summary,
        sample_summary,
        rho,
        p_value,
    )

    print("\nGradientSHAP analysis complete.")
    print(
        feature_summary[
            ["shap_rank", "feature", "mean_abs_shap", "median_abs_shap", "mean_shap"]
        ].head(15).to_string(index=False)
    )
    print("\nGroup summary:")
    print(group_summary.to_string(index=False))
    if rho is not None:
        print(f"\nSHAP/permutation rank Spearman rho={rho:.4f}, p={p_value:.4g}")
    print(f"\nResults saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
