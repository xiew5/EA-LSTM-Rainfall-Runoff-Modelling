from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

# Keep Matplotlib's cache inside a writable location on managed machines.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.datasetzoo.camelsgb import load_camels_gb_attributes
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "experimentNSE30tun0hs128_3006_170741"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract EA-LSTM catchment embeddings and analyse catchment similarity."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/catchment_embeddings/<run-name>.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument(
        "--query-basin",
        action="append",
        default=[],
        help="Print nearest catchments for this gauge ID; can be repeated.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path, *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve paths stored as either absolute or project-relative paths."""
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def read_split(cfg: Config, split: str) -> list[str]:
    path = cfg.as_dict().get(f"{split}_basin_file")
    if path is None:
        return []
    path = resolve_project_path(Path(path))
    return [str(basin) for basin in load_basin_file(path)]


def unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def split_membership(
    basins: list[str], split_basins: dict[str, list[str]]
) -> pd.DataFrame:
    membership = pd.DataFrame(index=pd.Index(basins, name="basin_id"))
    for split, ids in split_basins.items():
        membership[f"is_{split}"] = membership.index.isin(set(ids))

    def combine(row: pd.Series) -> str:
        labels = [split for split in split_basins if bool(row[f"is_{split}"])]
        return "+".join(labels) if labels else "unassigned"

    membership["split"] = membership.apply(combine, axis=1)
    return membership


def regional_membership(basins: list[str]) -> pd.Series:
    """Add the project's NW/SE labels when their basin lists are available."""
    region_sets: dict[str, set[str]] = {}
    for region, filename in {
        "NW": "nw_region_basin_list.txt",
        "SE": "se_region_basin_list.txt",
    }.items():
        path = PROJECT_ROOT / "basins" / filename
        if path.exists():
            region_sets[region] = set(load_basin_file(path))

    values = []
    for basin in basins:
        labels = [region for region, ids in region_sets.items() if basin in ids]
        values.append("+".join(labels) if labels else "Other")
    return pd.Series(values, index=basins, name="region")


def load_inputs(
    cfg: Config, run_dir: Path, basins: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load raw attributes and recreate NeuralHydrology's standardized x_s."""
    if cfg.dataset.lower() != "camels_gb":
        raise ValueError(
            f"This project script currently supports dataset=camels_gb, got {cfg.dataset!r}."
        )

    hydroatlas = list(cfg.hydroatlas_attributes)
    if hydroatlas:
        raise ValueError("HydroATLAS attributes are not used by this project script.")

    # BaseDataset._load_combined_attributes() sorts columns alphabetically before
    # constructing x_s. Reproducing this order is essential for valid embeddings.
    attribute_names = sorted(cfg.static_attributes)
    data_dir = resolve_project_path(Path(cfg.data_dir))
    raw = load_camels_gb_attributes(data_dir=data_dir, basins=basins)
    raw = raw.loc[basins, attribute_names].astype(float)

    scaler = load_scaler(run_dir)
    means = scaler["attribute_means"].reindex(attribute_names)
    stds = scaler["attribute_stds"].reindex(attribute_names)
    if means.isna().any() or stds.isna().any():
        missing = sorted(set(means[means.isna()].index) | set(stds[stds.isna()].index))
        raise ValueError(f"Scaler is missing static attributes: {missing}")
    if (stds == 0).any():
        zero_std = stds.index[stds == 0].tolist()
        raise ValueError(f"Static attributes have zero training standard deviation: {zero_std}")

    normalized = (raw - means) / stds
    if not np.isfinite(normalized.to_numpy()).all():
        bad_basins = normalized.index[~np.isfinite(normalized).all(axis=1)].tolist()
        raise ValueError(f"Non-finite normalized attributes for basins: {bad_basins}")
    return raw, normalized, attribute_names


def extract_embeddings(
    cfg: Config, checkpoint: Path, normalized: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return static FC output, gate logits, and sigmoid input-gate embedding."""
    if cfg.model.lower() != "ealstm":
        raise ValueError(f"Expected an EA-LSTM run, got model={cfg.model!r}.")

    model = get_model(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    x_s = torch.as_tensor(normalized.to_numpy(), dtype=torch.float32)
    with torch.inference_mode():
        static_embedding = model.embedding_net.statics_embedding(x_s)
        gate_logits = model.input_gate(static_embedding)
        gate_embedding = torch.sigmoid(gate_logits)

    return tuple(
        value.detach().cpu().numpy().astype(np.float64)
        for value in (static_embedding, gate_logits, gate_embedding)
    )


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    return np.clip(safe @ safe.T, -1.0, 1.0)


def euclidean_distance_matrix(x: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(x * x, axis=1, keepdims=True)
    squared = squared_norms + squared_norms.T - 2.0 * (x @ x.T)
    return np.sqrt(np.maximum(squared, 0.0))


def pca(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :n_components] * singular_values[:n_components]
    variances = singular_values**2
    explained = variances / variances.sum() if variances.sum() else np.zeros_like(variances)
    return coordinates, explained[:n_components]


def kmeans(
    x: np.ndarray, n_clusters: int, *, seed: int = 42, n_init: int = 20, max_iter: int = 300
) -> tuple[np.ndarray, float]:
    """Small NumPy k-means implementation to avoid an extra sklearn dependency."""
    if not 1 <= n_clusters <= len(x):
        raise ValueError(f"clusters must be between 1 and {len(x)}")

    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_inertia = np.inf

    for _ in range(n_init):
        # k-means++ initialization.
        centers = [x[rng.integers(len(x))]]
        closest_sq = np.sum((x - centers[0]) ** 2, axis=1)
        for _cluster in range(1, n_clusters):
            total = closest_sq.sum()
            probabilities = closest_sq / total if total > 0 else None
            idx = rng.choice(len(x), p=probabilities)
            centers.append(x[idx])
            closest_sq = np.minimum(closest_sq, np.sum((x - x[idx]) ** 2, axis=1))
        centers_array = np.asarray(centers)

        labels = np.zeros(len(x), dtype=int)
        for _iteration in range(max_iter):
            distances_sq = np.sum((x[:, None, :] - centers_array[None, :, :]) ** 2, axis=2)
            new_labels = distances_sq.argmin(axis=1)
            if np.array_equal(labels, new_labels) and _iteration > 0:
                break
            labels = new_labels
            for cluster in range(n_clusters):
                members = x[labels == cluster]
                if len(members):
                    centers_array[cluster] = members.mean(axis=0)
                else:
                    centers_array[cluster] = x[rng.integers(len(x))]

        inertia = np.sum((x - centers_array[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = float(inertia)
            best_labels = labels.copy()

    assert best_labels is not None
    return best_labels, best_inertia


def nearest_neighbours(
    distances: np.ndarray,
    basins: list[str],
    metadata: pd.DataFrame,
    top_k: int,
    candidate_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create a tidy nearest-neighbour table ranked by gate Euclidean distance."""
    candidate_mask = (
        np.ones(len(basins), dtype=bool) if candidate_mask is None else candidate_mask.astype(bool)
    )
    rows: list[dict] = []
    for query_idx, query in enumerate(basins):
        candidates = np.flatnonzero(candidate_mask)
        candidates = candidates[candidates != query_idx]
        ranked = candidates[np.argsort(distances[query_idx, candidates])[:top_k]]
        for rank, neighbour_idx in enumerate(ranked, start=1):
            neighbour = basins[neighbour_idx]
            rows.append(
                {
                    "query_basin": query,
                    "query_split": metadata.at[query, "split"],
                    "query_region": metadata.at[query, "region"],
                    "rank": rank,
                    "neighbour_basin": neighbour,
                    "neighbour_split": metadata.at[neighbour, "split"],
                    "neighbour_region": metadata.at[neighbour, "region"],
                    "gate_euclidean_distance": distances[query_idx, neighbour_idx],
                }
            )
    return pd.DataFrame(rows)


def plot_pca(
    coordinates: pd.DataFrame,
    explained: np.ndarray,
    color_column: str,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    categories = sorted(coordinates[color_column].astype(str).unique())
    cmap = plt.get_cmap("tab10")
    for idx, category in enumerate(categories):
        subset = coordinates[coordinates[color_column].astype(str) == category]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=38,
            alpha=0.82,
            color=cmap(idx % 10),
            label=category,
            edgecolors="white",
            linewidths=0.35,
        )
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title(f"EA-LSTM catchment gate embeddings by {color_column}")
    ax.legend(title=color_column, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_heatmap(
    cosine: np.ndarray,
    basins: list[str],
    labels: np.ndarray,
    pc1: np.ndarray,
    output_file: Path,
) -> None:
    order = np.lexsort((pc1, labels))
    ordered = cosine[np.ix_(order, order)]
    off_diagonal = ordered[~np.eye(len(ordered), dtype=bool)]
    vmin = float(np.quantile(off_diagonal, 0.02)) if len(off_diagonal) else 0.0

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(ordered, cmap="viridis", vmin=vmin, vmax=1.0, aspect="auto")
    boundaries = np.flatnonzero(np.diff(labels[order])) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=0.5, alpha=0.8)
        ax.axvline(boundary, color="white", linewidth=0.5, alpha=0.8)
    ax.set_title("Gate cosine similarity (ordered by cluster and PC1)")
    ax.set_xlabel("Catchments")
    ax.set_ylabel("Catchments")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pairwise_distance_correlation(
    gate_distance: np.ndarray, normalized_attributes: pd.DataFrame
) -> float:
    static_distance = euclidean_distance_matrix(normalized_attributes.to_numpy())
    upper = np.triu_indices_from(gate_distance, k=1)
    gate_ranks = pd.Series(gate_distance[upper]).rank(method="average")
    static_ranks = pd.Series(static_distance[upper]).rank(method="average")
    return float(gate_ranks.corr(static_ranks, method="pearson"))


def write_summary(
    output_file: Path,
    run_dir: Path,
    epoch: int,
    cfg: Config,
    n_basins: int,
    explained: np.ndarray,
    distance_correlation: float,
    cluster_counts: pd.Series,
) -> None:
    lines = [
        "# Catchment embedding analysis",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Checkpoint: `model_epoch{epoch:03d}.pt`",
        f"- Model / hidden size: `{cfg.model}` / `{cfg.hidden_size}`",
        f"- Catchments: {n_basins}",
        f"- Static inputs: {len(cfg.static_attributes)}",
        (
            "- Similarity representation: "
            f"{cfg.hidden_size}-dimensional sigmoid EA-LSTM input gate"
        ),
        "- Neighbour ranking: Euclidean distance in input-gate space",
        f"- PCA variance (PC1 / PC2): {explained[0]:.4f} / {explained[1]:.4f}",
        (
            "- Spearman correlation between pairwise gate distance and standardized "
            f"static-attribute distance: {distance_correlation:.4f}"
        ),
        "",
        "## Cluster sizes",
        "",
    ]
    lines.extend(f"- Cluster {int(cluster)}: {int(count)}" for cluster, count in cluster_counts.items())
    lines.extend(
        [
            "",
            "Embeddings from different trained runs are not directly comparable. Run this analysis "
            "separately for each checkpoint and compare within-run geometry or aggregate diagnostics.",
            "",
        ]
    )
    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = resolve_project_path(args.run_dir)
    config_file = run_dir / "config.yml"
    checkpoint = run_dir / f"model_epoch{args.epoch:03d}.pt"
    if not config_file.exists():
        raise FileNotFoundError(config_file)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args.top_k < 1:
        raise ValueError("top-k must be positive")

    cfg = Config(config_file)
    split_basins = {split: read_split(cfg, split) for split in ("train", "validation", "test")}
    basins = unique_in_order(
        basin for split in ("train", "validation", "test") for basin in split_basins[split]
    )
    if not basins:
        raise ValueError("No catchments found in the run's basin files.")

    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" / "catchment_embeddings" / run_dir.name
    )
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, normalized, attribute_names = load_inputs(cfg, run_dir, basins)
    static_embedding, gate_logits, gate_embedding = extract_embeddings(
        cfg, checkpoint, normalized
    )

    metadata = split_membership(basins, split_basins)
    metadata["region"] = regional_membership(basins)

    coordinates_array, explained = pca(gate_embedding, n_components=2)
    labels, inertia = kmeans(gate_embedding, args.clusters)
    metadata["cluster"] = labels
    coordinates = metadata.copy()
    coordinates.insert(0, "PC2", coordinates_array[:, 1])
    coordinates.insert(0, "PC1", coordinates_array[:, 0])

    cosine = cosine_similarity_matrix(gate_embedding)
    euclidean = euclidean_distance_matrix(gate_embedding)
    distance_correlation = pairwise_distance_correlation(euclidean, normalized)

    static_columns = [f"static_embedding_{idx:03d}" for idx in range(static_embedding.shape[1])]
    gate_logit_columns = [f"gate_logit_{idx:03d}" for idx in range(gate_logits.shape[1])]
    gate_columns = [f"gate_{idx:03d}" for idx in range(gate_embedding.shape[1])]
    pd.DataFrame(static_embedding, index=metadata.index, columns=static_columns).join(metadata).to_csv(
        output_dir / "static_embeddings.csv"
    )
    pd.DataFrame(gate_logits, index=metadata.index, columns=gate_logit_columns).join(metadata).to_csv(
        output_dir / "gate_logits.csv"
    )
    pd.DataFrame(gate_embedding, index=metadata.index, columns=gate_columns).join(metadata).to_csv(
        output_dir / "catchment_gate_embeddings.csv"
    )
    coordinates.to_csv(output_dir / "pca_coordinates.csv")
    pd.DataFrame(cosine, index=basins, columns=basins).to_csv(
        output_dir / "gate_cosine_similarity.csv", index_label="basin_id"
    )
    pd.DataFrame(euclidean, index=basins, columns=basins).to_csv(
        output_dir / "gate_euclidean_distance.csv", index_label="basin_id"
    )

    all_neighbours = nearest_neighbours(euclidean, basins, metadata, args.top_k)
    all_neighbours.to_csv(output_dir / "nearest_neighbours_all.csv", index=False)
    train_mask = metadata["is_train"].to_numpy(dtype=bool)
    if train_mask.any():
        training_neighbours = nearest_neighbours(
            euclidean, basins, metadata, args.top_k, candidate_mask=train_mask
        )
        training_neighbours.to_csv(
            output_dir / "nearest_training_neighbours.csv", index=False
        )

    cluster_frame = raw.copy()
    cluster_frame.insert(0, "cluster", labels)
    cluster_means = cluster_frame.groupby("cluster")[attribute_names].mean()
    cluster_means.insert(0, "count", pd.Series(labels).value_counts().sort_index())
    cluster_means.to_csv(output_dir / "cluster_attribute_means.csv")

    plot_pca(coordinates, explained, "cluster", output_dir / "pca_by_cluster.png")
    plot_pca(coordinates, explained, "split", output_dir / "pca_by_split.png")
    plot_pca(coordinates, explained, "region", output_dir / "pca_by_region.png")
    plot_similarity_heatmap(
        cosine, basins, labels, coordinates_array[:, 0], output_dir / "similarity_heatmap.png"
    )

    run_metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "epoch": args.epoch,
        "model": cfg.model,
        "hidden_size": cfg.hidden_size,
        "number_of_catchments": len(basins),
        "static_attributes_in_model_order": attribute_names,
        "static_embedding_dimension": int(static_embedding.shape[1]),
        "gate_embedding_dimension": int(gate_embedding.shape[1]),
        "pca_explained_variance_ratio": explained.tolist(),
        "kmeans_clusters": args.clusters,
        "kmeans_inertia": inertia,
        "pairwise_gate_vs_static_distance_spearman": distance_correlation,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    write_summary(
        output_dir / "README.md",
        run_dir,
        args.epoch,
        cfg,
        len(basins),
        explained,
        distance_correlation,
        pd.Series(labels).value_counts().sort_index(),
    )

    print(f"Extracted {len(basins)} catchments from {run_dir.name}.")
    print(f"Gate embedding shape: {gate_embedding.shape}")
    print(f"Outputs written to: {output_dir}")
    print(f"PC1 + PC2 explained variance: {explained.sum():.3%}")
    print(f"Gate/static pairwise distance Spearman: {distance_correlation:.4f}")

    if args.query_basin:
        query_table = all_neighbours.set_index("query_basin")
        for basin in args.query_basin:
            basin = str(basin)
            if basin not in metadata.index:
                print(f"\nQuery basin {basin} was not present in this run.")
                continue
            print(f"\nNearest catchments to {basin}:")
            columns = ["rank", "neighbour_basin", "neighbour_split", "gate_euclidean_distance"]
            print(query_table.loc[[basin], columns].to_string(index=False))


if __name__ == "__main__":
    main()
