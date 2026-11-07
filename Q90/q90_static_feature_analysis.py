"""
Fast static-catchment feature screening for Q90 low-flow prediction.

Purpose
-------
This script is a quick, reproducible non-LSTM analysis for checking which
static catchment attributes look useful for predicting Q90. It is intended to
guide later, slower LSTM / EA-LSTM feature-set experiments.

Run from the project root:

    python q90_static_feature_analysis.py

Optionally override paths:

    python q90_static_feature_analysis.py --data-path Q90flow_percentile.csv
    python q90_static_feature_analysis.py --output-dir outputs/q90_static_feature_analysis

Notes
-----
- MAE is treated as the main model-selection metric.
- NSE is reported because it is common in hydrology.
- Q90 is expected to be in mm/d. If the source Q90 is in cumecs, convert it
  before running: Q90_mm_d = Q90_cumecs * 86.4 / Area_km2.
- Runoff_mm and BFIHOST are deliberately excluded from the feature matrix for
  the current screening experiment. Area_km2 is retained.
- PCA components are useful for dimensionality reduction but are not physically
  interpretable in the same way as original catchment attributes.
- Top-N feature selection is estimated once on the full dataset for speed. That
  is useful for screening but is not a fully nested, leakage-free feature
  selection experiment.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Keep matplotlib cache inside the project/output-friendly temp area on systems
# where the user's home matplotlib cache is not writable.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/q90_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/q90_xdg_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "Q90flow_percentile.csv"
OUTPUT_DIR = "outputs/q90_static_feature_analysis"
TARGET_COLUMN = "Q90"
STATIC_START_COLUMN = "Area_km2"
EXCLUDED_FEATURE_COLUMNS = ("Runoff_mm", "BFIHOST")
RANDOM_STATE = 42

ID_COLUMN_CANDIDATES = {
    "gauge_id",
    "gaugeid",
    "gauge_id_nrfa",
    "station_id",
    "site_id",
    "catchmentlabel",
    "catchment_label",
    "nrfa_id",
    "id",
}

METADATA_NAME_HINTS = (
    "created",
    "date",
    "time",
    "periodofrecord",
    "period_of_record",
)


class DataFrameColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed list of raw dataframe columns inside a sklearn Pipeline."""

    def __init__(self, columns: Sequence[str]):
        # Keep the exact constructor parameter so sklearn.clone can reproduce
        # this estimator during cross-validation.
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        missing = [col for col in list(self.columns) if col not in X.columns]
        if missing:
            raise ValueError(f"Selected columns missing from X: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, list(self.columns)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast static-feature screening experiment for Q90."
    )
    parser.add_argument("--data-path", default=DATA_PATH, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory where summary CSVs and plots will be saved.",
    )
    parser.add_argument(
        "--target-column",
        default=TARGET_COLUMN,
        help="Target column name. Default: Q90.",
    )
    parser.add_argument(
        "--static-start-column",
        default=STATIC_START_COLUMN,
        help=(
            "First CSV column to treat as a static catchment attribute. "
            "Default: Area_km2, equivalent to Excel column I in Q90flow_percentile.csv."
        ),
    )
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(data_path: str, target_column: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}. Change DATA_PATH or pass --data-path."
        )

    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns are: {list(df.columns)}"
        )

    print("\nLoaded dataset")
    print(f"Path: {path}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nFirst rows:")
    print(df.head())
    return df


def detect_id_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    """Detect site IDs and obvious metadata columns that should not be features."""

    id_columns: List[str] = []
    n_rows = len(df)

    for col in df.columns:
        if col == target_column:
            continue

        lower = col.lower()
        compact = lower.replace(" ", "").replace("-", "").replace("_", "")

        if lower in ID_COLUMN_CANDIDATES or compact in ID_COLUMN_CANDIDATES:
            id_columns.append(col)
            continue

        if any(hint in compact for hint in METADATA_NAME_HINTS):
            id_columns.append(col)
            continue

        # A nearly unique object column is more likely to be a label or metadata
        # field than a reusable catchment attribute.
        if df[col].dtype == "object":
            unique_ratio = df[col].nunique(dropna=True) / max(n_rows, 1)
            if unique_ratio > 0.8:
                id_columns.append(col)

    return sorted(set(id_columns))


def report_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("\nMissing-value counts:")
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)


def prepare_xy(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    static_start_column: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separate X and y using only static attributes from the configured start column."""

    if static_start_column not in df.columns:
        raise ValueError(
            f"Static start column '{static_start_column}' was not found. "
            f"Available columns are: {list(df.columns)}"
        )

    y = pd.to_numeric(df[target_column], errors="coerce")
    valid_target = y.notna()
    if valid_target.sum() < len(y):
        print(
            f"\nDropping {(~valid_target).sum()} rows because '{target_column}' is missing or non-numeric."
        )
    df_valid = df.loc[valid_target].copy()
    y = y.loc[valid_target].astype(float)

    start_idx = df_valid.columns.get_loc(static_start_column)
    static_df = df_valid.iloc[:, start_idx:].copy()
    if target_column not in static_df.columns:
        raise ValueError(
            f"Target column '{target_column}' must be at or after static start column "
            f"'{static_start_column}'."
        )

    id_columns = detect_id_columns(static_df, target_column)
    empty_columns = [
        col for col in static_df.columns if col != target_column and static_df[col].isna().all()
    ]
    excluded_columns = [col for col in EXCLUDED_FEATURE_COLUMNS if col in static_df.columns]
    drop_columns = sorted(set(id_columns + empty_columns + excluded_columns + [target_column]))
    X = static_df.drop(columns=[col for col in drop_columns if col in static_df.columns])

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after dropping IDs/metadata.")

    report_missing_values(static_df)

    print("\nFeature preparation")
    print(f"Target column: {target_column}")
    print(
        f"Static feature columns start at: {static_start_column} "
        f"(CSV position {start_idx + 1}; Excel column I in the current file)"
    )
    print(
        "Columns before the static start column were excluded: "
        f"{df_valid.columns[:start_idx].tolist()}"
    )
    print(f"Dropped ID/metadata columns: {id_columns if id_columns else 'None'}")
    print(f"Dropped completely empty columns: {empty_columns if empty_columns else 'None'}")
    print(f"Dropped requested feature columns: {excluded_columns if excluded_columns else 'None'}")
    print(f"Rows with valid target: {len(y)}")
    print(f"Usable raw feature columns: {X.shape[1]}")

    return X, y, id_columns


def split_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def make_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_feature_types(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_models() -> Dict[str, object]:
    """Fast non-LSTM regressors for screening."""

    return {
        "DummyMedian": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(
            alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            l2_regularization=0.01,
        ),
    }


def make_cv(n_samples: int) -> KFold:
    n_splits = 5 if n_samples >= 10 else max(2, min(n_samples, 5))
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def nse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denominator


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true_array, y_pred_array),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "R2": r2_score(y_true_array, y_pred_array),
        "NSE": nse_score(y_true_array, y_pred_array),
    }


def build_pipeline(
    X: pd.DataFrame,
    model: object,
    selected_columns: Optional[Sequence[str]] = None,
    use_pca: bool = False,
    pca_variance: Optional[float] = None,
) -> Pipeline:
    steps = []

    X_for_preprocessor = X
    if selected_columns is not None:
        steps.append(("select_raw_columns", DataFrameColumnSelector(selected_columns)))
        X_for_preprocessor = X.loc[:, list(selected_columns)]

    if use_pca:
        steps.extend(
            [
                ("preprocess", make_preprocessor(X_for_preprocessor, scale_numeric=True)),
                ("pca", PCA(n_components=pca_variance, random_state=RANDOM_STATE)),
                ("model", clone(model)),
            ]
        )
    else:
        scale_for_model = isinstance(model, (Ridge, Lasso, ElasticNet))
        steps.extend(
            [
                (
                    "preprocess",
                    make_preprocessor(X_for_preprocessor, scale_numeric=scale_for_model),
                ),
                ("model", clone(model)),
            ]
        )

    return Pipeline(steps)


def evaluate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    feature_set_name: str,
    model_name: str,
    pipeline: Pipeline,
    cv: KFold,
) -> Tuple[Dict[str, float], np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=None)

    metrics = regression_metrics(y, y_pred)
    row = {
        "feature_set": feature_set_name,
        "model": model_name,
        "n_raw_features": X.shape[1],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "NSE": metrics["NSE"],
    }
    return row, y_pred


def feature_set_columns(X: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        "All static features": X.columns.tolist(),
        "Remove SAAR": [
            col for col in X.columns if "saar" not in col.lower()
        ],
    }


def transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names: List[str] = []
        for name, _, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if isinstance(cols, slice):
                names.extend([f"{name}__col_{i}" for i in range(cols.start, cols.stop)])
            else:
                names.extend([f"{name}__{col}" for col in cols])
        return names


def raw_name_from_transformed(transformed_name: str, raw_columns: Iterable[str]) -> str:
    """Map sklearn transformed names back to raw feature names where possible."""

    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1)

    if transformed_name.startswith("cat__"):
        remainder = transformed_name.replace("cat__", "", 1)
        matches = [
            col for col in raw_columns if remainder == col or remainder.startswith(f"{col}_")
        ]
        if matches:
            return max(matches, key=len)
        return remainder.split("_")[0]

    return transformed_name


def rank_raw_features_with_random_forest(
    X: pd.DataFrame, y: pd.Series, output_dir: Path
) -> pd.DataFrame:
    """Rank original raw attributes by summed RF importance after preprocessing."""

    rf = RandomForestRegressor(
        n_estimators=700,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    preprocessor = make_preprocessor(X, scale_numeric=False)
    pipe = Pipeline([("preprocess", preprocessor), ("model", rf)])
    pipe.fit(X, y)

    names = transformed_feature_names(pipe.named_steps["preprocess"])
    importances = pipe.named_steps["model"].feature_importances_

    transformed = pd.DataFrame(
        {"transformed_feature": names, "importance": importances}
    )
    transformed["raw_feature"] = transformed["transformed_feature"].apply(
        lambda value: raw_name_from_transformed(value, X.columns)
    )

    ranked = (
        transformed.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(output_dir / "raw_feature_importance_ranking.csv", index=False)
    return ranked


def evaluate_feature_sets(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], np.ndarray]]:
    models = get_models()
    cv = make_cv(len(y))
    rows: List[Dict[str, float]] = []
    predictions: Dict[Tuple[str, str], np.ndarray] = {}

    numeric_cols, categorical_cols = split_feature_types(X)
    print("\nFeature types")
    print(f"Numeric raw features: {len(numeric_cols)}")
    print(f"Categorical raw features: {len(categorical_cols)}")

    print("\nEvaluating standard feature-removal sets...")
    for feature_set_name, cols in feature_set_columns(X).items():
        if not cols:
            print(f"Skipping '{feature_set_name}' because no columns remain.")
            continue
        for model_name, model in models.items():
            pipe = build_pipeline(X, model, selected_columns=cols)
            row, y_pred = evaluate_pipeline(
                X, y, feature_set_name, model_name, pipe, cv
            )
            row["n_raw_features"] = len(cols)
            rows.append(row)
            predictions[(feature_set_name, model_name)] = y_pred

    print("\nEvaluating PCA feature sets...")
    pca_rows: List[Dict[str, float]] = []
    for variance in (0.90, 0.95):
        feature_set_name = f"PCA {int(variance * 100)}% variance"
        for model_name, model in models.items():
            pipe = build_pipeline(
                X, model, selected_columns=None, use_pca=True, pca_variance=variance
            )
            row, y_pred = evaluate_pipeline(
                X, y, feature_set_name, model_name, pipe, cv
            )
            rows.append(row)
            predictions[(feature_set_name, model_name)] = y_pred

        # Fit once on all data only to report how many components are needed.
        pca_report_pipe = build_pipeline(
            X,
            RandomForestRegressor(random_state=RANDOM_STATE),
            use_pca=True,
            pca_variance=variance,
        )
        pca_report_pipe.fit(X, y)
        pca = pca_report_pipe.named_steps["pca"]
        pca_rows.append(
            {
                "feature_set": feature_set_name,
                "variance_threshold": variance,
                "n_components": int(pca.n_components_),
                "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
            }
        )
        print(
            f"{feature_set_name}: {pca.n_components_} components explain "
            f"{np.sum(pca.explained_variance_ratio_):.3f} variance."
        )

    pd.DataFrame(pca_rows).to_csv(
        output_dir / "pca_explained_variance.csv", index=False
    )

    print("\nRanking raw features for Top-N screening...")
    ranked = rank_raw_features_with_random_forest(X, y, output_dir)
    for top_n in (10, 20, 30):
        selected = ranked.head(min(top_n, len(ranked)))["raw_feature"].tolist()
        feature_set_name = f"Top {len(selected)} RF-ranked raw features"
        for model_name, model in models.items():
            pipe = build_pipeline(X, model, selected_columns=selected)
            row, y_pred = evaluate_pipeline(
                X, y, feature_set_name, model_name, pipe, cv
            )
            row["n_raw_features"] = len(selected)
            rows.append(row)
            predictions[(feature_set_name, model_name)] = y_pred

    results = pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    results.to_csv(output_dir / "model_performance_summary.csv", index=False)
    return results, predictions


def plot_pca_cumulative_explained_variance(
    X: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot cumulative explained variance for all PCA components.

    This is a descriptive plot fitted on the full dataset after the same
    imputation, one-hot encoding, and standardisation used by the PCA models.
    The cross-validated model-performance table remains the fair comparison.
    """

    preprocessor = make_preprocessor(X, scale_numeric=True)
    X_processed = preprocessor.fit_transform(X)
    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(X_processed)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    components = np.arange(1, len(cumulative) + 1)
    n90 = int(np.searchsorted(cumulative, 0.90) + 1)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)

    pca_curve = pd.DataFrame(
        {
            "component": components,
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )
    pca_curve.to_csv(output_dir / "pca_cumulative_explained_variance.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.plot(components, cumulative, marker="o", markersize=3, color="#2f6f8f")
    plt.axhline(0.90, color="#b23a48", linestyle="--", linewidth=1.8, label="90%")
    plt.axhline(0.95, color="#d18f00", linestyle="--", linewidth=1.8, label="95%")
    plt.axvline(n90, color="#b23a48", linestyle=":", linewidth=1.5)
    plt.axvline(n95, color="#d18f00", linestyle=":", linewidth=1.5)
    plt.scatter([n90, n95], [cumulative[n90 - 1], cumulative[n95 - 1]], color="#222222")
    plt.text(n90, 0.905, f"  {n90} PCs", color="#b23a48", va="bottom")
    plt.text(n95, 0.955, f"  {n95} PCs", color="#8a5d00", va="bottom")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance for static features")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_cumulative_explained_variance.png", dpi=200)
    plt.close()


def fit_best_model_and_importances(
    X: pd.DataFrame,
    y: pd.Series,
    best_feature_set: str,
    best_model_name: str,
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """Fit reporting models on all data and save top 10 / top 20 importances."""

    if best_feature_set.startswith("PCA"):
        print(
            "\nBest model uses PCA, so original-attribute importances are not directly interpretable."
        )
        return None

    ranked = pd.read_csv(output_dir / "raw_feature_importance_ranking.csv")
    if len(ranked) < 20:
        raise ValueError(
            f"Top 20 feature importance was requested, but only {len(ranked)} "
            "ranked raw features are available."
        )

    top10_cols = ranked.head(10)["raw_feature"].tolist()
    top20_cols = ranked.head(20)["raw_feature"].tolist()

    models = get_models()
    model = models[best_model_name]

    def fit_and_extract_importances(selected_cols: List[str]) -> pd.DataFrame:
        pipe = build_pipeline(X, model, selected_columns=selected_cols)
        pipe.fit(X, y)
        fitted_model = pipe.named_steps["model"]

        if not hasattr(fitted_model, "feature_importances_"):
            # Use RandomForest for interpretable screening importances if the
            # best fitted model itself has no native feature_importances_.
            print(
                f"\nBest fitted model '{best_model_name}' has no tree feature_importances_; "
                "fitting RandomForest for importance reporting."
            )
            pipe = build_pipeline(X, models["RandomForest"], selected_columns=selected_cols)
            pipe.fit(X, y)
            fitted_model = pipe.named_steps["model"]

        if not hasattr(fitted_model, "feature_importances_"):
            raise ValueError("No feature importances available after fallback fitting.")

        names = transformed_feature_names(pipe.named_steps["preprocess"])
        importances = fitted_model.feature_importances_
        transformed = pd.DataFrame(
            {"transformed_feature": names, "importance": importances}
        )
        transformed["raw_feature"] = transformed["transformed_feature"].apply(
            lambda value: raw_name_from_transformed(value, selected_cols)
        )

        importances_df = (
            transformed.groupby("raw_feature", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importances_df["rank"] = np.arange(1, len(importances_df) + 1)
        return importances_df

    top10 = fit_and_extract_importances(top10_cols).head(10)
    top20 = fit_and_extract_importances(top20_cols).head(20)
    if len(top20) != 20:
        raise ValueError(
            f"Expected 20 rows in top_20_feature_importances.csv, got {len(top20)}."
        )

    top10.to_csv(output_dir / "top_10_feature_importances.csv", index=False)
    top20.to_csv(output_dir / "top_20_feature_importances.csv", index=False)

    print("\nTop feature-importance keyword checks")
    for keyword in ("SAAR",):
        matches = top20[
            top20["raw_feature"].str.contains(keyword, case=False, na=False)
        ]
        if matches.empty:
            print(f"{keyword}: no matching feature names found.")
        else:
            best_match = matches.iloc[0]
            print(
                f"{keyword}: best matching feature is '{best_match['raw_feature']}' "
                f"with rank {int(best_match['rank'])}."
            )

    return top20


def plot_mae_comparison(results: pd.DataFrame, output_dir: Path) -> None:
    best_by_set_model = results.copy()
    best_by_set_model["label"] = (
        best_by_set_model["feature_set"] + "\n" + best_by_set_model["model"]
    )
    top = best_by_set_model.sort_values("MAE").head(35).iloc[::-1]

    plt.figure(figsize=(12, max(7, 0.28 * len(top))))
    plt.barh(top["label"], top["MAE"], color="#2f6f8f")
    plt.xlabel("MAE")
    plt.title("Q90 MAE comparison: best-performing feature set / model combinations")
    plt.tight_layout()
    plt.savefig(output_dir / "mae_comparison.png", dpi=200)
    plt.close()


def plot_predictions(
    y: pd.Series,
    y_pred: np.ndarray,
    best_label: str,
    output_dir: Path,
) -> None:
    min_value = min(float(np.min(y)), float(np.min(y_pred)))
    max_value = max(float(np.max(y)), float(np.max(y_pred)))

    plt.figure(figsize=(7, 7))
    plt.scatter(y, y_pred, alpha=0.75, edgecolor="none", color="#2f6f8f")
    plt.plot([min_value, max_value], [min_value, max_value], color="#b23a48", lw=2)
    plt.xlabel("Observed Q90 (mm/d)")
    plt.ylabel("Predicted Q90 (mm/d)")
    plt.title(f"Predicted vs observed Q90 (mm/d)\n{best_label}")
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_observed_q90.png", dpi=200)
    plt.close()

    residuals = y.to_numpy() - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.75, edgecolor="none", color="#2f6f8f")
    plt.axhline(0, color="#b23a48", lw=2)
    plt.xlabel("Predicted Q90 (mm/d)")
    plt.ylabel("Residual: observed - predicted (mm/d)")
    plt.title(f"Q90 residuals (mm/d)\n{best_label}")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_q90.png", dpi=200)
    plt.close()


def plot_one_feature_importance_chart(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    plot_data = data.sort_values("importance", ascending=True)
    plt.figure(figsize=(9, max(5, 0.32 * len(plot_data))))
    plt.barh(plot_data["raw_feature"], plot_data["importance"], color="#2f6f8f")
    plt.xlabel("Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importances(top20: Optional[pd.DataFrame], output_dir: Path) -> None:
    if top20 is None or top20.empty:
        return

    top10 = top20.head(10)
    plot_one_feature_importance_chart(
        top10,
        output_dir / "top_10_feature_importances.png",
        "Top 10 raw static-feature importances",
    )
    plot_one_feature_importance_chart(
        top20,
        output_dir / "top_20_feature_importances.png",
        "Top 20 raw static-feature importances",
    )


def compare_feature_set_delta(
    results: pd.DataFrame,
    reference_set: str,
    comparison_set: str,
    tolerance_fraction: float = 0.01,
) -> str:
    """Compare best MAE per feature set against the all-feature baseline."""

    ref = results.loc[results["feature_set"] == reference_set, "MAE"]
    comp = results.loc[results["feature_set"] == comparison_set, "MAE"]
    if ref.empty or comp.empty:
        return "not available"

    ref_mae = ref.min()
    comp_mae = comp.min()
    delta = comp_mae - ref_mae
    tolerance = max(tolerance_fraction * ref_mae, 1e-12)

    if abs(delta) <= tolerance:
        return f"barely changed MAE ({comp_mae:.4g} vs {ref_mae:.4g})"
    if delta < 0:
        return f"improved MAE ({comp_mae:.4g} vs {ref_mae:.4g})"
    return f"worsened MAE ({comp_mae:.4g} vs {ref_mae:.4g})"


def recommend_feature_sets(results: pd.DataFrame) -> List[str]:
    """Pick 2-3 practical feature sets to try later in EA-LSTM."""

    non_pca = results[~results["feature_set"].str.startswith("PCA")]
    best_sets = (
        non_pca.groupby("feature_set", as_index=False)["MAE"]
        .min()
        .sort_values("MAE")
        .head(3)["feature_set"]
        .tolist()
    )
    return best_sets


def print_final_summary(results: pd.DataFrame) -> None:
    best = results.iloc[0]
    pca_best = results[results["feature_set"].str.startswith("PCA")]
    pca_note = "not competitive"
    if not pca_best.empty:
        best_pca_mae = pca_best["MAE"].min()
        best_mae = best["MAE"]
        pca_note = (
            "competitive"
            if best_pca_mae <= best_mae * 1.05
            else f"worse than best MAE ({best_pca_mae:.4g} vs {best_mae:.4g})"
        )

    print("\nFinal summary")
    print(f"Best model: {best['model']}")
    print(f"Best feature set: {best['feature_set']}")
    print(f"Best MAE: {best['MAE']:.6g}")
    print(f"Best RMSE: {best['RMSE']:.6g}")
    print(f"Best NSE: {best['NSE']:.6g}")
    print(
        "Removing SAAR: "
        + compare_feature_set_delta(results, "All static features", "Remove SAAR")
    )
    print(f"PCA 90% / 95% performance: {pca_note}")
    print("\nRecommended feature sets to consider later in EA-LSTM:")
    for feature_set in recommend_feature_sets(results):
        print(f"- {feature_set}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for deprecated_file in (
        "missing_value_counts.csv",
        "top_30_feature_importances.csv",
        "top_30_feature_importances.png",
    ):
        deprecated_path = output_dir / deprecated_file
        if deprecated_path.exists():
            deprecated_path.unlink()

    df = load_dataset(args.data_path, args.target_column)
    X, y, _ = prepare_xy(
        df,
        args.target_column,
        output_dir,
        static_start_column=args.static_start_column,
    )

    keyword_table = []
    for keyword in ("SAAR",):
        matches = [col for col in X.columns if keyword.lower() in col.lower()]
        keyword_table.append({"keyword": keyword, "matching_columns": ", ".join(matches)})
        print(f"\nColumns matching {keyword}: {matches if matches else 'None'}")
    pd.DataFrame(keyword_table).to_csv(
        output_dir / "keyword_feature_matches.csv", index=False
    )

    results, predictions = evaluate_feature_sets(X, y, output_dir)
    print("\nTop model/feature-set results by MAE:")
    print(results.head(15).to_string(index=False))

    best = results.iloc[0]
    best_key = (best["feature_set"], best["model"])
    best_label = f"{best['feature_set']} / {best['model']}"

    top20 = fit_best_model_and_importances(
        X,
        y,
        best_feature_set=best["feature_set"],
        best_model_name=best["model"],
        output_dir=output_dir,
    )

    plot_mae_comparison(results, output_dir)
    plot_pca_cumulative_explained_variance(X, output_dir)
    plot_predictions(y, predictions[best_key], best_label, output_dir)
    plot_feature_importances(top20, output_dir)

    print_final_summary(results)
    print(f"\nOutputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
