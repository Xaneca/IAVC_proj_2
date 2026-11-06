"""metrics.csv / metrics_summary.csv handling and resumability."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import Experiment

KEY_COLUMNS = ["representation", "pca", "k", "pca_dim", "seed"]
GROUP_COLUMNS = ["representation", "pca", "k", "pca_dim"]

# Fixed column order; per-class columns (ap_<class>, ...) are inserted after `map`.
FIRST_COLUMNS = KEY_COLUMNS + [
    "original_dim", "final_dim", "compaction_pct",
    "n_train", "n_test",
    "test_accuracy", "macro_f1", "map",
]
PER_CLASS_PREFIXES = ("ap_", "accuracy_", "macro_f1_", "n_train_", "n_test_", "best_params_", "cv_score_")
TIME_COLUMNS = [
    "time_sift_extraction", "time_sift_extraction_per_image",
    "time_descriptor_sampling", "time_codebook",
    "time_encoding", "time_encoding_per_image",
    "time_pca",
    "time_svm_cv", "time_svm_fit", "time_svm_train",
    "time_inference_per_image",
    "feature_storage_bytes_per_image", "feature_storage_mb_total",
]


def row_key(row) -> tuple:
    return (str(row["representation"]), str(row["pca"]).lower() == "true", int(row["k"]),
            int(row["pca_dim"]), int(row["seed"]))


def experiment_key(exp: Experiment, seed: int) -> tuple:
    return (exp.representation, exp.pca, exp.k, exp.pca_dim, seed)


def load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return {row_key(r) for _, r in df.iterrows()}


def order_columns(columns: list[str]) -> list[str]:
    first = [c for c in FIRST_COLUMNS if c in columns]
    per_class = sorted(c for c in columns if c not in first and c.startswith(PER_CLASS_PREFIXES))
    times = [c for c in TIME_COLUMNS if c in columns]
    rest = [c for c in columns if c not in set(first) | set(per_class) | set(times)]
    return first + per_class + times + rest


def upsert_row(path: Path, row: dict) -> None:
    """Insert ``row`` into metrics.csv, replacing an existing row with the same key (--force)."""
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        keep = [row_key(r) != row_key(row) for _, r in old.iterrows()]
        df = pd.concat([old[keep], new], ignore_index=True)
    else:
        df = new
    df["pca"] = df["pca"].astype(str).str.lower() == "true"
    df = df.sort_values(KEY_COLUMNS, key=_sort_key).reset_index(drop=True)
    df = df[order_columns(list(df.columns))]
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _sort_key(col: pd.Series) -> pd.Series:
    if col.name == "representation":
        return col.map({"bovw": 0, "vlad": 1, "fisher": 2}).fillna(9)
    return col


def write_summary(metrics_path: Path, summary_path: Path) -> pd.DataFrame | None:
    """Mean and sample std (ddof=1) over seeds for every numeric column."""
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    numeric = [c for c in df.columns if c not in KEY_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(GROUP_COLUMNS, sort=False)
    summary = grouped[numeric].agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    summary.insert(0, "n_seeds", grouped.size())
    summary = summary.reset_index()
    summary.to_csv(summary_path, index=False)
    return summary
