"""CSV outputs. Rows are replaced per experiment, so partial runs keep the other experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUMMARY_FILE = "experiments_summary.csv"
PER_QUERY_FILE = "per_query_results.csv"
RANKINGS_DIR = "rankings"


def _ordered(frame: pd.DataFrame, experiment_order: list[str]) -> pd.DataFrame:
    """Sort rows by config order; experiments no longer in the config go last."""
    position = {name: i for i, name in enumerate(experiment_order)}
    keys = frame["experiment"].map(lambda name: position.get(name, len(position)))
    return frame.iloc[np.argsort(keys.to_numpy(), kind="stable")]


def upsert_experiment(path: Path, rows: list[dict[str, Any]], experiment_order: list[str]) -> None:
    """Replace the rows of the experiment(s) in ``rows`` inside the CSV at ``path``."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False, na_values=[""])
        old = old[~old["experiment"].isin(set(new["experiment"]))]
        if not old.empty:
            new = pd.concat([old, new], ignore_index=True)
    _ordered(new, experiment_order).to_csv(path, index=False)


def save_ranking(output_dir: Path, experiment: str, query_ids: list[str], corpus_ids: np.ndarray,
                 labels: np.ndarray, relevance: np.ndarray, orders: np.ndarray, scores: np.ndarray) -> None:
    """Save the full ranking of every query, so figures can be regenerated without features."""
    directory = output_dir / RANKINGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, query_id in enumerate(query_ids):
        order = orders[index]
        frames.append(pd.DataFrame({
            "query_id": query_id, "rank": np.arange(1, order.size + 1), "corpus_id": corpus_ids[order],
            "score": scores[index][order], "relevant": relevance[order], "label": labels[order],
        }))
    pd.concat(frames, ignore_index=True).to_csv(directory / f"{experiment}.csv", index=False, float_format="%.6f")


def load_ranking(output_dir: Path, experiment: str) -> pd.DataFrame:
    return pd.read_csv(output_dir / RANKINGS_DIR / f"{experiment}.csv")
