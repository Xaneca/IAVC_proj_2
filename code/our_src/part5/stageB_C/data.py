"""Loading of the Stage A CSVs and construction of the ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd


def resolve_image_path(raw_path: str, csv_dir: Path) -> Path:
    """Resolve a ``path`` cell of a Stage A CSV.

    The CSVs store paths relative to their own directory, with the separators of the OS that
    wrote them (Windows here), so backslashes are normalised before resolving.
    """
    parts = PureWindowsPath(str(raw_path).strip()).parts
    path = Path(*parts)
    resolved = path if path.is_absolute() else csv_dir / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {raw_path!r} (resolved to {resolved})")
    return resolved


def _parse_bool(values: pd.Series) -> np.ndarray:
    """Parse ``is_relevant``, accepting booleans or 'True'/'False' strings."""
    if values.dtype == bool:
        return values.to_numpy()
    text = values.astype(str).str.strip().str.lower()
    unknown = set(text) - {"true", "false"}
    if unknown:
        raise ValueError(f"Unexpected is_relevant values: {sorted(unknown)}")
    return (text == "true").to_numpy()


def _load_images(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicated ids in {csv_path}")
    frame["image_path"] = [resolve_image_path(p, csv_path.parent) for p in frame["path"]]
    return frame


@dataclass
class Dataset:
    """Corpus, queries and category-level ground truth shared by every experiment."""

    corpus: pd.DataFrame
    queries: pd.DataFrame
    text_queries: pd.DataFrame

    @property
    def relevance(self) -> np.ndarray:
        """Boolean mask over the corpus. It applies to every query, image or text."""
        return _parse_bool(self.corpus["is_relevant"])

    @property
    def n_relevant(self) -> int:
        return int(self.relevance.sum())

    def query_ids(self, query_type: str) -> list[str]:
        frame = self.text_queries if query_type == "text" else self.queries
        return frame["id"].tolist()


def load_dataset(corpus_csv: Path, queries_csv: Path, text_queries_csv: Path) -> Dataset:
    """Read and validate the three Stage A CSVs."""
    corpus = _load_images(corpus_csv)
    queries = _load_images(queries_csv)
    text_queries = pd.read_csv(text_queries_csv)

    overlap = {Path(p).name for p in corpus["path"].str.replace("\\", "/")} & \
              {Path(p).name for p in queries["path"].str.replace("\\", "/")}
    if overlap:
        raise ValueError(f"Query files also present in the corpus: {sorted(overlap)}")
    dataset = Dataset(corpus, queries, text_queries)
    if dataset.n_relevant == 0:
        raise ValueError("The corpus has no relevant images")
    return dataset


def subset_dataset(dataset: Dataset, corpus_per_label: int, n_queries: int, n_text_queries: int) -> Dataset:
    """Deterministic small subset for smoke tests: the first images of each label, in corpus order."""
    corpus = dataset.corpus.groupby("label", sort=False).head(corpus_per_label).reset_index(drop=True)
    return Dataset(corpus, dataset.queries.head(n_queries).reset_index(drop=True),
                   dataset.text_queries.head(n_text_queries).reset_index(drop=True))
