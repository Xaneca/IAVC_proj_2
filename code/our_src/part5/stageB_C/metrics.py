"""Retrieval metrics.

Every function takes ``ranked_relevance``: a binary array with the relevance of the corpus images
in ranked order, best first.
"""
from __future__ import annotations

import numpy as np


def average_precision(ranked_relevance: np.ndarray, n_relevant: int) -> float:
    """Non-interpolated AP over the full ranking, normalised by the total number of relevant images."""
    relevant = np.asarray(ranked_relevance, dtype=bool)
    if n_relevant == 0:
        return 0.0
    hits = np.cumsum(relevant)
    ranks = np.arange(1, relevant.size + 1)
    return float((hits[relevant] / ranks[relevant]).sum() / n_relevant)


def precision_at(ranked_relevance: np.ndarray, k: int) -> float:
    """Fraction of relevant images among the top ``k``. The denominator is always ``k``."""
    return float(np.asarray(ranked_relevance, dtype=bool)[:k].sum() / k)


def recall_at(ranked_relevance: np.ndarray, k: int, n_relevant: int) -> float:
    """Fraction of all relevant images found among the top ``k``."""
    if n_relevant == 0:
        return 0.0
    return float(np.asarray(ranked_relevance, dtype=bool)[:k].sum() / n_relevant)


def evaluate_ranking(ranked_relevance: np.ndarray, n_relevant: int, precision_ks: list[int],
                     recall_ks: list[int]) -> dict[str, float]:
    """All per-query metrics, keyed ``ap``, ``p_at_K`` and ``r_at_K``."""
    out = {"ap": average_precision(ranked_relevance, n_relevant)}
    out.update({f"p_at_{k}": precision_at(ranked_relevance, k) for k in precision_ks})
    out.update({f"r_at_{k}": recall_at(ranked_relevance, k, n_relevant) for k in recall_ks})
    return out


def interpolated_pr_curve(ranked_relevance: np.ndarray, n_relevant: int, recall_grid: np.ndarray) -> np.ndarray:
    """Interpolated precision (the best precision at any recall >= r) on a fixed recall grid."""
    relevant = np.asarray(ranked_relevance, dtype=bool)
    hits = np.cumsum(relevant)
    precision = hits / np.arange(1, relevant.size + 1)
    recall = hits / max(n_relevant, 1)
    # Running maximum from the right gives the best precision at this recall or beyond.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    positions = np.searchsorted(recall, recall_grid, side="left")
    curve = np.zeros_like(recall_grid, dtype=float)
    valid = positions < relevant.size
    curve[valid] = envelope[positions[valid]]
    return curve
