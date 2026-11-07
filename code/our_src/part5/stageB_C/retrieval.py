"""Ranking of the corpus. The only decision a method makes is the order of the corpus."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .timing import Timer


def cosine_scores(corpus: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity, assuming L2-normalised vectors (so it equals the dot product)."""
    return corpus @ query


def rank_scores(scores: np.ndarray) -> np.ndarray:
    """Corpus indices by descending score. The stable sort makes ties deterministic."""
    return np.argsort(-scores, kind="stable")


def rank_queries(score_fn: Callable[[int], np.ndarray], n_queries: int,
                 repeats: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank the corpus for each query.

    Args:
        score_fn: Maps a query index to its scores over the corpus. It must be deterministic.
        n_queries: Number of queries.
        repeats: Times the scoring and sorting is timed per query, then averaged.

    Returns:
        ``orders`` (n_queries, n_corpus), ``scores`` (n_queries, n_corpus) and the mean seconds
        per query for similarity + sorting, shape (n_queries,).
    """
    orders, all_scores = [], []
    seconds = np.zeros(n_queries)
    for q in range(n_queries):
        for _ in range(repeats):
            with Timer() as timer:
                scores = score_fn(q)
                order = rank_scores(scores)
            seconds[q] += timer.elapsed
        orders.append(order)
        all_scores.append(scores)
    return np.stack(orders), np.stack(all_scores), seconds / repeats


def random_score_fn(seed: int, n_corpus: int) -> Callable[[int], np.ndarray]:
    """Random baseline: scores that depend only on the seed and the query index."""
    return lambda q: np.random.default_rng([seed, q]).random(n_corpus)
