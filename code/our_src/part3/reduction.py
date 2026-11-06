"""PCA on image-level encodings: validity checks, fitting and explained-variance curves.

This PCA compresses the final image vector (after encoding + normalization). It is not the
descriptor-level PCA of ``src.train`` (the ``C80`` in ``vlad_K32_C80``), which projects each SIFT
descriptor before the codebook is learned.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .config import PCAConfig
from .timing import Timer


def pca_invalid_reason(pca_dim: int, original_dim: int, n_train: int) -> str | None:
    """Return why ``pca_dim`` cannot be used, or ``None`` when it is valid.

    Constraints: ``pca_dim < original_dim`` and ``pca_dim <= n_train - 1`` (sklearn PCA can
    estimate at most ``min(n_samples, n_features)`` components and, after centring, only
    ``n_samples - 1`` of them carry variance).
    """
    if pca_dim <= 0:
        return f"pca_dim={pca_dim} must be positive"
    if pca_dim >= original_dim:
        return f"pca_dim={pca_dim} must be < original_dim={original_dim}"
    if pca_dim > n_train - 1:
        return f"pca_dim={pca_dim} must be <= n_train_samples - 1 = {n_train - 1}"
    return None


def _solver(cfg: PCAConfig, n_samples: int, n_features: int) -> str:
    if cfg.svd_solver != "auto":
        return cfg.svd_solver
    return "full" if min(n_samples, n_features) <= cfg.full_max_size else "randomized"


class PCAReducer:
    """Thin wrapper around sklearn PCA that records the fit time.

    ``svd_solver="full"`` is deterministic; with ``randomized`` the seed is passed as
    ``random_state`` so results are reproducible.
    """

    def __init__(self, n_components: int, cfg: PCAConfig, seed: int) -> None:
        self.n_components = n_components
        self.cfg = cfg
        self.seed = seed
        self.pca: PCA | None = None
        self.fit_seconds = 0.0

    def fit(self, x_train: np.ndarray) -> "PCAReducer":
        reason = pca_invalid_reason(self.n_components, x_train.shape[1], x_train.shape[0])
        if reason:
            raise ValueError(reason)
        solver = _solver(self.cfg, *x_train.shape)
        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.cfg.whiten,
            svd_solver=solver,
            random_state=None if solver == "full" else self.seed,
        )
        with Timer() as t:
            self.pca.fit(x_train)
        self.fit_seconds = t.elapsed
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.pca is not None, "fit() must be called first"
        return self.pca.transform(x).astype(np.float32, copy=False)


def cumulative_explained_variance(x_train: np.ndarray) -> np.ndarray:
    """Cumulative explained-variance ratio of a full PCA fitted on ``x_train``."""
    pca = PCA(n_components=min(x_train.shape), svd_solver="full")
    pca.fit(x_train)
    return np.cumsum(pca.explained_variance_ratio_)


def components_for_variance(cumulative: np.ndarray, threshold: float) -> int:
    """Smallest number of components whose cumulative explained variance reaches ``threshold``."""
    idx = int(np.searchsorted(cumulative, threshold - 1e-12, side="left"))
    return min(idx + 1, len(cumulative))
