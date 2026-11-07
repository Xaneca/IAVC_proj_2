"""K-means vocabulary fitting and BoVW / VLAD encoding through ``src.encoders``.

``src.encoders.VLADEncoder`` returns a globally L2-normalised vector. That only changes the scale
of the signed square root, so ``vlad_normalize`` gives the same result as on the raw residuals.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

from src.encoders import BoVWEncoder, VLADEncoder

from .normalization import hellinger, l2, vlad_normalize
from .timing import Timer

DESCRIPTOR_DIM = 128


def fit_vocabulary(descriptors: np.ndarray, k: int, params: dict[str, Any], seed: int) -> tuple[np.ndarray, float]:
    """Fit k-means on ``descriptors``. Returns the centers (float32) and the fit time in seconds."""
    if params["algorithm"] == "minibatch":
        model = MiniBatchKMeans(n_clusters=k, batch_size=params["batch_size"], n_init=params["n_init"],
                                max_iter=params["max_iter"], random_state=seed)
    else:
        model = KMeans(n_clusters=k, n_init=params["n_init"], max_iter=params["max_iter"], random_state=seed)
    with Timer() as timer, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # MKL memory-leak notice on Windows
        model.fit(descriptors)
    return model.cluster_centers_.astype(np.float32), timer.elapsed


class ImageEncoder:
    """``encode(descriptors) -> normalized float32 vector`` for one method and vocabulary."""

    def __init__(self, method: str, words: np.ndarray) -> None:
        self.method = method
        config = {"type": method, "num_words": words.shape[0], "pca_dim": 0, "words": words}
        self._encoder = BoVWEncoder(config) if method == "bovw" else VLADEncoder(config)

    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        raw = self._encoder.encode(descriptors.astype(np.float32))
        return hellinger(raw) if self.method == "bovw" else vlad_normalize(np.asarray(raw).ravel())


class VladPCA:
    """PCA fitted on the normalized corpus VLADs, followed by L2 again. No whitening."""

    def __init__(self, n_components: int, seed: int) -> None:
        self._pca = PCA(n_components=n_components, svd_solver="full", whiten=False, random_state=seed)
        self.fit_seconds = 0.0

    def fit(self, corpus: np.ndarray) -> "VladPCA":
        with Timer() as timer:
            self._pca.fit(corpus)
        self.fit_seconds = timer.elapsed
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return l2(self._pca.transform(vectors).astype(np.float32))

    @property
    def explained_variance(self) -> float:
        return float(self._pca.explained_variance_ratio_.sum())

    @property
    def storage_bytes(self) -> int:
        """Bytes of the projection matrix and the mean, as float32."""
        return int((self._pca.components_.size + self._pca.mean_.size) * 4)
