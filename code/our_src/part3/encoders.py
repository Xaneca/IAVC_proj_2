"""BoVW / VLAD / Fisher encoders behind a common interface.

Codebooks are fitted here (seeded MiniBatchKMeans / KMeans / diagonal GMM). Encoding reuses the
encoder classes of ``src.encoders`` unchanged, by building the same configuration dictionaries that
``src.train`` would save (``words`` for BoVW/VLAD, ``means``/``covariances``/``priors`` for Fisher,
``pca_dim=0``: no descriptor-level PCA). Encodings are global (no spatial tiling).

Output dimensions with D=128 SIFT dimensions:
    bovw   K
    vlad   K * D
    fisher 2 * K * D   (+ K if ``fisher_include_weight_grads``)
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from src.encoders import BoVWEncoder, FisherVectorEncoder, VLADEncoder

from .config import CodebookConfig
from .sift_cache import load_descriptors
from .timing import Timer

DESCRIPTOR_DIM = 128


def encoding_dim(representation: str, k: int, fisher_include_weight_grads: bool = False,
                 descriptor_dim: int = DESCRIPTOR_DIM) -> int:
    """Exact dimension of the image encoding (before any PCA)."""
    if representation == "bovw":
        return k
    if representation == "vlad":
        return k * descriptor_dim
    if representation == "fisher":
        return 2 * k * descriptor_dim + (k if fisher_include_weight_grads else 0)
    raise ValueError(f"Unknown representation: {representation}")


def fingerprint(obj: Any) -> str:
    """Short stable hash of a JSON-serialisable object (used to key cache files)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class Codebook:
    representation: str
    k: int
    seed: int
    params: dict[str, np.ndarray]
    fit_seconds: float
    n_descriptors: int
    # Time to draw the descriptor subsample of this seed (shared by all codebooks of the seed).
    sampling_seconds: float = 0.0

    def src_config(self) -> dict[str, Any]:
        """Configuration dictionary in the format expected by ``src.encoders``."""
        if self.representation in ("bovw", "vlad"):
            return {"type": self.representation, "num_words": self.k, "pca_dim": 0,
                    "words": self.params["words"]}
        return {"type": "fv", "num_words": self.k, "pca_dim": 0, "means": self.params["means"],
                "covariances": self.params["covariances"], "priors": self.params["priors"]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"representation": self.representation, "k": self.k, "seed": self.seed,
                "fit_seconds": self.fit_seconds, "n_descriptors": self.n_descriptors,
                "sampling_seconds": self.sampling_seconds}
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as stream:
            np.savez(stream, meta=json.dumps(meta), **self.params)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "Codebook":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            params = {k: data[k] for k in data.files if k != "meta"}
        return cls(params=params, **meta)


def fit_codebook(representation: str, k: int, descriptors: np.ndarray, seed: int,
                 cfg: CodebookConfig) -> Codebook:
    """Fit the k-means vocabulary (BoVW, VLAD) or the diagonal GMM (Fisher). Only the fit is timed."""
    if representation in ("bovw", "vlad"):
        km = cfg.kmeans
        if km["algorithm"] == "minibatch":
            model = MiniBatchKMeans(n_clusters=k, batch_size=km.get("batch_size", 4096),
                                    n_init=km.get("n_init", 3), max_iter=km.get("max_iter", 100),
                                    random_state=seed)
        else:
            model = KMeans(n_clusters=k, n_init=km.get("n_init", 1), max_iter=km.get("max_iter", 300),
                           random_state=seed)
        with Timer() as t, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # MKL memory-leak notice on Windows
            model.fit(descriptors)
        params = {"words": model.cluster_centers_.astype(np.float32)}
    elif representation == "fisher":
        g = cfg.gmm
        model = GaussianMixture(n_components=k, covariance_type="diag", init_params=g.get("init_params", "kmeans"),
                                max_iter=g.get("max_iter", 100), n_init=g.get("n_init", 1), tol=g.get("tol", 1e-3),
                                reg_covar=g.get("reg_covar", 1e-6), random_state=seed)
        with Timer() as t, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(descriptors.astype(np.float64))
        params = {"means": model.means_, "covariances": model.covariances_, "priors": model.weights_}
    else:
        raise ValueError(f"Unknown representation: {representation}")
    return Codebook(representation, k, seed, params, t.elapsed, int(descriptors.shape[0]))


class GlobalEncoder:
    """Common interface: ``encode(descriptors (N, 128)) -> float32 vector of length dim``."""

    def __init__(self, codebook: Codebook, fisher_include_weight_grads: bool = False) -> None:
        self.representation = codebook.representation
        self.k = codebook.k
        self.include_weight_grads = fisher_include_weight_grads
        self.dim = encoding_dim(self.representation, self.k, fisher_include_weight_grads)
        cfg = codebook.src_config()
        if self.representation == "bovw":
            self._encoder = BoVWEncoder(cfg)
        elif self.representation == "vlad":
            self._encoder = VLADEncoder(cfg)
        else:
            self._encoder = FisherVectorEncoder(cfg)

    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors.shape[0] == 0:
            return np.zeros(self.dim, dtype=np.float32)
        out = np.asarray(self._encoder.encode(descriptors), dtype=np.float32).ravel()
        if self.representation == "fisher" and not self.include_weight_grads:
            # skimage layout: [d_pi (K), d_mu (K*D), d_sigma (K*D)] -> drop d_pi.
            out = out[self.k:]
        assert out.shape[0] == self.dim, (self.representation, out.shape, self.dim)
        return out


def encode_chunk(paths: list[str], indices: list[int], codebooks: list[Codebook],
                 fisher_include_weight_grads: bool, max_descriptors: int | None, seed: int
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Encode a chunk of images with every codebook (run inside a joblib worker).

    Each image's descriptors are loaded once and reused by all encoders. Returns, per codebook,
    the encodings ``(n_images, dim)`` and the per-image encoding times in seconds (loading the
    cached descriptors is not included).
    """
    encoders = [GlobalEncoder(cb, fisher_include_weight_grads) for cb in codebooks]
    feats = [np.zeros((len(paths), e.dim), dtype=np.float32) for e in encoders]
    times = [np.zeros(len(paths)) for _ in encoders]
    for row, (path, idx) in enumerate(zip(paths, indices)):
        desc = load_descriptors(path)
        if max_descriptors is not None and desc.shape[0] > max_descriptors:
            rng = np.random.default_rng([seed, idx, 1])
            desc = desc[rng.choice(desc.shape[0], max_descriptors, replace=False)]
        desc = desc.astype(np.float32)
        for e, enc in enumerate(encoders):
            with Timer() as t:
                feats[e][row] = enc.encode(desc)
            times[e][row] = t.elapsed
    return list(zip(feats, times))
