"""Dense SIFT from ``src.features`` with an on-disk cache shared by all BoVW and VLAD variants.

``src.features.computeFeatures`` hard-codes its parameters, so the same two steps it performs
(``standarizeImage`` + ``vl_phow``) are called here with the values from the config.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.features import standarizeImage, vl_phow

from .timing import Timer

TIMINGS_FILE = "timings.json"


def extract_dense_sift(image_path: Path, params: dict[str, Any]) -> np.ndarray:
    """Dense SIFT descriptors of one image, shape (n_descriptors, 128), uint8."""
    image = standarizeImage(str(image_path), max_size=params["max_size"])
    _, descriptors = vl_phow(image, step=params["step"], sizes=tuple(params["sizes"]),
                             magnification=params["magnification"], window_size=params["window_size"],
                             contrast_threshold=params["contrast_threshold"], fast=params["fast"])
    return descriptors.astype(np.uint8, copy=False)


class SiftStore:
    """Descriptors of every image, computed once and kept on disk and in memory.

    Extraction runs sequentially so the recorded per-image times are not inflated by contention.
    The times are stored next to the cache, so runs reusing it still report them.
    """

    def __init__(self, directory: Path, params: dict[str, Any], logger: logging.Logger) -> None:
        self.directory = directory
        self.params = params
        self.logger = logger
        self.times: dict[str, float] = {}
        self._descriptors: dict[str, np.ndarray] = {}

    def _timings_path(self) -> Path:
        return self.directory / TIMINGS_FILE

    def ensure(self, images: dict[str, Path], force: bool = False) -> None:
        """Load the descriptors of ``images`` (id -> path), extracting the ones not cached."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._timings_path()
        self.times = json.loads(path.read_text()) if path.exists() else {}
        n_extracted = 0
        for image_id, image_path in images.items():
            if image_id in self._descriptors:
                continue
            cached = self.directory / f"{image_id}.npz"
            if cached.exists() and image_id in self.times and not force:
                with np.load(cached) as data:
                    self._descriptors[image_id] = data["descriptors"]
                continue
            with Timer() as timer:
                descriptors = extract_dense_sift(image_path, self.params)
            tmp = cached.with_suffix(".tmp")
            with open(tmp, "wb") as stream:
                np.savez_compressed(stream, descriptors=descriptors)
            os.replace(tmp, cached)
            self._descriptors[image_id] = descriptors
            self.times[image_id] = timer.elapsed
            n_extracted += 1
            if n_extracted % 25 == 0:
                self.logger.info("  dense SIFT: %d images extracted", n_extracted)
        if n_extracted:
            path.write_text(json.dumps(self.times))
        self.logger.info("Dense SIFT: %d of %d images extracted, others loaded from %s",
                         n_extracted, len(images), self.directory)

    def get(self, image_id: str) -> np.ndarray:
        return self._descriptors[image_id]

    def sample(self, image_ids: list[str], max_descriptors: int | None, seed: int) -> np.ndarray:
        """Seeded descriptor sample, spread evenly over the images, as float32."""
        if max_descriptors is None:
            return np.vstack([self.get(i) for i in image_ids]).astype(np.float32)
        per_image = max(1, math.ceil(max_descriptors / len(image_ids)))
        parts = []
        for index, image_id in enumerate(image_ids):
            descriptors = self.get(image_id)
            if descriptors.shape[0] > per_image:
                rng = np.random.default_rng([seed, index])
                descriptors = descriptors[rng.choice(descriptors.shape[0], per_image, replace=False)]
            parts.append(descriptors)
        pooled = np.vstack(parts)
        if pooled.shape[0] > max_descriptors:
            rng = np.random.default_rng(seed)
            pooled = pooled[rng.choice(pooled.shape[0], max_descriptors, replace=False)]
        return pooled.astype(np.float32)
