"""Dense SIFT extraction (once per image) with an on-disk cache.

Extraction reuses ``src.features.computeFeatures`` unchanged (PHOW: dense SIFT, 4 scales, step 4).
``src.features.extractFeatures`` also has a cache, but it stores uncompressed keypoints plus
descriptors (~5.3 MB/image, ~26 GB for the dataset). The encodings here are global (no spatial
tiling), so only the uint8 descriptors are stored, compressed (~3.4 MB/image, ~17 GB).

Per-image extraction times are kept in ``_timings.json`` so that resumed runs can still report
``time_sift_extraction``.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.features import computeFeatures

from .timing import Timer

TIMINGS_FILE = "_timings.json"


def cache_file(sift_dir: Path, image_id: str) -> Path:
    return Path(sift_dir) / f"{image_id}.npz"


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def extract_and_cache(image_path: str, out_path: str) -> float:
    """Extract dense SIFT for one image, store the descriptors and return the extraction time."""
    with Timer() as t:
        _, descriptors, _ = computeFeatures(str(image_path))
    _atomic_savez(Path(out_path), descriptors=descriptors.astype(np.uint8, copy=False))
    return t.elapsed


def load_descriptors(path: str | Path) -> np.ndarray:
    with np.load(path) as data:
        return data["descriptors"]


def _load_timings(sift_dir: Path) -> dict[str, float]:
    path = sift_dir / TIMINGS_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def _save_timings(sift_dir: Path, timings: dict[str, float]) -> None:
    tmp = sift_dir / (TIMINGS_FILE + ".tmp")
    tmp.write_text(json.dumps(timings))
    os.replace(tmp, sift_dir / TIMINGS_FILE)


def ensure_sift_cache(ids: list[str], image_path, sift_dir: Path, n_jobs: int,
                      logger: logging.Logger, block: int = 200) -> dict[str, float]:
    """Extract SIFT for every image in ``ids`` that is not cached yet.

    Returns the per-image extraction times (seconds) of ``ids``; images cached by an older run
    without a recorded time are absent from the dict.
    """
    sift_dir.mkdir(parents=True, exist_ok=True)
    timings = _load_timings(sift_dir)
    todo = [i for i in ids if not cache_file(sift_dir, i).exists()]
    if todo:
        logger.info("Extracting dense SIFT for %d/%d images into %s", len(todo), len(ids), sift_dir)
        with Parallel(n_jobs=n_jobs) as parallel:
            for start in range(0, len(todo), block):
                part = todo[start:start + block]
                times = parallel(delayed(extract_and_cache)(str(image_path(i)), str(cache_file(sift_dir, i)))
                                 for i in part)
                timings.update(dict(zip(part, times)))
                _save_timings(sift_dir, timings)
                logger.info("  SIFT %d/%d", min(start + block, len(todo)), len(todo))
    else:
        logger.info("Dense SIFT already cached for all %d images", len(ids))
    missing = [i for i in ids if i not in timings]
    if missing:
        logger.warning("No recorded SIFT extraction time for %d cached images", len(missing))
    return {i: timings[i] for i in ids if i in timings}


def _sample_from_images(paths: list[str], indices: list[int], per_image: int, seed: int) -> np.ndarray:
    out = []
    for path, idx in zip(paths, indices):
        desc = load_descriptors(path)
        rng = np.random.default_rng([seed, idx])
        if desc.shape[0] > per_image:
            desc = desc[rng.choice(desc.shape[0], per_image, replace=False)]
        out.append(desc)
    return np.vstack(out)


def sample_descriptors(ids: list[str], sift_dir: Path, n_total: int, seed: int, n_jobs: int,
                       chunk_size: int = 50) -> np.ndarray:
    """Seeded subsample of about ``n_total`` descriptors, spread evenly over the images ``ids``.

    Each image contributes ``ceil(n_total / len(ids))`` descriptors (or all of them if fewer);
    the pooled set is then trimmed to ``n_total`` with the same seed. Returns float32.
    """
    per_image = max(1, math.ceil(n_total / len(ids)))
    paths = [str(cache_file(sift_dir, i)) for i in ids]
    chunks = [(paths[s:s + chunk_size], list(range(s, min(s + chunk_size, len(ids)))))
              for s in range(0, len(ids), chunk_size)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_sample_from_images)(p, idx, per_image, seed)
                                    for p, idx in chunks)
    pooled = np.vstack(parts)
    if pooled.shape[0] > n_total:
        rng = np.random.default_rng(seed)
        pooled = pooled[rng.choice(pooled.shape[0], n_total, replace=False)]
    return pooled.astype(np.float32)
