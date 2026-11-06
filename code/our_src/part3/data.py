"""Dataset loading: image IDs, multilabel categories and the train/test partition.

The split lists are the historical ones used in Part 1 (``data/splits_internal``); they are
loaded and validated through ``our_src.part1.stagea.data.manifest``, which refuses lists that
differ from the stored reference. Seeds never change the split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from our_src.part1.stagea.data import manifest

from .config import Config


@dataclass
class Dataset:
    train_ids: list[str]
    test_ids: list[str]
    labels: dict[str, set[str]]   # image_id -> set of categories
    image_dir: Path

    @property
    def all_ids(self) -> list[str]:
        return self.train_ids + self.test_ids

    def image_path(self, image_id: str) -> Path:
        return self.image_dir / f"{image_id}.jpg"

    def binary_labels(self, ids: list[str], target: str) -> np.ndarray:
        """1 when the image contains ``target`` (it may also contain other classes), else 0."""
        return np.array([int(target in self.labels[i]) for i in ids], dtype=np.int64)


def task_mask(dataset: Dataset, ids: list[str], target: str, negatives: str) -> np.ndarray:
    """Boolean mask of the images that take part in the binary problem of ``target``.

    ``negatives='all'``: every image (positives = contains target, negatives = the rest).
    ``negatives='background'``: positives plus background-only negatives (Part 1 protocol).
    """
    if negatives == "all":
        return np.ones(len(ids), dtype=bool)
    return np.array([target in dataset.labels[i] or "background" in dataset.labels[i] for i in ids])


def _smoke_subset(ids: list[str], dataset: Dataset, target: str, n_pos: int, n_neg: int,
                  negatives: str) -> list[str]:
    rng = np.random.default_rng(0)
    pos = [i for i in ids if target in dataset.labels[i]]
    neg = [i for i in ids if target not in dataset.labels[i]
           and (negatives == "all" or "background" in dataset.labels[i])]
    pick = list(rng.choice(pos, min(n_pos, len(pos)), replace=False)) + \
        list(rng.choice(neg, min(n_neg, len(neg)), replace=False))
    return sorted(str(i) for i in pick)


def load_dataset(cfg: Config) -> Dataset:
    rows = manifest(cfg.paths.split_dir)
    labels = {r["image_id"]: set(r["labels"]) for r in rows}
    train_ids = sorted(r["image_id"] for r in rows if r["split"] in cfg.data.train_splits)
    test_ids = sorted(r["image_id"] for r in rows if r["split"] in cfg.data.test_splits)
    dataset = Dataset(train_ids, test_ids, labels, cfg.paths.image_dir)

    if cfg.data.negatives == "background":
        # Encode only the images that take part in at least one binary problem.
        def keep(ids):
            masks = [task_mask(dataset, ids, t, "background") for t in cfg.data.target_classes]
            union = np.logical_or.reduce(masks)
            return [i for i, m in zip(ids, union) if m]
        dataset.train_ids, dataset.test_ids = keep(train_ids), keep(test_ids)

    if cfg.smoke:
        s = cfg.smoke_images
        target = cfg.data.target_classes[0]
        dataset.train_ids = _smoke_subset(dataset.train_ids, dataset, target, s["train_pos"],
                                          s["train_neg"], cfg.data.negatives)
        dataset.test_ids = _smoke_subset(dataset.test_ids, dataset, target, s["test_pos"],
                                         s["test_neg"], cfg.data.negatives)

    missing = [i for i in dataset.all_ids if not dataset.image_path(i).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing, e.g. {missing[:3]}")
    return dataset
