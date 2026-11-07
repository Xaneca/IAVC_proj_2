"""Vector normalizations used by the retrieval methods. All operate along the last axis.

All-zero vectors stay all-zero instead of producing NaNs.
"""
from __future__ import annotations

import numpy as np


def l1(x: np.ndarray) -> np.ndarray:
    norm = np.abs(x).sum(axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def signed_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


def hellinger(histogram: np.ndarray) -> np.ndarray:
    """BoVW: L1 then element-wise square root. The result has unit L2 norm."""
    return np.sqrt(l1(histogram.astype(np.float32)))


def vlad_normalize(vector: np.ndarray) -> np.ndarray:
    """VLAD: signed square root, then global L2."""
    return l2(signed_sqrt(vector.astype(np.float32)))
