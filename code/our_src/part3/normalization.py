"""Composable feature normalization steps (sqrt, l1, l2, none).

Every step works row-wise on a 2D array ``(n_samples, n_features)``. Zero rows stay zero.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def signed_power(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Signed power normalization ``sign(x) * |x|**alpha`` (alpha=0.5 is the signed sqrt)."""
    return np.sign(x) * np.power(np.abs(x), alpha)


def l1_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.abs(x).sum(axis=1, keepdims=True)
    return x / np.maximum(norms, _EPS)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, _EPS)


def apply_steps(x: np.ndarray, steps: list[str], sqrt_alpha: float = 0.5) -> np.ndarray:
    """Apply the normalization ``steps`` from left to right and return a float32 array."""
    out = np.asarray(x, dtype=np.float32)
    if out.ndim == 1:
        return apply_steps(out[None, :], steps, sqrt_alpha)[0]
    for step in steps:
        if step == "sqrt":
            out = signed_power(out, sqrt_alpha)
        elif step == "l1":
            out = l1_normalize(out)
        elif step == "l2":
            out = l2_normalize(out)
        elif step == "none":
            continue
        else:
            raise ValueError(f"Unknown normalization step: {step}")
    return out.astype(np.float32, copy=False)
