"""Cumulative explained-variance curves of PCA on (normalized) TRAIN encodings."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .reduction import components_for_variance, cumulative_explained_variance  # noqa: E402

SURFACE, INK, INK_MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def run_pca_variance(representation: str, k: int, seed: int, x_train: np.ndarray, used_dims: list[int],
                     thresholds: list[float], out_dir: Path, logger: logging.Logger) -> dict:
    """Save the curve (.csv + .png) and return d<threshold> values (e.g. d80, d90, d95)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{representation}_K{k}_seed{seed}"
    cumulative = cumulative_explained_variance(x_train.astype(np.float64))
    n = np.arange(1, len(cumulative) + 1)
    pd.DataFrame({"n_components": n, "cumulative_explained_variance": cumulative}).to_csv(
        out_dir / f"{stem}.csv", index=False)

    d_values = {f"d{round(t * 100)}": components_for_variance(cumulative, t) for t in thresholds}
    info = {"representation": representation, "k": k, "seed": seed, "original_dim": x_train.shape[1],
            "n_train": x_train.shape[0], "max_components": len(cumulative), **d_values}
    for d in used_dims:
        info[f"variance_at_{d}"] = float(cumulative[min(d, len(cumulative)) - 1])
    logger.info("PCA variance %s: %s", stem, ", ".join(f"{key}={v}" for key, v in d_values.items()))

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(n, cumulative, color=SERIES, linewidth=2)
    for t in thresholds:
        ax.axhline(t, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=0)
        ax.text(len(cumulative), t, f" {t:.0%} (d={d_values[f'd{round(t * 100)}']})", va="center",
                ha="left", fontsize=8, color=INK_MUTED)
    for d in used_dims:
        if d <= len(cumulative):
            y = cumulative[d - 1]
            ax.plot([d], [y], marker="o", markersize=7, color=SERIES, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=3)
            ax.annotate(f"{d}: {y:.0%}", (d, y), textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Number of principal components (log scale)", color=INK)
    ax.set_ylabel("Cumulative explained variance", color=INK)
    ax.set_title(f"{representation.upper()} K={k} (dim {x_train.shape[1]}), seed {seed}: "
                 f"PCA on {x_train.shape[0]} train images", fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return info
