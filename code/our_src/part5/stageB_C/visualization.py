"""Figures, built only from the saved CSVs so they can be regenerated without features."""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from .config import Config
from .data import Dataset
from .metrics import interpolated_pr_curve
from .results import PER_QUERY_FILE, SUMMARY_FILE, load_ranking

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
RELEVANT, IRRELEVANT = "#0ca30c", "#d03b3b"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY, "text.color": INK,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False, "font.size": 10,
}


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _random_map(summary: pd.DataFrame) -> float | None:
    rows = summary[summary["method"] == "random"]
    return float(rows["map"].iloc[0]) if len(rows) else None


def _reference_line(ax: plt.Axes, value: float | None, label: str) -> None:
    if value is not None:
        ax.axhline(value, color=MUTED, linestyle="--", linewidth=1.5, label=label)


# ------------------------------------------------------------------ qualitative grids
class _Thumbnails:
    """Loads each image once; the grids share most of their images."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._cache: dict[Path, np.ndarray] = {}

    def get(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            with Image.open(path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((self.size, self.size))
                self._cache[path] = np.asarray(thumb)
        return self._cache[path]


def _image_panel(ax: plt.Axes, image: np.ndarray, colour: str, title: str, caption: str) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(colour)
        spine.set_linewidth(3.5)
    ax.set_title(title, fontsize=8, color=INK_SECONDARY, pad=4)
    ax.set_xlabel(caption, fontsize=9, color=INK, labelpad=4)


def _text_panel(ax: plt.Axes, text: str, kind: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.5)
    ax.text(0.5, 0.5, textwrap.fill(f"“{text}”", 18), ha="center", va="center", fontsize=9,
            transform=ax.transAxes)
    ax.set_title("text query", fontsize=8, color=INK_SECONDARY, pad=4)
    ax.set_xlabel(kind, fontsize=9, labelpad=4)


def retrieval_grid(cfg: Config, dataset: Dataset, experiment: str, per_query: pd.DataFrame,
                   thumbnails: _Thumbnails, out_dir: Path) -> None:
    """Best and worst query of one experiment with their top-ranked images."""
    ranking = load_ranking(cfg.output_dir, experiment)
    top_k = cfg.figures["top_k"]
    rows = per_query[per_query["experiment"] == experiment]
    picks = [("Best", rows.loc[rows["ap"].idxmax()]), ("Worst", rows.loc[rows["ap"].idxmin()])]
    corpus_paths = dict(zip(dataset.corpus["id"], dataset.corpus["image_path"]))
    query_frame = dataset.queries.set_index("id")
    text_frame = dataset.text_queries.set_index("id")

    fig, axes = plt.subplots(2, top_k + 1, figsize=(1.55 * (top_k + 1), 4.3), squeeze=False)
    for row_axes, (kind, row) in zip(axes, picks):
        query_id = row["query_id"]
        if row["query_type"] == "text":
            _text_panel(row_axes[0], text_frame.loc[query_id, "text"], text_frame.loc[query_id, "type"])
        else:
            _image_panel(row_axes[0], thumbnails.get(query_frame.loc[query_id, "image_path"]), INK, "query",
                         query_frame.loc[query_id, "label"])
        top = ranking[ranking["query_id"] == query_id].nsmallest(top_k, "rank")
        for ax, hit in zip(row_axes[1:], top.itertuples()):
            mark = "✓" if hit.relevant else "✗"
            _image_panel(ax, thumbnails.get(corpus_paths[hit.corpus_id]), RELEVANT if hit.relevant else IRRELEVANT,
                         f"#{hit.rank}  {hit.score:.3f}", f"{hit.label} {mark}")
        row_axes[0].text(0, 1.2, f"{kind} query: {query_id} (AP = {row['ap']:.3f})", fontsize=10, fontweight="bold",
                         transform=row_axes[0].transAxes)
    fig.suptitle(f"{experiment}: top-{top_k} results (✓ green = relevant, ✗ red = irrelevant)",
                 fontsize=11, y=1.0)
    # Fixed margins: tight_layout would shrink the panels to fit the long row headings.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.07, wspace=0.08, hspace=0.55)
    _save(fig, out_dir / "retrieval" / f"{experiment}.png", cfg.figures["dpi"])


# ------------------------------------------------------------------ parameter sweeps
def _power_of_two_axis(ax: plt.Axes, values: np.ndarray) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(values, [str(int(v)) for v in values])
    ax.minorticks_off()


def map_vs_k(cfg: Config, summary: pd.DataFrame, out_dir: Path) -> None:
    panels = [(m, title, colour) for m, title, colour in
              (("bovw", "BoVW", SERIES[0]), ("vlad", "VLAD", SERIES[1])) if (summary["method"] == m).any()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), sharey=True, squeeze=False)
    top = summary.loc[summary["method"].isin([p[0] for p in panels]), "map"].max()
    for ax, (method, title, colour) in zip(axes[0], panels):
        rows = summary[summary["method"] == method].sort_values("k")
        ax.plot(rows["k"], rows["map"], color=colour, marker="o", label=f"{title} mAP")
        _reference_line(ax, _random_map(summary), "random baseline")
        _power_of_two_axis(ax, rows["k"].to_numpy())
        ax.set_title(f"{title}: mAP vs vocabulary size", loc="left")
        ax.set_xlabel("vocabulary size k")
        ax.set_ylim(0, top * 1.15)
        ax.legend(loc="lower right")
    axes[0][0].set_ylabel("mAP")
    fig.tight_layout()
    _save(fig, out_dir / "map_vs_k.png", cfg.figures["dpi"])


def map_vs_pca(cfg: Config, summary: pd.DataFrame, out_dir: Path) -> None:
    rows = summary[summary["method"] == "vlad_pca"].sort_values("pca_components")
    if rows.empty:
        return
    k = rows["k"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(rows["pca_components"], rows["map"], color=SERIES[1], marker="o", label=f"VLAD k={k:g} + PCA")
    reference = summary[(summary["method"] == "vlad") & (summary["k"] == k)]
    top = rows["map"].max()
    if len(reference):
        value = float(reference["map"].iloc[0])
        dim = int(reference["embedding_dim"].iloc[0])
        ax.axhline(value, color=SERIES[0], linewidth=2, label=f"VLAD k={k:g} without PCA ({dim}-d)")
        top = max(top, value)
    _reference_line(ax, _random_map(summary), "random baseline")
    _power_of_two_axis(ax, rows["pca_components"].to_numpy())
    ax.set_ylim(0, top * 1.15)
    ax.set_title(f"VLAD k={k:g}: mAP vs PCA components", loc="left")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("mAP")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir / "map_vs_pca.png", cfg.figures["dpi"])


# ------------------------------------------------------------------ text prompts
def clip_text_prompts(cfg: Config, per_query: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    """AP per prompt, one chart per text experiment. Bars are grouped by the configured type families."""
    families: dict[str, list[str]] = cfg.figures.get("text_type_families", {})
    family_of = {kind: name for name, kinds in families.items() for kind in kinds}
    order = list(families) + ["other"]
    for experiment in summary.loc[summary["query_type"] == "text", "experiment"]:
        rows = per_query[per_query["experiment"] == experiment].copy()
        rows["family"] = rows["text_type"].map(lambda kind: family_of.get(kind, "other"))
        rows["family_rank"] = rows["family"].map(order.index)
        rows = rows.sort_values(["family_rank", "ap"], ascending=[False, True])

        fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.6))
        colours = [SERIES[order.index(f)] for f in rows["family"]]
        y = np.arange(len(rows))
        ax.barh(y, rows["ap"], color=colours, height=0.72, edgecolor=SURFACE, linewidth=2)
        for position, value in zip(y, rows["ap"]):
            ax.text(value + 0.012, position, f"{value:.3f}", va="center", fontsize=9, color=INK_SECONDARY, zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
        labels = [f"{text}  [{kind}]" for text, kind in zip(rows["query_text"], rows["text_type"])]
        ax.set_yticks(y, [textwrap.fill(label, 52) for label in labels], fontsize=9)
        ax.grid(axis="y", visible=False)
        baseline = _random_map(summary)
        if baseline is not None:
            ax.axvline(baseline, color=MUTED, linestyle="--", linewidth=1.5, label=f"random baseline ({baseline:.3f})")
        for name in order:
            if (rows["family"] == name).any():
                ax.barh([0], [0], color=SERIES[order.index(name)], label=name)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("AP")
        ax.set_title(f"{experiment}: AP per text prompt", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.12 - 0.3 / len(rows)), ncol=4, fontsize=8.5)
        fig.tight_layout()
        _save(fig, out_dir / f"text_prompts_{experiment}.png", cfg.figures["dpi"])


# ------------------------------------------------------------------ query sensitivity
def ap_heatmap(cfg: Config, summary: pd.DataFrame, per_query: pd.DataFrame, out_dir: Path) -> None:
    experiments = summary.loc[summary["query_type"] == "image", "experiment"].tolist()
    rows = per_query[per_query["experiment"].isin(experiments)]
    if rows.empty:
        return
    table = rows.pivot(index="experiment", columns="query_id", values="ap").loc[experiments]
    fig, ax = plt.subplots(figsize=(0.75 * table.shape[1] + 2.6, 0.33 * table.shape[0] + 1.5))
    image = ax.imshow(table.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for (r, c), value in np.ndenumerate(table.to_numpy()):
        ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if value > 0.5 else INK)
    fig.colorbar(image, ax=ax, label="AP", fraction=0.03, pad=0.02)
    ax.set_title("AP per query and method (image queries)", loc="left")
    fig.tight_layout()
    _save(fig, out_dir / "ap_heatmap.png", cfg.figures["dpi"])


# ------------------------------------------------------------------ precision-recall
def pr_curves(cfg: Config, dataset: Dataset, summary: pd.DataFrame, out_dir: Path) -> None:
    """Mean interpolated precision-recall curve of the best variant (by mAP) of each group."""
    grid = np.linspace(0, 1, cfg.figures["pr_recall_points"])
    groups = [g for g in cfg.groups if (summary["group"] == g).any()]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    slots = iter(SERIES)
    for group in groups:
        rows = summary[summary["group"] == group]
        best = rows.loc[rows["map"].idxmax()]
        ranking = load_ranking(cfg.output_dir, best["experiment"])
        curves = [interpolated_pr_curve(part.sort_values("rank")["relevant"].to_numpy(), dataset.n_relevant, grid)
                  for _, part in ranking.groupby("query_id")]
        is_random = best["method"] == "random"
        ax.plot(grid, np.mean(curves, axis=0), color=MUTED if is_random else next(slots),
                linestyle="--" if is_random else "-", linewidth=1.5 if is_random else 2,
                label=f"{best['experiment']} (mAP {best['map']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("mean interpolated precision")
    ax.set_title("Precision-recall: best variant of each method", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    fig.tight_layout()
    _save(fig, out_dir / "pr_curves.png", cfg.figures["dpi"])


# ------------------------------------------------------------------ entry point
def make_all_figures(cfg: Config, dataset: Dataset, logger: logging.Logger) -> None:
    summary_path = cfg.output_dir / SUMMARY_FILE
    per_query_path = cfg.output_dir / PER_QUERY_FILE
    if not summary_path.exists() or not per_query_path.exists():
        logger.warning("No results in %s; figures skipped", cfg.output_dir)
        return
    summary = pd.read_csv(summary_path)
    per_query = pd.read_csv(per_query_path, keep_default_na=False, na_values=[""])
    out_dir = cfg.output_dir / "figures"
    thumbnails = _Thumbnails(cfg.figures["thumb_size"])
    with plt.rc_context(STYLE):
        for experiment in summary["experiment"]:
            retrieval_grid(cfg, dataset, experiment, per_query, thumbnails, out_dir)
        map_vs_k(cfg, summary, out_dir)
        map_vs_pca(cfg, summary, out_dir)
        clip_text_prompts(cfg, per_query, summary, out_dir)
        ap_heatmap(cfg, summary, per_query, out_dir)
        pr_curves(cfg, dataset, summary, out_dir)
    logger.info("Figures written to %s", out_dir)
