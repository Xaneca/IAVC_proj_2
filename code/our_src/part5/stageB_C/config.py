"""Configuration loading, validation and expansion of experiment groups."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # the `code/` directory
METHODS = ("random", "bovw", "vlad", "vlad_pca", "deep")
QUERY_TYPES = ("image", "text")
DEEP_FAMILIES = ("torchvision", "dinov2", "clip")
DINOV2_PATCH_SIZE = 14


@dataclass(frozen=True)
class Experiment:
    """One row of ``experiments_summary.csv``."""

    name: str
    group: str
    method: str
    query_type: str
    k: int | None = None
    n_components: int | None = None
    family: str | None = None
    model: str | None = None
    weights: str | None = None
    input_size: int | None = None


@dataclass
class Config:
    """Parsed configuration. Sections without logic stay as plain dictionaries."""

    corpus_csv: Path
    queries_csv: Path
    text_queries_csv: Path
    output_dir: Path
    cache_dir: Path
    seed: int
    device: str
    precision_at: list[int]
    recall_at: list[int]
    sift: dict[str, Any]
    kmeans: dict[str, Any]
    deep: dict[str, Any]
    ranking_repeats: int
    figures: dict[str, Any]
    groups: dict[str, dict[str, Any]]
    smoke: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def experiments(self, n_corpus: int) -> list[Experiment]:
        """Expand all groups into experiments, clamping PCA to its maximum for this corpus."""
        return [exp for group in self.groups for exp in expand_group(group, self.groups[group], n_corpus)]


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def expand_group(group: str, spec: dict[str, Any], n_corpus: int) -> list[Experiment]:
    """Turn one config entry into experiments.

    PCA on ``n_corpus`` centered vectors has rank at most ``n_corpus - 1``, so larger requests
    are clamped; requests that clamp to the same value produce a single experiment.
    """
    method, query_type = spec.get("method"), spec.get("query_type", "image")
    if method not in METHODS:
        raise ValueError(f"experiments.{group}: unknown method {method!r}, expected one of {METHODS}")
    if query_type not in QUERY_TYPES:
        raise ValueError(f"experiments.{group}: unknown query_type {query_type!r}")
    if query_type == "text" and spec.get("family") != "clip":
        raise ValueError(f"experiments.{group}: text queries require a CLIP model")

    if method == "random":
        return [Experiment(group, group, method, query_type)]
    if method in ("bovw", "vlad"):
        return [Experiment(f"{method}_k{int(k)}", group, method, query_type, k=int(k))
                for k in _as_list(spec["k"])]
    if method == "vlad_pca":
        out: list[Experiment] = []
        seen: set[tuple[int, int]] = set()
        for k in _as_list(spec["k"]):
            for requested in _as_list(spec["n_components"]):
                dims = min(int(requested), n_corpus - 1)
                if (int(k), dims) not in seen:
                    seen.add((int(k), dims))
                    out.append(Experiment(f"vlad_k{int(k)}_pca{dims}", group, method, query_type,
                                          k=int(k), n_components=dims))
        return out

    family = spec.get("family")
    if family not in DEEP_FAMILIES:
        raise ValueError(f"experiments.{group}: unknown family {family!r}, expected one of {DEEP_FAMILIES}")
    input_size = spec.get("input_size")
    if family != "clip" and input_size is None:
        raise ValueError(f"experiments.{group}: input_size is required for family {family}")
    if family == "dinov2" and int(input_size) % DINOV2_PATCH_SIZE:
        raise ValueError(f"experiments.{group}: input_size must be a multiple of {DINOV2_PATCH_SIZE}")
    name = spec.get("name", f"{group}_{spec['model']}")
    return [Experiment(name, group, method, query_type, family=family, model=spec["model"],
                       weights=spec.get("weights"), input_size=input_size)]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path, smoke: bool = False) -> Config:
    """Load the YAML config; with ``smoke`` apply the subset overrides and redirect the outputs."""
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    groups = {name: dict(spec) for name, spec in raw["experiments"].items()}
    kmeans = dict(raw["kmeans"])
    output_dir = _resolve(raw["paths"]["output_dir"])
    if kmeans.get("algorithm") not in ("minibatch", "full"):
        raise ValueError("kmeans.algorithm must be 'minibatch' or 'full'")

    smoke_section = None
    if smoke:
        smoke_section = raw["smoke"]
        kmeans["max_descriptors"] = smoke_section["max_descriptors"]
        output_dir = output_dir / "smoke"
        for name, overrides in smoke_section.get("experiment_overrides", {}).items():
            groups[name].update(overrides)

    paths = raw["paths"]
    return Config(
        corpus_csv=_resolve(paths["corpus_csv"]),
        queries_csv=_resolve(paths["queries_csv"]),
        text_queries_csv=_resolve(paths["text_queries_csv"]),
        output_dir=output_dir,
        cache_dir=_resolve(paths["cache_dir"]),
        seed=int(raw["seed"]),
        device=str(raw.get("device", "auto")),
        precision_at=[int(k) for k in raw["metrics"]["precision_at"]],
        recall_at=[int(k) for k in raw["metrics"]["recall_at"]],
        sift=dict(raw["sift"]),
        kmeans=kmeans,
        deep=dict(raw["deep"]),
        ranking_repeats=int(raw["timing"]["ranking_repeats"]),
        figures=dict(raw["figures"]),
        groups=groups,
        smoke=smoke_section,
        raw=raw,
    )
