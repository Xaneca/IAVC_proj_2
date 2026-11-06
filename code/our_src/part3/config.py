"""Configuration loading and validation for the Part 3 experiments."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPRESENTATIONS = ("bovw", "vlad", "fisher")
NORMALIZATION_STEPS = ("sqrt", "l1", "l2", "none")
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # the `code/` directory


@dataclass(frozen=True)
class Experiment:
    """One row family of the results table: a representation, a vocabulary size and an
    optional PCA dimension (``pca_dim == 0`` means no PCA, as in src's ``C0``)."""

    representation: str
    k: int
    pca_dim: int = 0

    @property
    def pca(self) -> bool:
        return self.pca_dim > 0

    @property
    def name(self) -> str:
        suffix = f"_pca{self.pca_dim}" if self.pca else ""
        return f"{self.representation}_K{self.k}{suffix}"


@dataclass
class Paths:
    image_dir: Path
    split_dir: Path
    output_dir: Path
    cache_dir: Path


@dataclass
class DataConfig:
    train_splits: list[str]
    test_splits: list[str]
    target_classes: list[str]
    negatives: str


@dataclass
class EncodingConfig:
    max_descriptors_per_image: int | None
    fisher_include_weight_grads: bool


@dataclass
class CodebookConfig:
    n_descriptors: int
    kmeans: dict[str, Any]
    gmm: dict[str, Any]


@dataclass
class NormalizationConfig:
    post_encoding: list[str]
    post_pca: list[str]
    sqrt_alpha: float


@dataclass
class PCAConfig:
    whiten: bool
    svd_solver: str
    full_max_size: int


@dataclass
class SVMConfig:
    kernel: str
    C_grid: list[float]
    gamma_grid: list[Any]
    class_weight: str | None
    max_iter: int
    cv_folds: int
    cv_scoring: str
    cv_n_jobs: int


@dataclass
class PCAVarianceConfig:
    enabled: bool
    targets: list[tuple[str, int]]
    thresholds: list[float]


@dataclass
class Config:
    paths: Paths
    data: DataConfig
    seeds: list[int]
    n_jobs: int
    chunk_size: int
    encoding: EncodingConfig
    codebook: CodebookConfig
    normalization: NormalizationConfig
    pca: PCAConfig
    svm: SVMConfig
    inference_max_images: int | None
    experiments: list[Experiment]
    pca_variance: PCAVarianceConfig
    smoke: bool = False
    smoke_images: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("raw")
        out["paths"] = {k: str(v) for k, v in out["paths"].items()}
        out["experiments"] = [e.name for e in self.experiments]
        return out


def expand_experiments(spec: dict[str, Any]) -> list[Experiment]:
    """Turn the ``experiments`` section into an ordered list of :class:`Experiment`."""
    experiments: list[Experiment] = []
    for representation, section in spec.items():
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation '{representation}', expected one of {REPRESENTATIONS}")
        section = section or {}
        for k in section.get("no_pca_k", []) or []:
            experiments.append(Experiment(representation, int(k), 0))
        for k, dims in (section.get("pca", {}) or {}).items():
            for dim in dims:
                if int(dim) <= 0:
                    raise ValueError(f"PCA dimension must be positive, got {dim} for {representation} K={k}")
                experiments.append(Experiment(representation, int(k), int(dim)))
    if len(set(experiments)) != len(experiments):
        raise ValueError("Duplicated experiments in the config")
    return experiments


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _check_steps(steps: list[str], name: str) -> list[str]:
    steps = [str(s).lower() for s in (steps or [])]
    bad = [s for s in steps if s not in NORMALIZATION_STEPS]
    if bad:
        raise ValueError(f"normalization.{name}: unknown steps {bad}, expected {NORMALIZATION_STEPS}")
    return steps


def load_config(path: str | Path, smoke: bool = False, seeds: list[int] | None = None) -> Config:
    """Load ``config.yaml``, apply the ``smoke`` overrides and validate every section."""
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw_original = copy.deepcopy(raw)

    if smoke:
        s = raw["smoke"]
        raw["seeds"] = s["seeds"]
        raw["experiments"] = s["experiments"]
        raw["codebook"]["n_descriptors"] = s["n_descriptors"]
        raw["svm"]["cv_folds"] = s["cv_folds"]
        raw["svm"]["C_grid"] = s["C_grid"]
        raw["pca_variance"]["targets"] = s["pca_variance_targets"]
        raw["paths"]["output_dir"] = str(Path(raw["paths"]["output_dir"]) / "smoke")
    if seeds:
        raw["seeds"] = list(seeds)

    data = DataConfig(**raw["data"])
    if data.negatives not in ("all", "background"):
        raise ValueError("data.negatives must be 'all' or 'background'")
    if not data.target_classes:
        raise ValueError("data.target_classes must not be empty")
    if set(data.train_splits) & set(data.test_splits):
        raise ValueError("train_splits and test_splits overlap")

    svm = SVMConfig(**raw["svm"])
    if svm.kernel not in ("linear", "rbf"):
        raise ValueError("svm.kernel must be 'linear' or 'rbf'")
    if svm.cv_folds < 2:
        raise ValueError("svm.cv_folds must be >= 2")

    pca = PCAConfig(**raw["pca"])
    if pca.svd_solver not in ("full", "randomized", "auto"):
        raise ValueError("pca.svd_solver must be full, randomized or auto")

    norm = raw["normalization"]
    normalization = NormalizationConfig(
        post_encoding=_check_steps(norm.get("post_encoding"), "post_encoding"),
        post_pca=_check_steps(norm.get("post_pca"), "post_pca"),
        sqrt_alpha=float(norm.get("sqrt_alpha", 0.5)),
    )

    codebook = CodebookConfig(**raw["codebook"])
    if codebook.kmeans.get("algorithm") not in ("minibatch", "full"):
        raise ValueError("codebook.kmeans.algorithm must be 'minibatch' or 'full'")

    pv = raw.get("pca_variance", {}) or {}
    pca_variance = PCAVarianceConfig(
        enabled=bool(pv.get("enabled", False)),
        targets=[(t["representation"], int(t["k"])) for t in pv.get("targets", [])],
        thresholds=[float(t) for t in pv.get("thresholds", [0.8, 0.9, 0.95])],
    )

    seeds_list = [int(s) for s in raw["seeds"]]
    if not seeds_list:
        raise ValueError("At least one seed is required")

    return Config(
        paths=Paths(**{k: _resolve(v) for k, v in raw["paths"].items()}),
        data=data,
        seeds=seeds_list,
        n_jobs=int(raw.get("n_jobs", -2)),
        chunk_size=int(raw.get("chunk_size", 25)),
        encoding=EncodingConfig(**raw["encoding"]),
        codebook=codebook,
        normalization=normalization,
        pca=pca,
        svm=svm,
        inference_max_images=(raw.get("timing", {}) or {}).get("inference_max_images"),
        experiments=expand_experiments(raw["experiments"]),
        pca_variance=pca_variance,
        smoke=smoke,
        smoke_images=dict(raw["smoke"]["images"]) if smoke else {},
        raw=raw_original,
    )
