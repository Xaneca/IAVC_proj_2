"""From an experiment definition to its normalized vectors, costs, rankings and metrics.

Everything learned (vocabularies, PCA) is fitted on the corpus only. Queries go through the same
pipeline one at a time, so their timings match a real query.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import Config, Experiment
from .data import Dataset
from .encoders import DESCRIPTOR_DIM, ImageEncoder, VladPCA, fit_vocabulary
from .metrics import evaluate_ranking
from .retrieval import cosine_scores, random_score_fn, rank_queries
from .sift import SiftStore
from .timing import BYTES_PER_VALUE, Timer, representation_bytes


def fingerprint(obj: Any) -> str:
    """Short stable hash of a JSON-serialisable object, used to key cache files."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class Representation:
    """Normalized vectors of one experiment, with what they cost to compute."""

    corpus: np.ndarray | None = None
    queries: np.ndarray | None = None
    corpus_sift_s: float = 0.0
    corpus_encoding_s: float = 0.0
    vocabulary_fit_s: float = 0.0
    pca_fit_s: float = 0.0
    query_sift_s: np.ndarray | None = None
    query_encoding_s: np.ndarray | None = None
    vocabulary_bytes: int = 0
    pca_bytes: int = 0
    model_weights_bytes: int = 0
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)


class FeatureBuilder:
    """Builds representations, reusing disk caches and sharing results within a run."""

    def __init__(self, cfg: Config, dataset: Dataset, device: str, logger: logging.Logger, force: bool = False) -> None:
        self.cfg, self.dataset, self.device, self.logger, self.force = cfg, dataset, device, logger, force
        self.corpus_ids: list[str] = dataset.corpus["id"].tolist()
        self.query_ids: list[str] = dataset.queries["id"].tolist()
        self._sift: SiftStore | None = None
        self._computed: set[Path] = set()  # files written in this run, valid even with --force
        self._encoders: dict[tuple, Any] = {}

    # ------------------------------------------------------------------ cache helpers
    def _cached_npz(self, path: Path, compute: Callable[[], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Load ``path``, or compute the arrays and store them atomically."""
        if path.exists() and (not self.force or path in self._computed):
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        arrays = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(tmp, path)
        self._computed.add(path)
        return arrays

    # ------------------------------------------------------------------ dense SIFT
    def _sift_key(self) -> str:
        return fingerprint(self.cfg.sift)

    def sift(self) -> SiftStore:
        if self._sift is None:
            store = SiftStore(self.cfg.cache_dir / "sift" / self._sift_key(), self.cfg.sift, self.logger)
            frame = self.dataset
            images = dict(zip(frame.corpus["id"], frame.corpus["image_path"]))
            images.update(zip(frame.queries["id"], frame.queries["image_path"]))
            store.ensure(images, force=self.force)
            self._sift = store
        return self._sift

    # ------------------------------------------------------------------ vocabulary
    def _vocabulary_key(self, k: int) -> str:
        return fingerprint({"sift": self._sift_key(), "k": k, "seed": self.cfg.seed, "kmeans": self.cfg.kmeans,
                            "corpus": self.corpus_ids})

    def vocabulary(self, k: int) -> dict[str, np.ndarray]:
        def compute() -> dict[str, np.ndarray]:
            sample = self.sift().sample(self.corpus_ids, self.cfg.kmeans["max_descriptors"], self.cfg.seed)
            words, seconds = fit_vocabulary(sample, k, self.cfg.kmeans, self.cfg.seed)
            self.logger.info("Fitted vocabulary k=%d on %d descriptors in %.1fs", k, len(sample), seconds)
            return {"words": words, "fit_seconds": np.array(seconds), "n_descriptors": np.array(len(sample))}

        return self._cached_npz(self.cfg.cache_dir / "vocabularies" / f"k{k}_{self._vocabulary_key(k)}.npz", compute)

    # ------------------------------------------------------------------ BoVW and VLAD
    def _encode_images(self, encoder: ImageEncoder, image_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        sift = self.sift()
        vectors, seconds = [], []
        for image_id in image_ids:
            with Timer() as timer:
                vectors.append(encoder.encode(sift.get(image_id)))
            seconds.append(timer.elapsed)
        return np.stack(vectors).astype(np.float32), np.asarray(seconds)

    def encoding(self, method: str, k: int) -> dict[str, np.ndarray]:
        def compute() -> dict[str, np.ndarray]:
            encoder = ImageEncoder(method, self.vocabulary(k)["words"])
            corpus, corpus_seconds = self._encode_images(encoder, self.corpus_ids)
            queries, query_seconds = self._encode_images(encoder, self.query_ids)
            self.logger.info("Encoded %s k=%d in %.1fs", method, k, corpus_seconds.sum() + query_seconds.sum())
            return {"corpus": corpus, "queries": queries, "corpus_seconds": corpus_seconds,
                    "query_seconds": query_seconds}

        key = fingerprint({"vocabulary": self._vocabulary_key(k), "method": method, "queries": self.query_ids})
        return self._cached_npz(self.cfg.cache_dir / "encodings" / f"{method}_k{k}_{key}.npz", compute)

    def _build_local(self, exp: Experiment) -> Representation:
        method = "vlad" if exp.method == "vlad_pca" else exp.method
        vocabulary = self.vocabulary(exp.k)
        encoded = self.encoding(method, exp.k)
        sift = self.sift()
        rep = Representation(
            corpus=encoded["corpus"], queries=encoded["queries"],
            corpus_sift_s=float(sum(sift.times[i] for i in self.corpus_ids)),
            corpus_encoding_s=float(encoded["corpus_seconds"].sum()),
            vocabulary_fit_s=float(vocabulary["fit_seconds"]),
            query_sift_s=np.array([sift.times[i] for i in self.query_ids]),
            query_encoding_s=encoded["query_seconds"].copy(),
            vocabulary_bytes=int(vocabulary["words"].size * BYTES_PER_VALUE),
            extra={"n_vocabulary_descriptors": int(vocabulary["n_descriptors"])},
        )
        if exp.method == "vlad_pca":
            self._apply_pca(rep, exp.n_components)
        return rep

    def _apply_pca(self, rep: Representation, n_components: int) -> None:
        """Project the VLADs. PCA is cheap to refit, so it is not cached."""
        pca = VladPCA(n_components, self.cfg.seed).fit(rep.corpus)
        with Timer() as timer:
            corpus = pca.transform(rep.corpus)
        rep.corpus_encoding_s += timer.elapsed
        queries = []
        for index, vector in enumerate(rep.queries):
            with Timer() as timer:
                queries.append(pca.transform(vector[None])[0])
            rep.query_encoding_s[index] += timer.elapsed
        rep.corpus, rep.queries = corpus, np.stack(queries)
        rep.pca_fit_s = pca.fit_seconds
        rep.pca_bytes = pca.storage_bytes
        rep.extra["pca_explained_variance"] = pca.explained_variance

    # ------------------------------------------------------------------ deep models
    def _deep_encoder(self, exp: Experiment):
        from .deep import DeepEncoder

        key = (exp.family, exp.model, exp.weights, exp.input_size)
        if key not in self._encoders:
            self.logger.info("Loading %s on %s", exp.model, self.device)
            self._encoders.clear()  # keep at most one model in GPU memory
            self._encoders[key] = DeepEncoder(exp.family, exp.model, self.device, exp.weights, exp.input_size)
        return self._encoders[key]

    def _build_deep(self, exp: Experiment) -> Representation:
        identity = {"family": exp.family, "model": exp.model, "weights": exp.weights, "input_size": exp.input_size,
                    "device": self.device}
        directory = self.cfg.cache_dir / "embeddings"
        slug = str(exp.model).replace("/", "-")
        batch = self.cfg.deep["corpus_batch_size"]
        data = self.dataset

        def compute_corpus() -> dict[str, np.ndarray]:
            encoder = self._deep_encoder(exp)
            features, seconds = encoder.encode_corpus(list(data.corpus["image_path"]), batch)
            self.logger.info("%s: encoded %d corpus images in %.2fs", exp.model, len(features), seconds)
            return {"features": features, "seconds": np.array(seconds), "weights_bytes": np.array(encoder.weights_bytes)}

        def compute_queries() -> dict[str, np.ndarray]:
            features, seconds = self._deep_encoder(exp).encode_queries(list(data.queries["image_path"]))
            return {"features": features, "seconds": seconds}

        def compute_text() -> dict[str, np.ndarray]:
            features, seconds = self._deep_encoder(exp).encode_texts(data.text_queries["text"].tolist())
            return {"features": features, "seconds": seconds}

        corpus_key = fingerprint({**identity, "ids": self.corpus_ids, "batch": batch})
        corpus = self._cached_npz(directory / f"{slug}_corpus_{corpus_key}.npz", compute_corpus)
        if exp.query_type == "text":
            key = fingerprint({**identity, "texts": data.text_queries["text"].tolist()})
            queries = self._cached_npz(directory / f"{slug}_text_{key}.npz", compute_text)
        else:
            key = fingerprint({**identity, "ids": self.query_ids})
            queries = self._cached_npz(directory / f"{slug}_queries_{key}.npz", compute_queries)
        return Representation(
            corpus=corpus["features"], queries=queries["features"],
            corpus_encoding_s=float(corpus["seconds"]), query_encoding_s=queries["seconds"],
            query_sift_s=np.zeros(len(queries["features"])),
            model_weights_bytes=int(corpus["weights_bytes"]), device=self.device,
        )

    # ------------------------------------------------------------------ dispatch
    def build(self, exp: Experiment) -> Representation:
        if exp.method == "random":
            n_queries = len(self.dataset.query_ids(exp.query_type))
            return Representation(query_sift_s=np.zeros(n_queries), query_encoding_s=np.zeros(n_queries))
        return self._build_deep(exp) if exp.method == "deep" else self._build_local(exp)


def run_rankings(cfg: Config, exp: Experiment, rep: Representation, n_corpus: int,
                 n_queries: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank the corpus for every query of the experiment."""
    if exp.method == "random":
        score_fn = random_score_fn(cfg.seed, n_corpus)
    else:
        score_fn = lambda q: cosine_scores(rep.corpus, rep.queries[q])  # noqa: E731
    return rank_queries(score_fn, n_queries, cfg.ranking_repeats)


def per_query_rows(cfg: Config, dataset: Dataset, exp: Experiment, orders: np.ndarray) -> list[dict[str, Any]]:
    """One row of metrics per query."""
    relevance = dataset.relevance
    frame = dataset.text_queries if exp.query_type == "text" else dataset.queries
    rows = []
    for index, query in enumerate(frame.itertuples()):
        metrics = evaluate_ranking(relevance[orders[index]], dataset.n_relevant, cfg.precision_at, cfg.recall_at)
        rows.append({
            "experiment": exp.name, "group": exp.group, "query_id": query.id, "query_type": exp.query_type,
            "query_text": query.text if exp.query_type == "text" else "",
            "text_type": query.type if exp.query_type == "text" else "",
            **metrics,
        })
    return rows


def summary_row(cfg: Config, exp: Experiment, rep: Representation, query_rows: list[dict[str, Any]],
                ranking_seconds: np.ndarray, n_corpus: int) -> dict[str, Any]:
    """The ``experiments_summary.csv`` row of one experiment."""
    ap = np.array([row["ap"] for row in query_rows])
    dim = 0 if rep.corpus is None else int(rep.corpus.shape[1])
    extraction = rep.corpus_sift_s + rep.corpus_encoding_s
    query_extraction = rep.query_sift_s + rep.query_encoding_s
    row: dict[str, Any] = {
        "experiment": exp.name, "group": exp.group, "method": exp.method, "k": exp.k,
        "pca_components": exp.n_components, "model": exp.model, "query_type": exp.query_type,
        "n_queries": len(query_rows), "map": float(ap.mean()),
        "ap_std": float(ap.std(ddof=1)) if len(ap) > 1 else 0.0,
    }
    for key in [f"p_at_{k}" for k in cfg.precision_at] + [f"r_at_{k}" for k in cfg.recall_at]:
        row[f"mean_{key}"] = float(np.mean([r[key] for r in query_rows]))
    row.update({
        "embedding_dim": dim,
        "sift_total_s": rep.corpus_sift_s,
        "sift_per_image_s": rep.corpus_sift_s / n_corpus,
        "encoding_total_s": rep.corpus_encoding_s,
        "encoding_per_image_s": rep.corpus_encoding_s / n_corpus,
        "extraction_total_s": extraction,
        "extraction_per_image_s": extraction / n_corpus,
        "vocabulary_fit_s": rep.vocabulary_fit_s,
        "pca_fit_s": rep.pca_fit_s,
        "corpus_bytes": representation_bytes(n_corpus, dim),
        "vocabulary_bytes": rep.vocabulary_bytes,
        "pca_bytes": rep.pca_bytes,
        "auxiliary_bytes": rep.vocabulary_bytes + rep.pca_bytes,
        "model_weights_bytes": rep.model_weights_bytes,
        "query_sift_s": float(rep.query_sift_s.mean()),
        "query_encoding_s": float(rep.query_encoding_s.mean()),
        "query_extraction_s": float(query_extraction.mean()),
        "query_ranking_s": float(ranking_seconds.mean()),
        "query_total_s": float(query_extraction.mean() + ranking_seconds.mean()),
        "device": rep.device,
        "pca_explained_variance": rep.extra.get("pca_explained_variance"),
        "n_vocabulary_descriptors": rep.extra.get("n_vocabulary_descriptors"),
        "descriptor_dim": DESCRIPTOR_DIM if exp.method in ("bovw", "vlad", "vlad_pca") else None,
    })
    return row
