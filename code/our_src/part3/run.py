"""Part 3 entry point: large vocabularies + PCA vs small vocabularies.

Run from the ``code/`` directory:

    python -m our_src.part3.run --config our_src/part3/config.yaml [--seeds 0 1] [--force] [--smoke]
    python -m our_src.part3.run --config our_src/part3/config.yaml --clean-cache
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .classification import evaluate, train_svm
from .config import PROJECT_ROOT, Config, Experiment, load_config
from .data import Dataset, load_dataset, task_mask
from .encoders import Codebook, encode_chunk, encoding_dim, fingerprint, fit_codebook
from .normalization import apply_steps
from .pca_variance import run_pca_variance
from .reduction import PCAReducer, pca_invalid_reason
from .results import experiment_key, load_done, upsert_row, write_summary
from .sift_cache import cache_file, ensure_sift_cache, sample_descriptors
from .timing import Timer

BYTES_PER_VALUE = 4  # features are stored as float32

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------------- setup
def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("part3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(out_dir / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def write_run_json(cfg: Config, state: str, extra: dict | None = None) -> None:
    import sklearn, skimage  # noqa: E401
    info = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv, "python": platform.python_version(),
            "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "skimage": skimage.__version__,
                         "pandas": pd.__version__},
            "config": cfg.to_dict(), **(extra or {})}
    (cfg.paths.output_dir / "run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


# ----------------------------------------------------------------------------- cache keys
def codebook_fingerprint(cfg: Config, dataset: Dataset, rep: str, k: int, seed: int) -> str:
    section = cfg.codebook.gmm if rep == "fisher" else cfg.codebook.kmeans
    return fingerprint({"rep": rep, "k": k, "seed": seed, "section": section,
                        "n_descriptors": cfg.codebook.n_descriptors, "train_ids": dataset.train_ids})


def encoding_fingerprint(cfg: Config, dataset: Dataset, rep: str, k: int, seed: int) -> str:
    return fingerprint({"codebook": codebook_fingerprint(cfg, dataset, rep, k, seed), "ids": dataset.all_ids,
                        "cap": cfg.encoding.max_descriptors_per_image,
                        "fisher_weights": cfg.encoding.fisher_include_weight_grads})


def codebook_path(cfg, dataset, rep, k, seed) -> Path:
    fp = codebook_fingerprint(cfg, dataset, rep, k, seed)
    return cfg.paths.cache_dir / "codebooks" / f"{rep}_K{k}_seed{seed}_{fp}.npz"


def encoding_path(cfg, dataset, rep, k, seed) -> Path:
    fp = encoding_fingerprint(cfg, dataset, rep, k, seed)
    return cfg.paths.cache_dir / "encodings" / f"{rep}_K{k}_seed{seed}_{fp}.npz"


# ----------------------------------------------------------------------------- codebooks / encodings
def get_codebooks(cfg: Config, dataset: Dataset, seed: int, needed: list[tuple[str, int]],
                  logger: logging.Logger) -> dict[tuple[str, int], Codebook]:
    """Load cached codebooks of this seed; fit the missing ones on one shared descriptor sample."""
    books: dict[tuple[str, int], Codebook] = {}
    missing = []
    for rep, k in needed:
        path = codebook_path(cfg, dataset, rep, k, seed)
        if path.exists():
            books[(rep, k)] = Codebook.load(path)
        else:
            missing.append((rep, k))
    if not missing:
        return books
    with Timer() as t:
        descriptors = sample_descriptors(dataset.train_ids, cfg.paths.cache_dir / "sift",
                                         cfg.codebook.n_descriptors, seed, cfg.n_jobs)
    logger.info("Seed %d: sampled %d descriptors from %d train images in %.1fs",
                seed, len(descriptors), len(dataset.train_ids), t.elapsed)
    for rep, k in missing:
        book = fit_codebook(rep, k, descriptors, seed, cfg.codebook)
        book.sampling_seconds = t.elapsed
        book.save(codebook_path(cfg, dataset, rep, k, seed))
        books[(rep, k)] = book
        logger.info("Seed %d: fitted %s K=%d codebook in %.1fs", seed, rep, k, book.fit_seconds)
    return books


def get_encodings(cfg: Config, dataset: Dataset, seed: int, books: dict[tuple[str, int], Codebook],
                  logger: logging.Logger) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """Raw (un-normalized) encodings of all images (train first, then test) and per-image times.

    Missing encodings are computed in a single parallel pass: every image's descriptors are
    loaded once and encoded with all missing codebooks.
    """
    out, missing = {}, []
    for key in books:
        path = encoding_path(cfg, dataset, *key, seed)
        if path.exists():
            with np.load(path) as data:
                out[key] = (data["features"], data["times"])
        else:
            missing.append(key)
    if not missing:
        return out

    ids = dataset.all_ids
    sift_dir = cfg.paths.cache_dir / "sift"
    paths = [str(cache_file(sift_dir, i)) for i in ids]
    cs = cfg.chunk_size
    chunks = [(paths[s:s + cs], list(range(s, min(s + cs, len(ids))))) for s in range(0, len(ids), cs)]
    codebooks = [books[key] for key in missing]
    logger.info("Seed %d: encoding %d images with %s", seed, len(ids),
                ", ".join(f"{r} K={k}" for r, k in missing))
    parts = []
    with Timer() as t:
        gen = Parallel(n_jobs=cfg.n_jobs, return_as="generator")(
            delayed(encode_chunk)(p, idx, codebooks, cfg.encoding.fisher_include_weight_grads,
                                  cfg.encoding.max_descriptors_per_image, seed) for p, idx in chunks)
        step = max(1, len(chunks) // 10)
        for n, part in enumerate(gen, 1):
            parts.append(part)
            if n % step == 0 or n == len(chunks):
                logger.info("  encoded %d/%d images (%.0fs)", min(n * cs, len(ids)), len(ids), t.elapsed)
    for e, key in enumerate(missing):
        feats = np.vstack([p[e][0] for p in parts])
        times = np.concatenate([p[e][1] for p in parts])
        path = encoding_path(cfg, dataset, *key, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as stream:
            np.savez(stream, features=feats, times=times, ids=np.asarray(ids))
        os.replace(tmp, path)
        out[key] = (feats, times)
    return out


# ----------------------------------------------------------------------------- one experiment
def run_experiment(cfg: Config, dataset: Dataset, exp: Experiment, seed: int, book: Codebook,
                   raw: np.ndarray, enc_times: np.ndarray, sift_times: dict[str, float],
                   logger: logging.Logger) -> dict:
    n_train, n_test = len(dataset.train_ids), len(dataset.test_ids)
    norm = cfg.normalization
    x_train = apply_steps(raw[:n_train], norm.post_encoding, norm.sqrt_alpha)
    x_test = apply_steps(raw[n_train:], norm.post_encoding, norm.sqrt_alpha)
    original_dim = x_train.shape[1]

    reducer = None
    if exp.pca:
        reducer = PCAReducer(exp.pca_dim, cfg.pca, seed).fit(x_train)
        x_train = apply_steps(reducer.transform(x_train), norm.post_pca, norm.sqrt_alpha)
        x_test = apply_steps(reducer.transform(x_test), norm.post_pca, norm.sqrt_alpha)
    final_dim = x_train.shape[1]

    row: dict = {"representation": exp.representation, "pca": exp.pca, "k": exp.k, "pca_dim": exp.pca_dim,
                 "seed": seed, "original_dim": original_dim, "final_dim": final_dim,
                 "compaction_pct": (1 - final_dim / original_dim) * 100 if exp.pca else 0.0,
                 "n_train": n_train, "n_test": n_test}

    models, aps, accs, f1s = {}, [], [], []
    t_cv = t_fit = 0.0
    n_conv = 0
    for target in cfg.data.target_classes:
        m_tr = task_mask(dataset, dataset.train_ids, target, cfg.data.negatives)
        m_te = task_mask(dataset, dataset.test_ids, target, cfg.data.negatives)
        y_tr = dataset.binary_labels(dataset.train_ids, target)[m_tr]
        y_te = dataset.binary_labels(dataset.test_ids, target)[m_te]
        svm = train_svm(x_train[m_tr], y_tr, cfg.svm, seed, logger)
        metrics = evaluate(svm.model, x_test[m_te], y_te)
        models[target] = svm.model
        t_cv += svm.cv_seconds
        t_fit += svm.fit_seconds
        n_conv += svm.convergence_warnings
        aps.append(metrics["ap"]); accs.append(metrics["accuracy"]); f1s.append(metrics["macro_f1"])
        row.update({f"ap_{target}": metrics["ap"], f"accuracy_{target}": metrics["accuracy"],
                    f"macro_f1_{target}": metrics["macro_f1"], f"n_train_{target}": int(m_tr.sum()),
                    f"n_train_pos_{target}": int(y_tr.sum()), f"n_test_{target}": int(m_te.sum()),
                    f"n_test_pos_{target}": int(y_te.sum()), f"cv_score_{target}": svm.best_cv_score,
                    f"best_params_{target}": json.dumps(svm.best_params)})
    # Mean over target classes. With a single target class, map == its AP and
    # test_accuracy / macro_f1 equal that class's values.
    row.update({"test_accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)), "map": float(np.mean(aps))})

    # Inference time, one test image at a time: encoding (measured per image during the encoding
    # pass) + normalization + PCA + SVM predict for every target class. SIFT is excluded.
    test_raw, test_enc = raw[n_train:], enc_times[n_train:]
    n_inf = n_test if cfg.inference_max_images is None else min(n_test, int(cfg.inference_max_images))
    per_image = np.empty(n_inf)
    for i in range(n_inf):
        with Timer() as t:
            x = apply_steps(test_raw[i:i + 1], norm.post_encoding, norm.sqrt_alpha)
            if reducer is not None:
                x = apply_steps(reducer.transform(x), norm.post_pca, norm.sqrt_alpha)
            for model in models.values():
                model.predict(x)
        per_image[i] = test_enc[i] + t.elapsed

    sift = [sift_times[i] for i in dataset.all_ids if i in sift_times]
    n_all = n_train + n_test
    row.update({
        "time_sift_extraction": float(np.sum(sift)) if len(sift) == n_all else float("nan"),
        "time_sift_extraction_per_image": float(np.mean(sift)) if sift else float("nan"),
        "time_descriptor_sampling": book.sampling_seconds,
        "time_codebook": book.fit_seconds,
        "time_encoding": float(enc_times.sum()),
        "time_encoding_per_image": float(enc_times.mean()),
        "time_pca": reducer.fit_seconds if reducer else 0.0,
        "time_svm_cv": t_cv, "time_svm_fit": t_fit, "time_svm_train": t_cv + t_fit,
        "time_inference_per_image": float(per_image.mean()),
        "feature_storage_bytes_per_image": final_dim * BYTES_PER_VALUE,
        "feature_storage_mb_total": final_dim * BYTES_PER_VALUE * n_all / 1e6,
        "experiment": exp.name, "negatives": cfg.data.negatives, "svm_kernel": cfg.svm.kernel,
        "post_encoding": "+".join(norm.post_encoding) or "none", "post_pca": "+".join(norm.post_pca) or "none",
        "codebook_n_descriptors": book.n_descriptors, "svm_convergence_warnings": n_conv,
        "n_inference_images": n_inf,
    })
    return row


# ----------------------------------------------------------------------------- main loop
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the seeds of the config")
    parser.add_argument("--force", action="store_true", help="Recompute rows already in metrics.csv")
    parser.add_argument("--smoke", action="store_true", help="Tiny smoke test written to <output_dir>/smoke")
    parser.add_argument("--clean-cache", action="store_true", help="Delete the cache directory and exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, smoke=args.smoke, seeds=args.seeds)

    if args.clean_cache:
        cache = cfg.paths.cache_dir.resolve()
        if PROJECT_ROOT.resolve() not in cache.parents:
            raise SystemExit(f"Refusing to delete a cache outside the project: {cache}")
        if cache.exists():
            shutil.rmtree(cache)
            print(f"Deleted {cache}")
        else:
            print(f"Nothing to delete: {cache}")
        return 0

    out = cfg.paths.output_dir
    logger = setup_logging(out)
    logger.info("=" * 70)
    logger.info("Part 3 run%s, seeds=%s, force=%s", " (SMOKE)" if cfg.smoke else "", cfg.seeds, args.force)
    write_run_json(cfg, "running")

    dataset = load_dataset(cfg)
    logger.info("Images: %d train (%s), %d test (%s); targets=%s, negatives=%s",
                len(dataset.train_ids), "+".join(cfg.data.train_splits), len(dataset.test_ids),
                "+".join(cfg.data.test_splits), cfg.data.target_classes, cfg.data.negatives)

    metrics_path = out / "metrics.csv"
    done = set() if args.force else load_done(metrics_path)
    n_train = len(dataset.train_ids)  # PCA is fitted on all encoded train images
    fisher_w = cfg.encoding.fisher_include_weight_grads
    pv_dir = out / "pca_variance"
    pv_seed = cfg.seeds[0]

    # Validity checks before any computation; skipped experiments never crash the run.
    skipped, plan = [], defaultdict(list)
    for seed in cfg.seeds:
        for exp in cfg.experiments:
            if experiment_key(exp, seed) in done:
                continue
            if exp.pca:
                reason = pca_invalid_reason(exp.pca_dim, encoding_dim(exp.representation, exp.k, fisher_w), n_train)
                if reason:
                    logger.warning("SKIP %s seed=%d: %s", exp.name, seed, reason)
                    skipped.append({"experiment": exp.name, "representation": exp.representation, "k": exp.k,
                                    "pca_dim": exp.pca_dim, "seed": seed, "reason": reason})
                    continue
            plan[seed].append(exp)
    pd.DataFrame(skipped, columns=["experiment", "representation", "k", "pca_dim", "seed", "reason"]).to_csv(
        out / "skipped_experiments.csv", index=False)

    pv_todo = []
    if cfg.pca_variance.enabled:
        pv_todo = [(r, k) for r, k in cfg.pca_variance.targets
                   if args.force or not (pv_dir / f"{r}_K{k}_seed{pv_seed}.png").exists()]

    n_pending = sum(len(v) for v in plan.values())
    logger.info("%d experiment rows to run (%d already done, %d skipped)", n_pending,
                len(done), len(skipped))
    if not n_pending and not pv_todo:
        write_summary(metrics_path, out / "metrics_summary.csv")
        write_run_json(cfg, "complete", {"skipped": skipped})
        logger.info("Nothing to do.")
        return 0

    with Timer() as t_sift:
        sift_times = ensure_sift_cache(dataset.all_ids, dataset.image_path, cfg.paths.cache_dir / "sift",
                                       cfg.n_jobs, logger)
    logger.info("SIFT stage took %.1fs (sum of per-image extraction times: %.1fs)",
                t_sift.elapsed, sum(sift_times.values()))

    pv_rows = []
    for seed in cfg.seeds:
        exps = plan.get(seed, [])
        pv_here = pv_todo if seed == pv_seed else []
        needed = sorted({(e.representation, e.k) for e in exps} | set(pv_here),
                        key=lambda rk: (["bovw", "vlad", "fisher"].index(rk[0]), rk[1]))
        if not needed:
            continue
        logger.info("-" * 70)
        logger.info("Seed %d: %d experiments, codebooks %s", seed, len(exps), needed)
        books = get_codebooks(cfg, dataset, seed, needed, logger)
        encodings = get_encodings(cfg, dataset, seed, books, logger)

        for rep, k in pv_here:
            raw, _ = encodings[(rep, k)]
            x = apply_steps(raw[:len(dataset.train_ids)], cfg.normalization.post_encoding,
                            cfg.normalization.sqrt_alpha)
            used = sorted(e.pca_dim for e in cfg.experiments if e.representation == rep and e.k == k and e.pca)
            pv_rows.append(run_pca_variance(rep, k, seed, x, used, cfg.pca_variance.thresholds, pv_dir, logger))

        for exp in exps:
            raw, times = encodings[(exp.representation, exp.k)]
            with Timer() as t:
                row = run_experiment(cfg, dataset, exp, seed, books[(exp.representation, exp.k)], raw, times,
                                     sift_times, logger)
            upsert_row(metrics_path, row)
            logger.info("Seed %d %-18s dim %5d -> %4d  mAP=%.4f acc=%.4f macroF1=%.4f  (%.1fs)", seed, exp.name,
                        row["original_dim"], row["final_dim"], row["map"], row["test_accuracy"], row["macro_f1"],
                        t.elapsed)

    if pv_rows:
        summary_path = pv_dir / "summary.csv"
        old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        new = pd.DataFrame(pv_rows)
        if not old.empty:
            keys = set(zip(new.representation, new.k, new.seed))
            old = old[[(r, k, s) not in keys for r, k, s in zip(old.representation, old.k, old.seed)]]
        pd.concat([old, new], ignore_index=True).to_csv(summary_path, index=False)

    write_summary(metrics_path, out / "metrics_summary.csv")
    write_run_json(cfg, "complete", {"skipped": skipped})
    logger.info("Done. Results in %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
