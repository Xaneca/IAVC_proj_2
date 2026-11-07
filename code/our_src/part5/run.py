"""Part 5 entry point: image retrieval experiments (Stages B and C).

Run from the ``code/`` directory:

    python -m our_src.part5.run                        # everything in the config
    python -m our_src.part5.run --only bovw vlad       # only these experiment groups
    python -m our_src.part5.run --smoke                # small subset, in <output_dir>/smoke
    python -m our_src.part5.run --force                # recompute features instead of reusing the cache
    python -m our_src.part5.run --figures-only         # regenerate figures from the saved rankings
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .stageB_C.config import Config, load_config
from .stageB_C.data import Dataset, load_dataset, subset_dataset
from .stageB_C.results import PER_QUERY_FILE, SUMMARY_FILE, save_ranking, upsert_experiment


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("part5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(output_dir / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_run_json(cfg: Config, state: str, device: str, versions: dict) -> None:
    info = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
            "python": platform.python_version(), "numpy": np.__version__, "device": device, **versions,
            "config": cfg.raw}
    (cfg.output_dir / "run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


def load_data(cfg: Config) -> Dataset:
    dataset = load_dataset(cfg.corpus_csv, cfg.queries_csv, cfg.text_queries_csv)
    if cfg.smoke:
        dataset = subset_dataset(dataset, cfg.smoke["corpus_per_label"], cfg.smoke["n_queries"],
                                 cfg.smoke["n_text_queries"])
    return dataset


def run_experiments(cfg: Config, dataset: Dataset, only: list[str] | None, force: bool, logger: logging.Logger) -> None:
    # Imported here so --figures-only needs neither torch nor the src package.
    from .stageB_C.deep import library_versions, resolve_device, seed_torch
    from .stageB_C.pipeline import FeatureBuilder, per_query_rows, run_rankings, summary_row

    device = resolve_device(cfg.device)
    versions = library_versions()
    seed_torch(cfg.seed)
    np.random.seed(cfg.seed)
    write_run_json(cfg, "running", device, versions)

    n_corpus = len(dataset.corpus)
    experiments = cfg.experiments(n_corpus)
    order = [exp.name for exp in experiments]
    selected = [exp for exp in experiments if not only or exp.group in only]
    logger.info("Corpus: %d images (%d relevant); %d image queries, %d text queries; device=%s",
                n_corpus, dataset.n_relevant, len(dataset.queries), len(dataset.text_queries), device)
    logger.info("Running %d of %d experiments", len(selected), len(experiments))

    builder = FeatureBuilder(cfg, dataset, device, logger, force=force)
    corpus_ids = dataset.corpus["id"].to_numpy()
    labels = dataset.corpus["label"].to_numpy()
    for exp in selected:
        rep = builder.build(exp)
        query_ids = dataset.query_ids(exp.query_type)
        orders, scores, ranking_seconds = run_rankings(cfg, exp, rep, n_corpus, len(query_ids))
        rows = per_query_rows(cfg, dataset, exp, orders)
        summary = summary_row(cfg, exp, rep, rows, ranking_seconds, n_corpus)
        save_ranking(cfg.output_dir, exp.name, query_ids, corpus_ids, labels, dataset.relevance, orders, scores)
        upsert_experiment(cfg.output_dir / PER_QUERY_FILE, rows, order)
        upsert_experiment(cfg.output_dir / SUMMARY_FILE, [summary], order)
        logger.info("%-24s dim=%-6d mAP=%.4f (std %.4f)  query=%.1f ms", exp.name, summary["embedding_dim"],
                    summary["map"], summary["ap_std"], summary["query_total_s"] * 1e3)
    write_run_json(cfg, "complete", device, versions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--only", nargs="+", metavar="GROUP", help="Experiment groups to run (see the config)")
    parser.add_argument("--smoke", action="store_true", help="Small subset, written to <output_dir>/smoke")
    parser.add_argument("--force", action="store_true", help="Recompute features instead of reusing the cache")
    parser.add_argument("--figures-only", action="store_true", help="Only regenerate figures from saved results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, smoke=args.smoke)
    if args.only:
        unknown = sorted(set(args.only) - set(cfg.groups))
        if unknown:
            parser.error(f"unknown groups {unknown}; available: {sorted(cfg.groups)}")

    logger = setup_logging(cfg.output_dir)
    logger.info("=" * 70)
    logger.info("Part 5 run%s, only=%s, force=%s", " (SMOKE)" if cfg.smoke else "", args.only, args.force)
    dataset = load_data(cfg)
    if not args.figures_only:
        run_experiments(cfg, dataset, args.only, args.force, logger)

    from .stageB_C.visualization import make_all_figures

    make_all_figures(cfg, dataset, logger)
    logger.info("Done. Results in %s", cfg.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
