"""Minimal sanity tests: python -m unittest our_src.part3.test_part3 (from code/)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .config import Experiment, expand_experiments, load_config
from .encoders import Codebook, GlobalEncoder, encoding_dim
from .normalization import apply_steps
from .reduction import PCAReducer, components_for_variance, pca_invalid_reason
from .results import FIRST_COLUMNS, load_done, upsert_row, write_summary
from .config import PCAConfig

CONFIG = Path(__file__).with_name("config.yaml")


class NormalizationTest(unittest.TestCase):
    def test_sqrt_l2_unit_norm(self):
        x = np.random.default_rng(0).normal(size=(20, 50))
        out = apply_steps(x, ["sqrt", "l2"])
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)

    def test_signed_sqrt_keeps_sign(self):
        out = apply_steps(np.array([[-4.0, 9.0, 0.0]]), ["sqrt"])
        np.testing.assert_allclose(out, [[-2.0, 3.0, 0.0]])

    def test_zero_rows_stay_zero_and_l1(self):
        out = apply_steps(np.array([[0.0, 0.0], [1.0, 3.0]]), ["sqrt", "l1"])
        np.testing.assert_allclose(out[0], 0.0)
        self.assertAlmostEqual(float(np.abs(out[1]).sum()), 1.0, places=6)

    def test_unknown_step(self):
        with self.assertRaises(ValueError):
            apply_steps(np.ones((1, 2)), ["l3"])


class PCATest(unittest.TestCase):
    def test_constraints(self):
        self.assertIsNone(pca_invalid_reason(128, 1024, 2472))
        self.assertIsNotNone(pca_invalid_reason(1024, 1024, 2472))      # not < original_dim
        self.assertIsNotNone(pca_invalid_reason(2048, 1024, 5000))
        self.assertIsNotNone(pca_invalid_reason(30, 4096, 30))          # > n_train - 1
        self.assertIsNone(pca_invalid_reason(29, 4096, 30))

    def test_fit_dims_and_refuses_invalid(self):
        x = np.random.default_rng(0).normal(size=(40, 100))
        cfg = PCAConfig(whiten=False, svd_solver="full", full_max_size=10_000)
        self.assertEqual(PCAReducer(10, cfg, 0).fit(x).transform(x).shape, (40, 10))
        with self.assertRaises(ValueError):
            PCAReducer(40, cfg, 0).fit(x)

    def test_components_for_variance(self):
        cum = np.array([0.5, 0.8, 0.9, 0.95, 1.0])
        self.assertEqual(components_for_variance(cum, 0.8), 2)
        self.assertEqual(components_for_variance(cum, 0.85), 3)
        self.assertEqual(components_for_variance(cum, 0.95), 4)


class EncoderDimTest(unittest.TestCase):
    def test_dims(self):
        rng = np.random.default_rng(0)
        desc = rng.integers(0, 255, size=(300, 128)).astype(np.float32)
        k = 4
        words = desc[:k].copy()
        for rep, params in [("bovw", {"words": words}), ("vlad", {"words": words}),
                            ("fisher", {"means": desc[:k].astype(np.float64), "covariances": np.full((k, 128), 100.0),
                                        "priors": np.full(k, 1 / k)})]:
            enc = GlobalEncoder(Codebook(rep, k, 0, params, 0.0, 300))
            self.assertEqual(enc.encode(desc).shape, (encoding_dim(rep, k),))
        self.assertEqual(encoding_dim("bovw", 512), 512)
        self.assertEqual(encoding_dim("vlad", 32), 32 * 128)
        self.assertEqual(encoding_dim("fisher", 32), 2 * 32 * 128)


class ConfigTest(unittest.TestCase):
    def test_default_config_experiments(self):
        cfg = load_config(CONFIG)
        exps = cfg.experiments
        self.assertEqual(len([e for e in exps if e.representation == "bovw"]), 4 + 4 + 3)
        self.assertEqual(len([e for e in exps if e.representation == "vlad"]), 2 + 4)
        self.assertEqual(len([e for e in exps if e.representation == "fisher"]), 2 + 4)
        self.assertIn(Experiment("bovw", 1024, 64), exps)
        self.assertEqual(cfg.normalization.post_encoding, ["sqrt", "l2"])
        self.assertEqual(cfg.seeds, [0, 1, 2])

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            expand_experiments({"bovw": {"no_pca_k": [128, 128]}})


class MetricsCsvTest(unittest.TestCase):
    def _row(self, seed, pca_dim=0):
        row = {c: 0 for c in FIRST_COLUMNS}
        row.update({"representation": "bovw", "pca": pca_dim > 0, "k": 512, "pca_dim": pca_dim, "seed": seed,
                    "map": 0.5 + seed / 10, "ap_motorbike": 0.5, "time_codebook": 1.0, "experiment": "x"})
        return row

    def test_columns_resume_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            for seed in (0, 1):
                upsert_row(path, self._row(seed))
            upsert_row(path, self._row(0))           # same key -> replaced, not duplicated
            upsert_row(path, self._row(0, pca_dim=128))
            header = path.read_text().splitlines()[0].split(",")
            self.assertEqual(header[:5], ["representation", "pca", "k", "pca_dim", "seed"])
            self.assertEqual(header[:len(FIRST_COLUMNS)], FIRST_COLUMNS)
            self.assertEqual(len(load_done(path)), 3)
            self.assertIn(("bovw", False, 512, 0, 1), load_done(path))
            summary = write_summary(path, Path(tmp) / "summary.csv")
            row = summary[summary.pca_dim == 0].iloc[0]
            self.assertEqual(row["n_seeds"], 2)
            self.assertAlmostEqual(row["map_mean"], 0.55)


if __name__ == "__main__":
    unittest.main()
