"""Sanity checks. Run from ``code/``: python -m unittest our_src.part5.stageB_C.test_part5"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .config import expand_group
from .data import resolve_image_path
from .metrics import average_precision, evaluate_ranking, interpolated_pr_curve, precision_at, recall_at
from .normalization import hellinger, l2, signed_sqrt, vlad_normalize
from .retrieval import random_score_fn, rank_queries, rank_scores


class MetricsTest(unittest.TestCase):
    TOY = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1])

    def test_toy_ranking(self) -> None:
        self.assertAlmostEqual(precision_at(self.TOY, 5), 0.4)
        self.assertAlmostEqual(recall_at(self.TOY, 10, 4), 1.0)
        self.assertAlmostEqual(average_precision(self.TOY, 4), (1 + 2 / 3 + 3 / 6 + 4 / 10) / 4)
        self.assertAlmostEqual(average_precision(self.TOY, 4), 0.6417, places=4)

    def test_ap_normalised_by_all_relevant(self) -> None:
        """A relevant image missing from the ranked list still lowers AP."""
        self.assertAlmostEqual(average_precision(np.array([1, 0]), 2), 0.5)

    def test_extremes(self) -> None:
        self.assertAlmostEqual(average_precision(np.array([1, 1, 0, 0]), 2), 1.0)
        self.assertAlmostEqual(average_precision(np.zeros(4), 2), 0.0)
        self.assertEqual(average_precision(np.zeros(4), 0), 0.0)

    def test_evaluate_ranking_keys(self) -> None:
        out = evaluate_ranking(self.TOY, 4, [5], [10])
        self.assertEqual(set(out), {"ap", "p_at_5", "r_at_10"})

    def test_pr_curve(self) -> None:
        curve = interpolated_pr_curve(self.TOY, 4, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        np.testing.assert_allclose(curve, [1.0, 1.0, 2 / 3, 0.5, 0.4])


class NormalizationTest(unittest.TestCase):
    def test_hellinger_has_unit_l2(self) -> None:
        out = hellinger(np.array([3, 0, 1, 12]))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)
        np.testing.assert_allclose(out, np.sqrt([3 / 16, 0, 1 / 16, 12 / 16]), rtol=1e-6)

    def test_signed_sqrt(self) -> None:
        np.testing.assert_allclose(signed_sqrt(np.array([-4.0, 0.0, 9.0])), [-2, 0, 3])

    def test_vlad_normalize_ignores_scale(self) -> None:
        """A prior global L2, as src's VLADEncoder applies, must not change the result."""
        raw = np.array([3.0, -4.0, 0.5, 2.0], dtype=np.float32)
        np.testing.assert_allclose(vlad_normalize(raw), vlad_normalize(l2(raw)), rtol=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(vlad_normalize(raw))), 1.0, places=6)

    def test_zero_vectors_stay_zero(self) -> None:
        for function in (l2, hellinger, vlad_normalize):
            np.testing.assert_array_equal(function(np.zeros(3, dtype=np.float32)), np.zeros(3))


class RetrievalTest(unittest.TestCase):
    def test_rank_descending_and_stable(self) -> None:
        np.testing.assert_array_equal(rank_scores(np.array([0.1, 0.9, 0.5, 0.9])), [1, 3, 2, 0])

    def test_random_is_reproducible_and_query_specific(self) -> None:
        fn = random_score_fn(0, 50)
        first, _, _ = rank_queries(fn, 2, repeats=2)
        second, _, _ = rank_queries(random_score_fn(0, 50), 2)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[0], first[1]))


class ConfigTest(unittest.TestCase):
    def test_pca_clamped_and_deduplicated(self) -> None:
        spec = {"method": "vlad_pca", "k": 32, "n_components": [8, 219, 500]}
        names = [e.name for e in expand_group("vlad_pca", spec, n_corpus=220)]
        self.assertEqual(names, ["vlad_k32_pca8", "vlad_k32_pca219"])

    def test_dinov2_input_size_checked(self) -> None:
        spec = {"method": "deep", "family": "dinov2", "model": "dinov2_vitb14", "input_size": 225}
        with self.assertRaises(ValueError):
            expand_group("dinov2", spec, 220)

    def test_text_requires_clip(self) -> None:
        spec = {"method": "deep", "query_type": "text", "family": "dinov2", "model": "x", "input_size": 224}
        with self.assertRaises(ValueError):
            expand_group("x", spec, 220)


class DataTest(unittest.TestCase):
    def test_windows_separators_resolved_against_csv_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a" / "b.jpg"
            target.parent.mkdir()
            target.touch()
            self.assertEqual(resolve_image_path("a\\b.jpg", Path(tmp)), target)
            self.assertEqual(resolve_image_path("a/b.jpg", Path(tmp)), target)
            with self.assertRaises(FileNotFoundError):
                resolve_image_path("a\\missing.jpg", Path(tmp))


class EncoderTest(unittest.TestCase):
    """Checks the wrappers around src.encoders against hand-computed encodings."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.words = rng.random((4, 128), dtype=np.float32) * 100
        self.descriptors = rng.random((200, 128), dtype=np.float32) * 100
        distances = ((self.descriptors[:, None] - self.words[None]) ** 2).sum(-1)
        self.assignments = distances.argmin(1)

    def test_bovw(self) -> None:
        from .encoders import ImageEncoder

        counts = np.bincount(self.assignments, minlength=4)
        out = ImageEncoder("bovw", self.words).encode(self.descriptors)
        np.testing.assert_allclose(out, np.sqrt(counts / counts.sum()), rtol=1e-5)

    def test_vlad(self) -> None:
        from .encoders import ImageEncoder

        residuals = np.zeros((4, 128), dtype=np.float32)
        for word in range(4):
            residuals[word] = (self.descriptors[self.assignments == word] - self.words[word]).sum(0)
        expected = vlad_normalize(residuals.ravel())
        out = ImageEncoder("vlad", self.words).encode(self.descriptors)
        self.assertEqual(out.shape, (512,))
        np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_pca_output_is_unit_norm(self) -> None:
        from .encoders import VladPCA

        corpus = l2(np.random.default_rng(1).normal(size=(10, 32)).astype(np.float32))
        pca = VladPCA(9, seed=0).fit(corpus)
        self.assertEqual(pca.transform(corpus).shape, (10, 9))
        np.testing.assert_allclose(np.linalg.norm(pca.transform(corpus), axis=1), 1.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
