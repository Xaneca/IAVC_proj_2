"""One-vs-rest SVM with stratified k-fold CV on TRAIN only, plus test metrics."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC

from .config import SVMConfig
from .timing import Timer


@dataclass
class SVMResult:
    model: Any
    best_params: dict[str, Any]
    best_cv_score: float
    cv_folds: int
    cv_seconds: float
    fit_seconds: float
    convergence_warnings: int

    @property
    def train_seconds(self) -> float:
        return self.cv_seconds + self.fit_seconds


def _estimator_and_grid(cfg: SVMConfig, seed: int):
    if cfg.kernel == "linear":
        est = LinearSVC(dual="auto", max_iter=cfg.max_iter, class_weight=cfg.class_weight, random_state=seed)
        grid = {"C": [float(c) for c in cfg.C_grid]}
    else:
        est = SVC(kernel="rbf", max_iter=-1, class_weight=cfg.class_weight, random_state=seed)
        grid = {"C": [float(c) for c in cfg.C_grid],
                "gamma": [g if isinstance(g, str) else float(g) for g in cfg.gamma_grid]}
    return est, grid


def train_svm(x: np.ndarray, y: np.ndarray, cfg: SVMConfig, seed: int,
              logger: logging.Logger | None = None) -> SVMResult:
    """Choose hyperparameters by stratified k-fold CV on (x, y), then refit on all of (x, y).

    The CV search and the final fit are timed separately.
    """
    n_min = int(np.bincount(y, minlength=2).min())
    if n_min < 2:
        raise ValueError(f"Need at least 2 samples per class for CV, got {np.bincount(y, minlength=2)}")
    folds = min(cfg.cv_folds, n_min)
    if folds < cfg.cv_folds and logger:
        logger.warning("Reducing CV folds from %d to %d (only %d samples in the minority class)",
                       cfg.cv_folds, folds, n_min)
    est, grid = _estimator_and_grid(cfg, seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(est, grid, scoring=cfg.cv_scoring, cv=cv, refit=False, n_jobs=cfg.cv_n_jobs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with Timer() as t_cv:
            search.fit(x, y)
        model = clone(est).set_params(**search.best_params_)
        with Timer() as t_fit:
            model.fit(x, y)
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    if n_conv and logger:
        logger.warning("SVM: %d ConvergenceWarning(s) (max_iter=%d)", n_conv, cfg.max_iter)
    return SVMResult(model, dict(search.best_params_), float(search.best_score_), folds,
                     t_cv.elapsed, t_fit.elapsed, n_conv)


def evaluate(model: Any, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Test metrics of a binary problem.

    - accuracy and macro F1 use ``predict`` (SVM threshold 0); macro F1 averages the F1 of the
      positive and negative class.
    - AP uses the continuous ``decision_function`` scores.
    """
    scores = model.decision_function(x)
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "ap": float(average_precision_score(y, scores)),
    }
