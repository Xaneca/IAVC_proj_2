"""Encoding-independent linear SVM selection, reusing src.utils.train_svm."""
import time
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import normalize
from src.utils import train_svm


def transform_features(x, normalization):
    x = np.asarray(x, dtype=np.float32)
    if normalization == 'none':
        return x
    if normalization not in ('l1', 'l2'):
        raise ValueError('normalization must be none, l1 or l2')
    return normalize(x, norm=normalization)


def train_svm_grid(features, train, validation, param_grid, test=None, seed=42):
    """Select by validation AP from any fixed-size encoding.

    features: mapping image ID -> finite 1-D vector (any encoding).
    train/validation/test: mapping image ID -> binary label (-1 or +1).
    param_grid: sklearn ParameterGrid dict/list; C, normalization, max_iter, tol.
    Optional test is scored once, after selecting parameters on validation.
    Returns selected weights/bias, every validation trial and optional test AP.
    There is no refit on validation and no automatic feature/model persistence.
    """
    partitions = [train, validation] + ([] if test is None else [test])
    for i, part in enumerate(partitions):
        if not part:
            raise ValueError('Partitions must be nonempty')
        if any(set(part) & set(other) for other in partitions[:i]):
            raise ValueError('Image IDs overlap between train/validation/test')
        if set(part.values()) != {-1, 1}:
            raise ValueError('Each partition must contain both -1 and +1 labels')
    def arrays(part):
        ids = sorted(part)
        x = np.stack([features[i] for i in ids]).astype(np.float32)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Features must be finite fixed-size vectors')
        return x, np.array([part[i] for i in ids])
    x_train, y_train = arrays(train)
    x_val, y_val = arrays(validation)
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError('Encoding dimensions differ across partitions')
    trials, selected = [], None
    for params in ParameterGrid(param_grid):
        unknown = set(params) - {'C', 'normalization', 'max_iter', 'tol'}
        if unknown:
            raise ValueError(f'Unsupported SVM parameters: {unknown}')
        p = dict(C=1., normalization='l2', max_iter=20000, tol=1e-4)
        p.update(params)
        if p['C'] <= 0 or p['max_iter'] < 1 or p['tol'] <= 0:
            raise ValueError('C, max_iter and tol must be positive')
        a = transform_features(x_train, p['normalization'])
        b = transform_features(x_val, p['normalization'])
        start = time.perf_counter()
        w, bias = train_svm(a, y_train, C=p['C'], max_iter=p['max_iter'],
                            tol=p['tol'], seed=seed)
        elapsed = time.perf_counter() - start
        start = time.perf_counter()
        scores = b @ w + bias
        inference = time.perf_counter() - start
        trial = {**p, 'train_ap': float(average_precision_score(y_train, a @ w + bias)),
                 'validation_ap': float(average_precision_score(y_val, scores)),
                 'fit_seconds': elapsed, 'validation_inference_seconds': inference,
                 'train_images': len(train), 'validation_images': len(validation)}
        trials.append(trial)
        # Deterministic first-wins tie break in ParameterGrid order.
        if selected is None or trial['validation_ap'] > selected['validation_ap']:
            selected = {**trial, 'weights': w.copy(), 'bias': float(bias)}
    if selected is None:
        raise ValueError('No hyperparameter candidates')
    test_ap = None
    if test is not None:
        x, y = arrays(test)
        if x.shape[1] != x_train.shape[1]:
            raise ValueError('Test encoding dimension differs')
        scores = transform_features(x, selected['normalization']) @ selected['weights'] + selected['bias']
        test_ap = float(average_precision_score(y, scores))
    return {'best': selected, 'trials': trials, 'test_ap': test_ap}
