"""Reproducible sampling from the internal train partition only."""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def subset(train_labels, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('Fractions must be in (0,1]')
    ids = np.array(sorted(train_labels))
    labels = np.array([train_labels[i] for i in ids])
    if set(labels) != {-1, 1}:
        raise ValueError('Train must contain both classes')
    if fraction == 1:
        return ids.tolist()
    count = int(np.floor(len(ids)*fraction))
    if count < 2 or count > len(ids)-2:
        raise ValueError('Fraction leaves too few samples for stratified sampling')
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=seed)
    selected, _ = next(splitter.split(ids, labels))
    chosen = sorted(ids[selected].tolist())
    if {train_labels[i] for i in chosen} != {-1, 1}:
        raise ValueError('Requested fraction did not retain both classes')
    return chosen
