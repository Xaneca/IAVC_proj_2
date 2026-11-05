"""Experiment loop, independent of image extraction."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, accuracy_score

TILINGS = {'global': 'normal', '2x2': 'flat_2x2', 'pyramid': 'pyramid_L2'}
NORMALIZATIONS = ('l1', 'l2', 'none')


def train_models(rows, categories, feature_builder, train_grid, labels_for,
                 config, output, provenance):
    """Save nine selected models/class; C selection sees train/val only.

    feature_builder(layout, norm) supplies already normalized feature vectors.
    Exactly one combination is held in memory at a time.
    """
    if not categories or len(set(categories)) != len(categories):
        raise ValueError('Expected unique, nonempty categories')
    output = Path(output)
    (output/'models').mkdir(parents=True, exist_ok=True)
    reports = []
    for tiling, layout in TILINGS.items():
        for norm in NORMALIZATIONS:
            features = feature_builder(layout, norm)
            for category in categories:
                partitions = {s: labels_for(rows, s, category) for s in ('train', 'val', 'test')}
                result = train_grid(features, partitions['train'], partitions['val'],
                    dict(C=config['C_values'], normalization=['none'],
                         max_iter=[config['max_iter']], tol=[config['tol']]), seed=config['seed'])
                best = result['best']
                name = f'{category}__{tiling}__{norm}.npz'
                metrics = dict(category=category, negative_class='background', tiling=tiling,
                    layout=layout, normalization=norm, k=provenance['k'],
                    dimension=len(best['weights']), C=best['C'],
                    fit_seconds=best['fit_seconds'], model=f'models/{name}',
                    split_sha256=provenance['split_sha256'])
                for split, labels in partitions.items():
                    ids = sorted(labels)
                    if set(labels.values()) != {-1, 1}:
                        raise ValueError(f'{category}/{split} must contain both labels')
                    scores = np.stack([features[i] for i in ids]) @ best['weights'] + best['bias']
                    y = np.array([labels[i] for i in ids])
                    if not np.isfinite(scores).all():
                        raise ValueError('Nonfinite SVM scores')
                    metrics[f'{split}_ap'] = float(average_precision_score(y, scores))
                    metrics[f'{split}_images'] = len(ids)
                    if split == 'test':
                        predicted = np.where(scores >= 0, 1, -1)
                        p, r, f, _ = precision_recall_fscore_support(y, predicted,
                            average='binary', pos_label=1, zero_division=0)
                        metrics.update(test_precision=float(p), test_recall=float(r),
                            test_f1=float(f), test_accuracy=float(accuracy_score(y, predicted)))
                metadata = dict(**provenance, category=category, negative_class='background',
                    tiling=tiling, layout=layout, normalization=norm, C=best['C'],
                    seed=config['seed'], max_iter=config['max_iter'], tol=config['tol'],
                    training_split='train', selection_split='val', test_used_for_selection=False,
                    feature_normalization_already_applied=True)
                np.savez_compressed(output/'models'/name, weights=best['weights'],
                                    bias=best['bias'], metadata=json.dumps(metadata))
                reports.append(metrics)
                # Checkpoint completed configurations, including their selected C.
                temporary = output/'metrics.csv.tmp'
                with temporary.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
                    writer.writeheader()
                    writer.writerows(reports)
                # Windows scanners/viewers can hold a brief sharing lock.
                for attempt in range(10):
                    try:
                        temporary.replace(output/'metrics.csv')
                        break
                    except PermissionError:
                        if attempt == 9:
                            raise
                        time.sleep(.15 * (attempt+1))
                print(f"{category}: {tiling}/{norm}, C={best['C']}, test AP={metrics['test_ap']:.4f}", flush=True)
            del features
    return reports
