import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from .experiment import train_models


class ExperimentTests(unittest.TestCase):
    def test_all_45_models_selection_isolation_and_reload(self):
        categories = ['motorbike', 'aeroplane', 'horse', 'car', 'person']
        rows = [dict(image_id=f'{split}_{label}', split=split,
                     labels=categories if label == 1 else ['background'])
                for split in ('train', 'val', 'test') for label in (-1, 1)]
        def labels_for(rows, split, category):
            return {r['image_id']: 1 if category in r['labels'] else -1
                    for r in rows if r['split'] == split}
        builds, calls = [], []
        def build(layout, norm):
            builds.append((layout, norm))
            dimension = {'normal': 2, 'flat_2x2': 8, 'pyramid_L2': 42}[layout]
            features = {}
            for row in rows:
                x = np.tile([4., 1.] if 'background' not in row['labels'] else [1., 4.], dimension//2)
                features[row['image_id']] = x if norm == 'none' else normalize(x[None, :], norm=norm)[0]
            return features
        # Train small real classifiers while checking what the selection API receives.
        def grid(features, train, validation, params, seed):
            self.assertTrue(all(i.startswith('train_') for i in train))
            self.assertTrue(all(i.startswith('val_') for i in validation))
            self.assertFalse(set(train) & set(validation))
            self.assertEqual(params['normalization'], ['none'])
            calls.append(1)
            best = None
            for c in params['C']:
                ids = sorted(train)
                model = LinearSVC(C=c, random_state=seed, dual=False).fit(
                    np.stack([features[i] for i in ids]), [train[i] for i in ids])
                ids = sorted(validation)
                ap = average_precision_score([validation[i] for i in ids],
                    model.decision_function(np.stack([features[i] for i in ids])))
                candidate = dict(weights=model.coef_[0], bias=float(model.intercept_[0]),
                                 C=c, validation_ap=float(ap), fit_seconds=0.)
                if best is None or ap > best['validation_ap']:
                    best = candidate
            return dict(best=best)
        config = dict(C_values=[.1, 1.], seed=42, max_iter=20000, tol=1e-4)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                reports = train_models(rows, categories, build, grid, labels_for, config,
                    output, dict(k=2, split_sha256='fixture'))
            self.assertEqual(len(calls), 45)
            self.assertEqual(len(builds), 9)
            self.assertEqual(len(set(builds)), 9)
            self.assertEqual(len(list((output/'models').glob('*.npz'))), 45)
            with (output/'metrics.csv').open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), 45)
            for category in categories:
                self.assertEqual(len([r for r in reports if r['category'] == category]), 9)
            for report in reports:
                with np.load(output/report['model'], allow_pickle=False) as archive:
                    metadata = json.loads(str(archive['metadata']))
                    self.assertEqual(metadata['normalization'], report['normalization'])
                    self.assertFalse(metadata['test_used_for_selection'])
                    self.assertEqual(metadata['split_sha256'], 'fixture')
                    features = build(metadata['layout'], metadata['normalization'])
                    labels = labels_for(rows, 'test', report['category'])
                    ids = sorted(labels)
                    scores = np.stack([features[i] for i in ids]) @ archive['weights'] + archive['bias']
                    self.assertEqual(average_precision_score([labels[i] for i in ids], scores), report['test_ap'])
                    self.assertEqual(len(archive['weights']), report['dimension'])
                self.assertEqual(report['test_ap'], 1.)
                self.assertEqual(report['test_accuracy'], 1.)


if __name__ == '__main__':
    unittest.main()
