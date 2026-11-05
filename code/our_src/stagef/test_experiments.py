import contextlib
import csv
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.svm import LinearSVC
from .common import transform


def modules():
    """Load the deployed package layout from either setup or installed folders."""
    root = Path(__file__).resolve().parents[1]
    name = '_fg_test_package'
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    for short, setup in [('stagef', 'stage_f_setup'), ('stageg', 'stage_g_setup'), ('stagec', 'stage_c_setup')]:
        folder = root/setup if (root/setup).is_dir() else root/short
        sub = types.ModuleType(f'{name}.{short}')
        sub.__path__ = [str(folder)]
        sys.modules[sub.__name__] = sub
    json_path = root/'stage_a_setup/a_json.py' if (root/'stage_a_setup').is_dir() else root/'a_json.py'
    spec = importlib.util.spec_from_file_location(f'{name}.a_json', json_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    return [importlib.import_module(f'{name}.{path}') for path in
            ('stagef.stage_f', 'stageg.stage_g', 'stagef.common', 'stageg.sampling', 'stagec.protocol')]


class FGTests(unittest.TestCase):
    def test_hellinger_map_and_operation_order(self):
        x = np.array([[1., 9.], [4., 0.], [0., 0.]])
        phi = transform(x, ['sqrt'])
        np.testing.assert_allclose(phi @ phi.T, np.sqrt(x[:, None, :]*x[None, :, :]).sum(axis=2))
        before = transform(x, ['l2', 'sqrt'])
        after = transform(x, ['sqrt', 'l2'])
        self.assertFalse(np.allclose(before[0], after[0]))
        np.testing.assert_allclose(after[0], np.sqrt(x[0])/np.sqrt(x[0].sum()), rtol=1e-6)
        np.testing.assert_array_equal(after[2], [0., 0.])
        with self.assertRaises(ValueError):
            transform([[-1., 1.]], ['sqrt'])

    def test_f_and_g_models_metrics_and_train_only_subsets(self):
        f, g, common, sampling, protocol = modules()
        raw, partitions = {}, {}
        for split, count in [('train', 40), ('val', 8), ('test', 8)]:
            labels = {}
            for n in range(count):
                image_id = f'{split}_{n:02}'
                label = 1 if n % 4 == 0 else -1
                labels[image_id] = label
                raw[image_id] = np.array([9., 1.] if label == 1 else [1., 9.])
            partitions[split] = labels
        params = dict(C=.1, max_iter=20000, tol=1e-4, seed=42)
        provenance = dict(category='motorbike', negative_class='background', k=2, layout='normal',
            e_normalization='l1', split_sha256='fixture', parameters=params)
        calls = []
        def trainer(x, y, C, max_iter, tol, seed):
            calls.append(dict(C=C, count=len(y), seed=seed))
            model = LinearSVC(C=C, max_iter=max_iter, tol=tol, random_state=seed,
                              dual=True, loss='hinge').fit(x, y)
            return model.coef_[0], model.intercept_[0]
        def real_fit(x, y, p, seed):
            return common.fit(x, y, p, seed, trainer=trainer)
        linear = dict(weights=np.array([1., -1.]), bias=0.)
        loaded = (raw, partitions, provenance, linear)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            with patch.object(f, 'load_experiment', return_value=loaded), patch.object(f, 'fit', side_effect=real_fit), patch.object(protocol, 'split_digest', return_value='fixture'), contextlib.redirect_stdout(io.StringIO()):
                state = f.run(dict(output_dir='stageF', split_dir='unused'), root)
            self.assertEqual(state['state'], 'complete')
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(c['count']==40 and c['C']==.1 for c in calls))
            self.assertEqual(len(list((root/'stageF/models').glob('*.npz'))), 5)
            with (root/'stageF/metrics.csv').open() as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 5)
            self.assertEqual(state['best_hellinger_variant'], 'e_norm_sqrt')
            with np.load(root/'stageF'/state['best_model'], allow_pickle=False) as model:
                metadata = json.loads(str(model['metadata']))
                self.assertEqual(metadata['operations'], ['l1', 'sqrt'])
                features = common.mapped(raw, metadata['operations'])
                report = common.evaluate_parts(features, partitions, model)
                self.assertEqual(report['test']['ap'], 1.)
            calls.clear()
            config = dict(output_dir='stageG', stage_f_dir='stageF', split_dir='unused',
                          fractions=[.25, 1.], seeds=list(range(10)))
            with patch.object(g, 'load_experiment', return_value=loaded), patch.object(g, 'fit', side_effect=real_fit), patch.object(protocol, 'split_digest', return_value='fixture'), contextlib.redirect_stdout(io.StringIO()):
                state = g.run(config, root)
            self.assertEqual(state['experiments'], 40)
            self.assertEqual(len(calls), 40)
            self.assertEqual({c['count'] for c in calls}, {10, 40})
            self.assertTrue(all(c['C']==.1 for c in calls))
            subsets = json.loads((root/'stageG/train_subsets.json').read_text())
            for key, ids in subsets.items():
                self.assertTrue(set(ids) <= set(partitions['train']))
                self.assertFalse(set(ids) & (set(partitions['val']) | set(partitions['test'])))
                fraction, seed = key.split(':')
                self.assertEqual(ids, sampling.subset(partitions['train'], float(fraction), int(seed)))
                self.assertLessEqual(abs(sum(partitions['train'][i]==1 for i in ids) - len(ids)*.25), 1)
                self.assertEqual({partitions['train'][i] for i in ids}, {-1, 1})
            with (root/'stageG/metrics_summary.csv').open() as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 4)
            self.assertTrue(all(float(r['test_ap_mean'])==1. and int(r['seeds'])==10 for r in summary))
            self.assertGreater((root/'stageG/ap_vs_train_images.png').stat().st_size, 1000)
            with self.assertRaises(ValueError):
                sampling.subset(partitions['train'], .01, 0)


if __name__ == '__main__':
    unittest.main()
