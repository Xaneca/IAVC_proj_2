import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from .evaluation import evaluate


class EvaluationTests(unittest.TestCase):
    def test_pr_ap_ranking(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            metrics = evaluate({'a': 1, 'b': -1, 'c': 1},
                               {'a': .9, 'b': .8, 'c': .7}, output, 'motorbike', 'test')
            self.assertAlmostEqual(metrics['ap'], 5/6)
            with (output/'precision_recall.csv').open() as stream:
                curve = list(csv.DictReader(stream))
            self.assertEqual(curve[-1]['threshold'], '')
            self.assertEqual(float(curve[-1]['recall']), 0)
            self.assertEqual(float(curve[0]['recall']), 1)
            with (output/'ranking.csv').open() as stream:
                ranking = list(csv.DictReader(stream))
            self.assertEqual([r['image_id'] for r in ranking], ['a', 'b', 'c'])
            self.assertGreater((output/'precision_recall.png').stat().st_size, 1000)
            self.assertEqual(json.loads((output/'ap.json').read_text())['ap'], metrics['ap'])

    def test_ties_and_invalid_evaluation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            metrics = evaluate({'z': 1, 'a': -1}, {'z': 0., 'a': 0.}, directory, 'car', 'val')
            self.assertEqual(metrics['ap'], .5)
            with self.assertRaises(ValueError):
                evaluate({'a': 1}, {'a': 0.}, directory, 'car', 'test')

    def test_historical_splits_and_mutation_rejection(self):
        root = Path(__file__).resolve().parents[1]
        data_path = root/'part1_setup/data.py'
        if not data_path.exists():
            data_path = root/'part1/data.py'
        spec = importlib.util.spec_from_file_location('split_data', data_path)
        data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data)
        reference = Path(__file__).with_name('split_reference')
        rows = data.manifest(reference)
        self.assertEqual(len(rows), 4946)
        self.assertEqual(len({r['image_id'] for r in rows}), len(rows))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            copy = Path(directory)
            for source in reference.glob('*.txt'):
                (copy/source.name).write_text(source.read_text())
            self.assertEqual(data.manifest(copy), rows)
            with (copy/'car_test.txt').open('a') as stream:
                stream.write('\nnew_image\n')
            with self.assertRaisesRegex(ValueError, 'historical'):
                data.manifest(copy)


if __name__ == '__main__':
    unittest.main()
