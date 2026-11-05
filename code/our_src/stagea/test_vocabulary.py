import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from . import vocabulary


class VocabularyTests(unittest.TestCase):
    def test_missing_sizes_share_training_sample_and_reuse_existing(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root/'src').mkdir()
            (root/'images').mkdir()
            for name in ('features.py', 'train.py', 'utils.py'):
                (root/'src'/name).write_text('# fixture')
            for image_id in ('a', 'b', 'v'):
                (root/'images'/f'{image_id}.jpg').write_bytes(b'fixture')
            rows = [dict(image_id=i, split=s, labels=['class'])
                    for i, s in [('a', 'train'), ('b', 'train'), ('v', 'val')]]
            config = dict(split_dir='splits', image_dir='images', output_dir='stageA',
                vocabulary_sizes=[2, 3], images_per_class=-1, descriptors_per_image=4, seed=42)
            extract = Mock(return_value=(None, np.ones((8, 128)), None))
            fit = Mock(side_effect=lambda sample, **kw: {'words': np.ones((kw['num_words'],128))})
            select = Mock(return_value=pd.DataFrame({'filename': ['a', 'b']}))
            with patch.object(vocabulary, 'manifest', return_value=rows):
                paths = vocabulary.train_vocabularies(config, root, extract, fit, select)
                self.assertEqual(extract.call_count, 2)
                self.assertEqual(fit.call_count, 2)
                self.assertEqual(fit.call_args_list[0].args[0].shape, (8,128))
                self.assertIs(fit.call_args_list[0].args[0], fit.call_args_list[1].args[0])
                vocabulary.train_vocabularies(config, root, extract, fit, select)
                self.assertEqual(extract.call_count, 2)
                self.assertEqual(fit.call_count, 2)
                config['vocabulary_sizes'] = [2, 3, 4]
                vocabulary.train_vocabularies(config, root, extract, fit, select)
                self.assertEqual(fit.call_count, 3)
                self.assertEqual(fit.call_args.kwargs['num_words'], 4)
                with self.assertRaisesRegex(ValueError, 'only internal training'):
                    vocabulary.train_vocabularies(config, root, extract, fit, select, sample_ids=['v'])
            for path in paths:
                saved = json.loads(path.read_text())
                self.assertEqual(saved['identity']['sample_image_ids'], ['a', 'b'])
                self.assertGreaterEqual(saved['fit_seconds'], 0.)
            self.assertTrue(all(call.kwargs['cache_dir'] is None for call in extract.call_args_list))


if __name__ == '__main__':
    unittest.main()
