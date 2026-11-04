"""Verify centroid-only persistence, cache reuse and split isolation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from .stage_a import manifest, train_vocabularies


class CentroidOnlyTests(unittest.TestCase):
    def test_centroid_only_and_resume(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root=Path(temp)
            (root/'src').mkdir(); (root/'images').mkdir(); (root/'splits').mkdir()
            for name in ('features.py','train.py','utils.py'):
                (root/'src'/name).write_text('# fixture')
            for name in ('a','b','v','t'):
                (root/'images'/f'{name}.jpg').write_bytes(b'image')
            for filename,content in {'x_train.txt':'a\nb\n','y_train.txt':'a\n',
                                     'x_val.txt':'v\n','x_test.txt':'t\n'}.items():
                (root/'splits'/filename).write_text(content)
            config=dict(image_dir='images',split_dir='splits',output_dir='out',seed=42,
                        vocabulary_sizes=[2,3],images_per_class=-1,descriptors_per_image=3)
            extracted=[]; fitted=[]
            def extract(path, cache_dir):
                self.assertIsNone(cache_dir)
                extracted.append(path.stem)
                return None,np.arange(20,dtype=np.uint8).reshape(10,2),None
            def fit(sample, **kw):
                self.assertEqual(sample.shape,(6,2))
                self.assertEqual(sample.dtype,np.float32)
                self.assertEqual(kw['pca_dim'],0)
                k=kw['num_words']; fitted.append(k)
                return {'words':sample[:k]}
            select=lambda *a,**kw:SimpleNamespace(filename=['a','a','b'])
            paths=train_vocabularies(config,root,extract,fit,select)
            self.assertEqual(extracted,['a','b'])
            self.assertEqual(fitted,[2,3])
            self.assertEqual(set((root/'out').rglob('*.*')),set(paths))
            for path in paths:
                data=json.loads(path.read_text())
                self.assertEqual(set(data),{'identity','centroids','descriptor_dimension'})
                self.assertNotIn('descriptors',data)
            train_vocabularies(config,root,extract,fit,select)
            self.assertEqual(extracted,['a','b'])
            self.assertEqual(fitted,[2,3])
            config['vocabulary_sizes'].append(4)
            train_vocabularies(config,root,extract,fit,select)
            self.assertEqual(fitted,[2,3,4])
            self.assertEqual(extracted,['a','b','a','b'])
            (root/'splits/y_val.txt').write_text('a\n')
            with self.assertRaises(ValueError): manifest(root/'splits')


if __name__=='__main__':
    unittest.main()
