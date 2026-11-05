"""Meaningful checks with the actual provided encoders and SVM trainer."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from src.encoders import KMeansPredictor
from .data import manifest
from .encoding import encode_image, feature_dimension, level_weights
from .svm import train_svm_grid, train_svm
from ..a_metrics import occupancy_metrics, pair_indices


class Part1Tests(unittest.TestCase):
    def test_count_conservation_and_empty_tiles(self):
        centers=np.array([[0,0],[10,10]],dtype=np.float32)
        points=np.array([[0,0],[0,8],[4,0],[4,8]])
        desc=np.array([[0,0],[10,10],[0,0],[10,10]],dtype=np.float32)
        vectors,stats=encode_image(points,desc,(8,16),KMeansPredictor(centers),2,batch_size=2)
        self.assertEqual(stats['squared_error_sum'],0)
        np.testing.assert_array_equal(vectors['normal'],[2,2])
        np.testing.assert_array_equal(vectors['flat_2x2'],[1,0,0,1,1,0,0,1])
        self.assertEqual(vectors['pyramid_L2'].shape,(42,))
        self.assertEqual(vectors['pyramid_L1'].sum(),8)
        self.assertEqual(vectors['pyramid_L2'].sum(),12)
        self.assertEqual(vectors['flat_2x2'].sum(),4)
        for name in vectors:
            self.assertEqual(len(vectors[name]),feature_dimension(name,2))
        np.testing.assert_array_equal(level_weights('pyramid_L2',2),[.25]*10+[.5]*32)
        empty,_=encode_image(np.empty((0,2)),np.empty((0,2)),(8,16),KMeansPredictor(centers),2)
        self.assertTrue(all(np.count_nonzero(v)==0 for v in empty.values()))

    def test_svm_any_encoding_and_optional_test(self):
        features={i:np.array(v,dtype=np.float32) for i,v in {
            't+':[3,1,0],'t-':[-3,1,0], 'v+':[2,1,0],'v-':[-2,1,0],
            's+':[4,1,0],'s-':[-4,1,0]}.items()}
        train={'t+':1,'t-':-1}; val={'v+':1,'v-':-1}
        grid={'C':[.1,1.],'normalization':['none','l2'],'max_iter':[10000]}
        with patch('our_src.stagea.svm.train_svm',wraps=train_svm) as trainer:
            result=train_svm_grid(features,train,val,grid,test={'s+':1,'s-':-1})
            self.assertEqual(trainer.call_count,4)
        self.assertEqual(result['best']['validation_ap'],1.)
        self.assertEqual(result['test_ap'],1.)
        result2=train_svm_grid(features,train,val,grid,test={'s+':-1,'s-':1})
        self.assertEqual(result2['best']['C'],result['best']['C'])
        self.assertEqual(result2['best']['normalization'],result['best']['normalization'])
        self.assertLess(result2['test_ap'],result['test_ap'])
        self.assertIsNone(train_svm_grid(features,train,val,grid)['test_ap'])
        with self.assertRaises(ValueError):
            train_svm_grid(features,train,{'t+':1,'v-':-1},grid)
        with self.assertRaises(ValueError):
            train_svm_grid(features,train,{'v+':1},grid)

    def test_multilabel_manifest_no_leakage(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            p=Path(directory)
            (p/'a_train.txt').write_text('same\n')
            (p/'b_train.txt').write_text('same\n')
            rows=manifest(p)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['labels'],['a','b'])
            (p/'b_val.txt').write_text('same\n')
            with self.assertRaises(ValueError):manifest(p)

    def test_entropy_and_pair_sampling(self):
        self.assertAlmostEqual(occupancy_metrics([10,10])['occupancy_entropy_normalized'],1.)
        self.assertEqual(occupancy_metrics([20,0])['vocabulary_utilization'],.5)
        pairs=pair_indices(8,100,42)
        self.assertEqual(len(pairs),28)
        self.assertEqual(len(set(map(tuple,pairs))),28)
        self.assertTrue(np.all(pairs[:,0]<pairs[:,1]))


if __name__=='__main__':
    unittest.main()
