import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.visualization import RelevantVisualWordsDisplay
from ..a_json import save
from .selection import select_encoding
from .histograms import vector_from_pyramid,svm_vector,cached_pyramid
from .statistics import class_statistics,chi_squared
from .ranking import word_contributions,display_weights,ranking_rows


class StageBTests(unittest.TestCase):
    def test_no_smoke_or_cancelled_winner(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            p=Path(directory)
            save(p/'run.json',{'state':'cancelled_by_user'})
            save(p/'smoke/run.json',{'state':'complete'})
            with self.assertRaises(ValueError):select_encoding(p,'motorbike')
            manual=select_encoding(p,'motorbike',32,'pyramid_L2')
            self.assertEqual(manual['source'],'explicit_override_not_Stage_A_winner')
            save(p/'run.json',{'state':'complete'})
            with (p/'svm_ap.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['category','k','layout','normalization','validation_ap','test_ap'])
                writer.writeheader()
                writer.writerows([dict(category='motorbike',k=32,layout='normal',normalization='l2',validation_ap=.4,test_ap=1),
                                  dict(category='motorbike',k=64,layout='pyramid_L2',normalization='l2',validation_ap=.8,test_ap=0)])
            self.assertEqual(select_encoding(p,'motorbike')['k'],64)

    def test_only_pyramid_cache_and_no_reextract(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            p=Path(directory)
            (p/'src').mkdir();(p/'images').mkdir()
            (p/'src/features.py').write_text('# source fixture')
            (p/'src/encoders.py').write_text('# source encoder fixture')
            (p/'images/id.jpg').write_bytes(b'image')
            save(p/'model.json',{'centroids':[]})
            centers=np.stack([np.zeros(128),np.ones(128)*10]).astype(np.float32)
            model=dict(path=p/'model.json',centroids=centers,k=2,image_dir='images')
            extract=lambda *args,**kw:(np.array([[1,1,1,1],[6,6,1,1]]),centers.copy(),(8,8,3))
            with patch('our_src.stageb.histograms.extractFeatures',side_effect=extract) as extractor:
                pyramid,path=cached_pyramid(p,p/'out','id',model,10)
                again,_=cached_pyramid(p,p/'out','id',model,10)
                self.assertEqual(extractor.call_count,1)
                (p/'src/features.py').write_text('# changed extractor identity')
                cached_pyramid(p,p/'out','id',model,10)
                self.assertEqual(extractor.call_count,2)
            np.testing.assert_array_equal(pyramid,again)
            with np.load(path,allow_pickle=False) as archive:
                self.assertEqual(set(archive.files),{'pyramid','metadata'})
            self.assertEqual(pyramid.shape,(42,))
            self.assertEqual(len(vector_from_pyramid(pyramid,2,'normal')),2)
            self.assertEqual(len(vector_from_pyramid(pyramid,2,'flat_2x2')),8)
            choice=dict(k=2,layout='pyramid_L2',normalization='l2')
            self.assertAlmostEqual(np.linalg.norm(svm_vector(pyramid,choice)),1.,places=6)

    def test_class_stats_and_chi_squared(self):
        x={'a':np.array([1.,0.]),'b':np.array([0.,1.]),'v':np.array([1.,0.])}
        rows=[dict(image_id='a',split='train',labels=['class']),
              dict(image_id='b',split='train',labels=['class']),
              dict(image_id='v',split='val',labels=['class'])]
        stats=class_statistics(x,rows)
        feature=next(r for r in stats if r['record_type']=='feature' and r['split']=='train')
        self.assertEqual(feature['mean'],.5);self.assertEqual(feature['variance'],.25)
        val=next(r for r in stats if r['record_type']=='class_summary' and r['split']=='val')
        self.assertAlmostEqual(val['chi2_to_train_class_mean'],1/3)
        self.assertEqual(chi_squared([0,0],[0,0]),0)
        self.assertEqual(chi_squared([1,0],[0,1]),1)

    def test_actual_svm_word_contributions_display(self):
        x=np.array([1.,2.,3.,4.,5.,6.]);w=np.array([2.,-1.,3.,-2.,1.,-4.])
        c=word_contributions(x,w,2)
        self.assertAlmostEqual(c.sum(),x@w)
        hist=np.array([3,3])
        class Adapter:
            def get_visual_words(self,_):
                return hist,np.array([0,1]),np.array([0.,0.]),np.zeros((2,4))
        display=RelevantVisualWordsDisplay(np.zeros((8,8,3)),display_weights(c,hist),Adapter())
        np.testing.assert_allclose(np.sort(display.scores),np.sort(c))
        ranked=ranking_rows({'a':x,'b':x*.5},{'a':1,'b':-1},'train',dict(weights=w,bias=1.))
        self.assertEqual(ranked[0]['image_id'],'b')
        self.assertGreater(ranked[0]['decision_score'],ranked[1]['decision_score'])


if __name__=='__main__':unittest.main()
