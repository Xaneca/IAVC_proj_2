import unittest
import numpy as np
from sklearn.mixture import GaussianMixture
from .encoders import create,encode,dimension,l2


class EncoderTests(unittest.TestCase):
    def test_bovw_and_intra_vlad_against_expected_blocks(self):
        centers=np.stack([np.zeros(128),np.full(128,10.)]).astype(np.float32)
        desc=np.stack([np.ones(128),np.full(128,2.),np.full(128,11.)]).astype(np.float32)
        bovw=encode(create('bovw',{'words':centers},2),desc,'bovw',2)
        np.testing.assert_allclose(bovw,l2([2.,1.]),rtol=1e-6)
        vlad=encode(create('vlad',{'words':centers},2),desc,'vlad',2)
        self.assertEqual(vlad.shape,(256,))
        np.testing.assert_allclose(np.linalg.norm(vlad.reshape(2,128),axis=1),[1/np.sqrt(2)]*2,rtol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(vlad),1.,rtol=1e-6)

    def test_fisher_signed_power_and_real_library_dimension(self):
        rng=np.random.default_rng(42)
        desc=np.concatenate([rng.normal(0.,.1,(10,128)),rng.normal(3.,.1,(10,128))])
        model=GaussianMixture(n_components=2,covariance_type='diag',random_state=42).fit(desc)
        encoder=create('fv',dict(means=model.means_,covariances=model.covariances_,priors=model.weights_),2)
        raw=encoder.encode(desc[:5])
        self.assertTrue(np.any(raw<0))
        expected=l2(np.sign(raw)*np.sqrt(np.abs(raw)))
        actual=encode(encoder,desc[:5],'fv',2)
        self.assertEqual(actual.shape,(514,))
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-7)
        self.assertTrue(np.isfinite(actual).all())

    def test_empty_images_have_fixed_zero_vectors(self):
        for kind in ('bovw','vlad','fv'):
            for k in (64,128):
                vector=encode(None,np.empty((0,128)),kind,k)
                self.assertEqual(vector.shape,(dimension(kind,k),))
                self.assertEqual(np.count_nonzero(vector),0)
        self.assertEqual(dimension('fv',64),16448)
        self.assertEqual(dimension('fv',128),32896)


    def test_supplied_api_and_no_model_mutation(self):
        import inspect
        from unittest.mock import patch
        import src.encoders as supplied
        from skimage.feature import fisher_vector
        self.assertEqual(set(inspect.signature(fisher_vector).parameters),{'descriptors','gmm','improved','alpha'})
        self.assertIn('2KD + K',fisher_vector.__doc__)
        rng=np.random.default_rng(7)
        train=rng.normal(size=(40,128))
        model=GaussianMixture(n_components=2,covariance_type='diag',random_state=42).fit(train)
        encoder=create('fv',dict(means=model.means_.copy(),covariances=model.covariances_.copy(),priors=model.weights_.copy()),2)
        self.assertIsInstance(encoder,supplied.FisherVectorEncoder)
        before=[v.copy() for v in (encoder.gmm.means_,encoder.gmm.covariances_,encoder.gmm.weights_)]
        with patch.object(supplied,'fisher_vector',wraps=fisher_vector) as call:
            first=encode(encoder,train[:3],'fv',2)
            repeated=encode(encoder,train[:3],'fv',2)
            encode(encoder,train[20:],'fv',2)  # validation/test-like examples cannot refit
            self.assertEqual(call.call_count,3)
        np.testing.assert_array_equal(first,repeated)
        self.assertAlmostEqual(float(np.linalg.norm(first)),1.,places=6)
        for expected,actual in zip(before,(encoder.gmm.means_,encoder.gmm.covariances_,encoder.gmm.weights_)):
            np.testing.assert_array_equal(expected,actual)

    def test_train_only_gmm_determinism_and_compatible_reuse(self):
        from pathlib import Path
        import json,tempfile,shutil
        from unittest.mock import patch
        from . import models
        from skimage.feature import learn_gmm
        rng=np.random.default_rng(3)
        descriptors={i:rng.normal(loc=n*2.,scale=.2,size=(30,128)).astype(np.float32) for n,i in enumerate(('a','b'))}
        rows=[dict(image_id=i,split=s,labels=['target']) for i,s in [('a','train'),('b','train'),('v','val'),('t','test')]]
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary); (project/'src').mkdir()
            source=Path(__file__).resolve().parents[3]/'src'
            for name in ('features.py','encoders.py'): shutil.copyfile(source/name,project/'src'/name)
            config=dict(seed=42,smoke_per_class=2,vocabulary_sizes=[2],descriptors_per_image=20,
                gmm=dict(max_iter=100,tol=.001,reg_covar=1e-6,n_init=1),image_dir='images')
            calls=[]
            def extract(path,cache_dir=None):
                self.assertIn(path.stem,('a','b')); calls.append(path.stem)
                return None,descriptors[path.stem].copy(),None
            def setup(folder):
                p=folder/'encoders/kmeans/centroids/K2.json'; p.parent.mkdir(parents=True)
                p.write_text(json.dumps(dict(identity=dict(k=2,seed=42,sample_image_ids=['a','b'],pca_dim=0,descriptors_per_image=20,images=[]))))
                return {2:dict(path=p,centroids=np.zeros((2,128)),fit_seconds=0.)}
            def fit(output):
                centers=setup(output)
                with patch.object(models,'load_centroids',return_value=centers),patch.object(models,'learn_gmm',wraps=learn_gmm) as fitted:
                    result=models.prepare(config,project,rows,rows,output,'split',extractor=extract)[1][2]
                    self.assertEqual(fitted.call_args.kwargs['gm_args']['random_state'],42)
                    self.assertEqual(fitted.call_args.kwargs['gm_args']['covariance_type'],'diag')
                return result,centers
            first,centers=fit(project/'first'); second,_=fit(project/'second')
            for key in ('means','covariances','priors'):
                np.testing.assert_array_equal(first['params'][key],second['params'][key])
            self.assertEqual(calls,['a','b','a','b'])
            with patch.object(models,'load_centroids',return_value=centers),patch.object(models,'learn_gmm',side_effect=AssertionError('Must reuse cached GMM')):
                cached=models.prepare(config,project,rows,rows,project/'first','split',extractor=extract)
            self.assertEqual(calls,['a','b','a','b'])
            with self.assertRaises(ValueError): models.load_gmm(first['path'],{'wrong':'identity'})

    def test_comparison_reporting(self):
        from pathlib import Path
        import tempfile
        from .reporting import consolidate
        records=[]
        for category,ap in [('cat',.2),('dog',.8)]:
            records.append(dict(experiment='E3',encoder='fv',k=64,dimension=16448,seed=42,
                val_ap=ap,test_ap=ap,C=.1,category=category,sift_seconds_shared=12.,encoder_fit_seconds=2.,
                encoding_seconds_shared=6.,svm_fit_seconds=1.,svm_search_fit_seconds=3.,
                train_inference_seconds=.1,val_inference_seconds=.1,test_inference_seconds=.1))
        with tempfile.TemporaryDirectory() as temporary:
            result=consolidate(temporary,records,3,dict(smoke_per_class=2))[0]
            self.assertAlmostEqual(result['validation_map'],.5)
            self.assertEqual(result['descriptor_bytes_per_image'],65792)
            self.assertEqual(result['encoding_seconds_per_image'],2.)
            self.assertEqual(result['sift_extraction_seconds_shared'],12.)
            self.assertTrue((Path(temporary)/'STAGE_H_REPORT.md').exists())


    def test_provided_dense_sift_cache_reuse_and_corruption(self):
        from pathlib import Path
        import tempfile,shutil
        from unittest.mock import patch
        from PIL import Image
        import src.features as provided
        from .provenance import DenseSIFT
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'src').mkdir()
            shutil.copyfile(Path(__file__).resolve().parents[3]/'src/features.py',root/'src/features.py')
            image=root/'sample.png'
            Image.fromarray(np.random.default_rng(1).integers(0,256,(96,96,3),dtype=np.uint8)).save(image)
            cached=DenseSIFT(root,dict(sift_cache_dir='cache'))
            with patch.object(provided,'computeFeatures',wraps=provided.computeFeatures) as compute:
                a=cached(image); b=cached(image)
                self.assertEqual(compute.call_count,1)
                np.testing.assert_array_equal(a[1],b[1])
                Image.fromarray(np.zeros((96,96,3),dtype=np.uint8)).save(image)
                cached(image); self.assertEqual(compute.call_count,2)
            files=list((root/'cache').rglob('sample.npz'))
            self.assertEqual(len(files),2)
            for p in files: p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): cached(image)


    def test_stage_roles_and_baseline_rejection(self):
        from pathlib import Path
        import tempfile,json
        from .reporting import compatible_baseline
        from our_src.part1.artifact_json import save
        part2=Path(__file__).resolve().parents[1]
        h=json.loads((part2/'stageh/config.json').read_text()); i=json.loads((part2/'stagei/config.json').read_text())
        self.assertEqual(h['encoders'],['bovw','vlad']); self.assertEqual(i['encoders'],['fv'])
        self.assertEqual(h['vocabulary_sizes'],[64,128]); self.assertEqual(h['C_values'],[.1,1.,10.])
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            save(folder/'run.json',dict(state='complete'))
            save(folder/'complete.json',dict(artifacts={}))
            baseline=dict(split_sha256='s',image_sha256={'a':'x'},rows=[],versions={},sources={},config=h)
            save(folder/'manifest.json',baseline)
            current=dict(baseline,image_sha256={'a':'different'},config=i)
            with self.assertRaisesRegex(ValueError,'image_sha256'): compatible_baseline(folder,current)


if __name__=='__main__':
    unittest.main()
