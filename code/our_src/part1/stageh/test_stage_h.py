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


if __name__=='__main__':
    unittest.main()
