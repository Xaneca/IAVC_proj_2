"""Small normalization adapters around the supplied encoders and libraries."""
import numpy as np
from src.encoders import BoVWEncoder, VLADEncoder, FisherVectorEncoder, vlad


def l2(vector):
    x = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def dimension(encoder, k, d=128):
    return {'bovw': k, 'vlad': k*d, 'fv': k*(2*d+1)}[encoder]


class IntraVLAD(VLADEncoder):
    def encode(self, descriptors):
        descriptors = self.proj.transform(descriptors).astype(np.float32)
        words = self.clusterer.predict(descriptors)
        assignments = np.zeros((len(words), self.cfg['num_words']), dtype=np.float32)
        assignments[np.arange(len(words)), words] = 1.
        return vlad(descriptors, self.cfg['words'].astype(np.float32), assignments,
            normalize_components=True, normalize_mass=False, square_root=False,
            unnormalized=False).ravel()


def encode(encoder, descriptors, kind, k):
    if len(descriptors) == 0:
        return np.zeros(dimension(kind, k), dtype=np.float32)
    x = np.asarray(encoder.encode(descriptors), dtype=np.float32)
    if kind == 'fv':
        x = np.sign(x)*np.sqrt(np.abs(x))
    x = l2(x)
    if x.shape != (dimension(kind, k),) or not np.isfinite(x).all():
        raise ValueError(f'Invalid {kind} encoding shape or values')
    return x


def create(kind, params, k):
    config = dict(type=kind, num_words=k, pca_dim=0, **params)
    return {'bovw': BoVWEncoder, 'vlad': IntraVLAD, 'fv': FisherVectorEncoder}[kind](config)
