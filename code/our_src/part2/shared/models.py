"""Train-only vocabularies/GMMs with deterministic sampling and safe reuse."""
import json
import time
import warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from src.features import extractFeatures
from .provenance import DenseSIFT
import importlib.metadata
from src.train import learn_gmm
from our_src.part1.artifact_json import load, save, key, file_signature
from our_src.part1.stagea.results import load_centroids
from our_src.part1.stagea.vocabulary import train_vocabularies


def load_gmm(path, identity):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        arrays = {name: data[name].copy() for name in ('means', 'covariances', 'priors')}
    if not metadata.get('converged'):
        raise ValueError('Cached GMM has no successful convergence record')
    if metadata['identity'] != identity:
        raise ValueError('GMM identity mismatch')
    k = identity['k']
    if (arrays['means'].shape != (k,128) or arrays['covariances'].shape != (k,128) or
        arrays['priors'].shape != (k,) or any(not np.isfinite(x).all() for x in arrays.values()) or
        np.any(arrays['covariances'] <= 0) or np.any(arrays['priors'] <= 0) or
        not np.isclose(arrays['priors'].sum(), 1)):
        raise ValueError('Invalid persisted GMM parameters')
    return arrays, metadata


def prepare(config, project, all_rows, active, output, digest, extractor=None):
    project, output = Path(project), Path(output)
    extractor = extractor or DenseSIFT(project, config)
    smoke = config.get('smoke_per_class') is not None
    vocabulary_output = project/config['centroid_cache_dir'] if config.get('centroid_cache_dir') else (output/'encoders/kmeans' if smoke else project/config['stage_a_dir'])
    folder = vocabulary_output/'centroids'
    vocabulary_config = {**config, 'output_dir': str(vocabulary_output.relative_to(project))}
    sample_ids = sorted(r['image_id'] for r in active if r['split']=='train') if smoke else None
    missing = []
    for k in config['vocabulary_sizes']:
        candidates = [p for p in folder.glob('*.json') if load(p).get('identity',{}).get('k')==k
                      and load(p).get('identity',{}).get('seed')==config['seed']]
        if not candidates:
            missing.append(k)
        else:
            load_centroids(folder, {**config, 'vocabulary_sizes':[k]}, all_rows, project)
    if missing:
        train_vocabularies({**vocabulary_config, 'vocabulary_sizes':missing}, project,
                           sample_ids=sample_ids, extract=extractor)
    centers = load_centroids(folder, config, all_rows, project)
    identities = [load(m['path'])['identity'] for m in centers.values()]
    ids = sorted(identities[0]['sample_image_ids'])
    for ident in identities:
        if (sorted(ident['sample_image_ids']) != ids or ident.get('pca_dim',0) != 0 or
            ident['descriptors_per_image'] != config['descriptors_per_image']):
            raise ValueError('K-means models must use the same raw SIFT sample/protocol')
    if smoke and ids != sample_ids:
        raise ValueError('Smoke vocabularies were fitted to a different subset')
    train_ids = {r['image_id'] for r in all_rows if r['split']=='train'}
    if not set(ids) <= train_ids:
        raise ValueError('Encoder training sample contains non-training images')
    references = {str(k): dict(path=str(m['path']), sha256=file_signature(m['path'])) for k,m in centers.items()}
    save(output/'encoders/kmeans_references.json', references)
    if 'fv' not in config.get('encoders',['bovw','vlad','fv']):
        return centers, {}
    # Same image order, per-image descriptor cap, seed and RNG as Stage A.
    rng = np.random.default_rng(config['seed'])
    base_identity = dict(seed=config['seed'], split_sha256=digest, sample_image_ids=ids,
        descriptors_per_image=config['descriptors_per_image'], pca_dim=0,
        sample_images=identities[0]['images'], sift_sha256=file_signature(project/'src/features.py'),
        gmm_parameters=config['gmm'], implementation_sha256=file_signature(Path(__file__)),
        provider_sha256=file_signature(project/'src/encoders.py'),
        versions={p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','scikit-image')})
    gmms, pending = {}, []
    for k in config['vocabulary_sizes']:
        identity = {**base_identity, 'k':k}
        path = output/'encoders'/f'gmm_K{k}_seed{config["seed"]}_{key(identity)}.npz'
        if path.exists():
            arrays, metadata = load_gmm(path, identity)
            gmms[k] = dict(params=arrays, metadata=metadata, path=path)
        else:
            pending.append((k, identity, path))
    if pending:
        chunks = []
        for n, image_id in enumerate(ids, 1):
            _, descriptors, _ = extractor(project/config['image_dir']/f'{image_id}.jpg', cache_dir=None)
            if len(descriptors):
                chosen = rng.choice(len(descriptors), min(len(descriptors), config['descriptors_per_image']), replace=False)
                chunks.append(descriptors[chosen].astype(np.float32))
            print(f'GMM sample {n}/{len(ids)}', flush=True)
        if not chunks:
            raise ValueError('No training SIFT descriptors for GMM fitting')
        sample = np.concatenate(chunks).astype(np.float64)
        if len(sample) < max(k for k, _, _ in pending):
            raise ValueError('Too few training descriptors for the requested GMM sizes')
        for k, identity, path in pending:
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                gmm = learn_gmm(sample, n_modes=k, gm_args=dict(covariance_type='diag',
                    random_state=config['seed'], **config['gmm']))
            if not gmm.converged_:
                raise ValueError(f'GMM K={k} did not converge; adjust gmm.max_iter/tol on training only')
            metadata = dict(identity=identity, fit_seconds=time.perf_counter()-start,
                sample_descriptors=len(sample), converged=bool(gmm.converged_), n_iter=int(gmm.n_iter_),
                warnings=[str(w.message) for w in caught])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, means=gmm.means_, covariances=gmm.covariances_,
                                priors=gmm.weights_, metadata=json.dumps(metadata))
            arrays, metadata = load_gmm(path, identity)
            gmms[k] = dict(params=arrays, metadata=metadata, path=path)
            print(f'GMM K={k} trained in {metadata["fit_seconds"]:.2f}s', flush=True)
    references = {str(k): dict(path=str(m['path']), sha256=file_signature(m['path'])) for k,m in centers.items()}
    save(output/'encoders/kmeans_references.json', references)
    return centers, gmms
