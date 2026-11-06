"""Run identities, optional reuse of the provided Dense SIFT cache, resource checks."""
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from src.features import extractFeatures
from our_src.part1.artifact_json import file_signature, key, save, load


def identity(config, project, rows, split_sha):
    project=Path(project)
    sources={str(p.relative_to(project)):file_signature(p) for p in Path(__file__).parent.glob('*.py')}
    for relative in ('src/features.py','src/encoders.py','src/train.py','src/utils.py',
        'our_src/part1/stagea/svm.py','our_src/part1/stagea/vocabulary.py',
        'our_src/part1/stagec/evaluation.py','our_src/part1/stagec/protocol.py','our_src/part1/stagef/common.py'):
        sources[relative]=file_signature(project/relative)
    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','scikit-image','cyvlfeat','pandas')}
    return dict(config=config,split_sha256=split_sha,rows=rows,sources=sources,versions=versions,
        image_sha256={r['image_id']:file_signature(project/config['image_dir']/(r['image_id']+'.jpg')) for r in rows},
        smoke_per_class=config.get('smoke_per_class'),hardware=dict(platform=platform.platform(),cpu=platform.processor(),
            threads={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))


class DenseSIFT:
    """Optional content-addressed wrapper around the provided cache API; no new extractor."""
    def __init__(self, project, config):
        self.project=Path(project); self.folder=config.get('sift_cache_dir')
        self.extraction_seconds=0.; self.cache_read_seconds=0.; self.hits=0; self.misses=0
        self.extractor_sha=file_signature(self.project/'src/features.py')
        self.cache_key=key(dict(source=self.extractor_sha,versions={p:importlib.metadata.version(p) for p in ('numpy','cyvlfeat','scikit-image')}))

    def __call__(self, image, cache_dir=None):
        image=Path(image)
        if not self.folder:
            start=time.perf_counter(); result=extractFeatures(image,cache_dir=None)
            self.extraction_seconds+=time.perf_counter()-start; self.misses+=1
            return result
        directory=self.project/self.folder/self.cache_key/file_signature(image)
        path=directory/(image.stem+'.npz'); checksum=directory/(image.stem+'.sha256.json')
        if path.exists() and (not checksum.exists() or load(checksum)['sha256']!=file_signature(path)):
            raise ValueError(f'Incompatible/corrupt SIFT cache: {path}')
        hit=path.exists()
        start=time.perf_counter(); result=extractFeatures(image,cache_dir=directory)
        seconds=time.perf_counter()-start
        if hit: self.cache_read_seconds+=seconds; self.hits+=1
        else: self.extraction_seconds+=seconds; self.misses+=1
        if result[1].ndim!=2 or result[1].shape[1]!=128 or not np.isfinite(result[1]).all():
            raise ValueError('Invalid Dense SIFT descriptors')
        if not checksum.exists(): save(checksum,dict(sha256=file_signature(path),image_sha256=file_signature(image)))
        return result


def resources(rows,config,project):
    train=sum(r['split']=='train' for r in rows)
    n=train*config['descriptors_per_image']; k=max(config['vocabulary_sizes']); d=128
    available=None
    if os.name=='nt':
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','page','avail_page','virtual','avail_virtual','extended')]
        m=Memory(); m.length=ctypes.sizeof(m); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); available=m.available
    else:
        try: available=os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
        except (ValueError,OSError,AttributeError): pass
    # Dense sklearn diagonal-GMM EM uses sample plus multiple N*K float64 workspaces.
    use_gmm='fv' in config.get('encoders',['bovw','vlad','fv'])
    required=(n*d*8 + 3*n*k*8 + n*d*4 + 1024**3) if use_gmm else (4*n*d*4 + 1024**3)
    from .encoders import dimension
    feature_bytes=len(rows)*sum(dimension(e,k) for k in config['vocabulary_sizes'] for e in config.get('encoders',['bovw','vlad','fv']))*4
    free=shutil.disk_usage(Path(project).anchor).free
    return dict(training_images=train,descriptor_upper_bound=n,available_ram_bytes=available,
        estimated_ram_bytes=required,temporary_feature_bytes=feature_bytes,free_disk_bytes=free,
        feasible=available is not None and available>=required and free>=feature_bytes+1024**3,
        rationale='Conservative GMM EM/K-means estimate with descriptor chunks and 1 GiB headroom; no full SIFT disk cache assumed.')
