"""Only raw 3-level pyramid histograms are persisted; never SIFT arrays."""
import json
from pathlib import Path
import time
import numpy as np
from src.features import extractFeatures
from src.encoders import KMeansPredictor
from ..artifact_json import key, file_signature
from ..stagea.encoding import encode_image, level_weights
from ..stagea.svm import transform_features


def vector_from_pyramid(pyramid, k, layout):
    pyramid=np.asarray(pyramid)
    if pyramid.shape!=(21*k,) or not np.isfinite(pyramid).all() or np.any(pyramid<0):
        raise ValueError('Expected nonnegative raw 21K pyramid counts')
    if layout in ('normal','pyramid_L0'):
        return pyramid[:k]
    if layout=='flat_2x2':
        return pyramid[k:5*k]
    if layout=='pyramid_L1':
        return pyramid[:5*k]
    if layout=='pyramid_L2':
        return pyramid
    raise ValueError('Unknown layout')


def svm_vector(pyramid, choice):
    k,layout=choice['k'],choice['layout']
    raw=vector_from_pyramid(pyramid,k,layout).astype(np.float32)
    weighted=raw*level_weights(layout,k)
    return transform_features(weighted[None,:],choice['normalization'])[0]


def cached_pyramid(project, output, image_id, model, batch_size):
    project=Path(project)
    image=project/model['image_dir']/f'{image_id}.jpg'
    stat=image.stat()
    identity={'image_id':image_id,'image_path':str(image.resolve()),
              'image_bytes':stat.st_size,'image_mtime_ns':stat.st_mtime_ns,
              'model_sha256':file_signature(model['path']),
              'extractor_sha256':file_signature(project/'src/features.py'),
              'encoders_sha256':file_signature(project/'src/encoders.py'),
              'encoding_sha256':file_signature(Path(__file__).parents[1]/'stagea/encoding.py')}
    path=Path(output)/'histograms'/model['path'].stem/f'{image_id}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            pyramid=archive['pyramid'].copy()
        if metadata['identity']==identity:
            vector_from_pyramid(pyramid,model['k'],'pyramid_L2')
            return pyramid,path
    start=time.perf_counter()
    points,desc,shape=extractFeatures(image,cache_dir=None)
    vectors,_=encode_image(points,desc.astype(np.float32),shape,
                         KMeansPredictor(model['centroids']),model['k'],batch_size)
    pyramid=vectors['pyramid_L2'].astype(np.uint32)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,pyramid=pyramid,
            metadata=json.dumps({'identity':identity,'compute_seconds':time.perf_counter()-start,
                                 'layout':'pyramid_L2','grids':[1,2,4],'normalization':'none'}))
    temporary.replace(path)
    return pyramid,path
