"""In-memory encodings reusing the supplied predictor and tiling functions."""
import time
import numpy as np
from src.encoders import KMeansPredictor, spatial_tiling, removeSpatialInformation
from ..a_metrics import count_regions, layouts


LAYOUTS = layouts([0, 1, 2])


def feature_dimension(name, k):
    return sum(g*g for g in LAYOUTS[name]['grids']) * k


def level_weights(name, k):
    return np.concatenate([np.full(g*g*k, w, dtype=np.float32)
                           for g, w in zip(LAYOUTS[name]['grids'], LAYOUTS[name]['weights'])])


def encode_image(points, descriptors, shape, predictor, k, batch_size=2048):
    """Quantize once, count finest grid once, derive all coarser layouts."""
    start = time.perf_counter()
    occupancy = np.zeros(k, dtype=np.int64)
    indices = np.empty(len(descriptors), dtype=np.int32)
    error_sum = 0.
    for offset in range(0, len(descriptors), batch_size):
        ids, distances = predictor.predict_transform(descriptors[offset:offset+batch_size])
        indices[offset:offset+len(ids)] = ids
        occupancy += np.bincount(ids, minlength=k)
        error_sum += float(np.sum(distances.astype(np.float64) ** 2))
    assignment_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if len(indices):
        fine = count_regions(points, indices, shape, k, 4, spatial_tiling).astype(np.uint32)
    else:
        fine = np.zeros((16,k), dtype=np.uint32)
    cells = fine.reshape(4,4,k)
    flat = cells.reshape(2,2,2,2,k).sum(axis=(1,3)).reshape(4,k).astype(np.uint32)
    normal = removeSpatialInformation(flat.reshape(1,-1))[0].astype(np.uint32)
    if not np.array_equal(normal, occupancy):
        raise ValueError('Global and spatial counts disagree')
    shared_count_seconds = time.perf_counter() - start
    counts = {1: normal, 2: flat.reshape(-1), 4: fine.reshape(-1)}
    features, assembly_seconds = {}, {}
    for name, spec in LAYOUTS.items():
        start = time.perf_counter()
        features[name] = np.concatenate([counts[g] for g in spec['grids']])
        assembly_seconds[name] = time.perf_counter() - start
    return features, {'occupancy': occupancy, 'squared_error_sum': error_sum,
                      'descriptors': len(indices), 'assignment_seconds': assignment_seconds,
                      'shared_count_seconds': shared_count_seconds,
                      'assembly_seconds': assembly_seconds}
