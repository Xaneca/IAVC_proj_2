"""Metrics on fixed-size encodings; no classifier or test-set selection."""
import numpy as np
from sklearn.preprocessing import normalize
from .metrics_utils import occupancy_metrics, pair_indices, separability
from .encoding import LAYOUTS, level_weights


def evaluate_representation(matrix, rows, k, layout, stats, model, extraction_seconds, seed, max_pairs):
    reports = []
    for split in ('train','val'):
        idx = [i for i,r in enumerate(rows) if r['split']==split]
        if not idx:
            continue
        x = matrix[idx]
        weighted = x.astype(np.float32) * level_weights(layout,k)
        weighted = normalize(weighted, norm='l2')
        labels = [set(rows[i]['labels']) for i in idx]
        pairs = pair_indices(len(idx),max_pairs,seed)
        sep = separability(weighted,labels,pairs)
        occupancy = sum((stats[i]['occupancy'] for i in idx),np.zeros(k,dtype=np.int64))
        total = sum(stats[i]['descriptors'] for i in idx)
        error = sum(stats[i]['squared_error_sum'] for i in idx)
        occ = occupancy_metrics(occupancy)
        reports.append({
            'k':k, 'seed':seed, 'layout':layout, 'split':split, 'images':len(idx),
            'representation_dimension':x.shape[1], 'spatial_cells':sum(g*g for g in LAYOUTS[layout]['grids']),
            'visual_words':k, 'sift_dimension':128,
            'quantization_mean_squared_l2':error/total if total else None,
            'quantization_squared_l2_sum':error, 'descriptors':total,
            'vocabulary_utilization':occ['vocabulary_utilization'],
            'mean_image_vocabulary_utilization':float(np.mean([np.count_nonzero(stats[i]['occupancy'])/k for i in idx])),
            'histogram_sparsity':float(np.count_nonzero(x==0)/x.size),
            'cluster_occupancy_entropy_nats':occ['occupancy_entropy_nats'],
            'cluster_occupancy_entropy_normalized':occ['occupancy_entropy_normalized'],
            'effective_visual_words':occ['effective_visual_words'],
            'sift_extraction_seconds_shared':sum(extraction_seconds[i] for i in idx),
            'kmeans_fit_seconds':model['fit_seconds'],
            'kmeans_fit_timing_status':'historical_not_recorded' if model['fit_seconds'] is None else 'recorded',
            'kmeans_fit_timing_source':model.get('fit_timing_source','not_recorded'),
            'assignment_seconds_shared':sum(stats[i]['assignment_seconds'] for i in idx),
            'spatial_count_seconds_shared':sum(stats[i]['shared_count_seconds'] for i in idx),
            'layout_assembly_seconds':sum(stats[i]['assembly_seconds'][layout] for i in idx),
            'centroid_json_disk_bytes':model['path'].stat().st_size,
            'centroid_float32_array_bytes':model['centroids'].nbytes,
            'feature_dense_uint32_bytes':x.nbytes, 'feature_disk_bytes':0,
            'storage_status':'features_in_RAM_only',
            'vocabulary_stability_status':'not_evaluated_single_seed',
            'vocabulary_stability_score':None, **sep})
    return reports
