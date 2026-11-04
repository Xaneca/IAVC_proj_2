"""BoVW metrics and layouts; no fitting or filesystem side effects."""
import numpy as np


def layouts(levels):
    result = {"normal": {"grids": [1], "weights": [1.]},
              "flat_2x2": {"grids": [2], "weights": [1.]}}
    for maximum in levels:
        if maximum < 0 or maximum > 2:
            raise ValueError("Supported pyramid maximum levels: 0, 1, 2")
        weights = ([1.] if maximum == 0 else
                   [2. ** (-maximum)] + [2. ** (level - maximum - 1)
                                                     for level in range(1, maximum + 1)])
        result[f"pyramid_L{maximum}"] = {
            "grids": [2 ** level for level in range(maximum + 1)], "weights": weights}
    return result


def count_regions(points, assignments, image_size, k, grid, tiling):
    bounds = np.array([[y / grid, x / grid, (y + 1) / grid, (x + 1) / grid]
                       for y in range(grid) for x in range(grid)])
    counts = np.stack([np.bincount(assignments[mask], minlength=k)
                      for mask, _ in tiling(points, bounds, image_size[:2])])
    if int(counts.sum()) != len(assignments):
        raise ValueError("Spatial cells must cover every descriptor exactly once")
    return counts


def vector_from_counts(counts, layout):
    return np.concatenate([np.asarray(counts[str(g)]).reshape(-1)
                           for g in layout["grids"]])


def weighted_l2(vector, layout, k):
    vector = np.asarray(vector, dtype=np.float64).copy()
    offset = 0
    for grid, weight in zip(layout["grids"], layout["weights"]):
        end = offset + grid * grid * k
        vector[offset:end] *= weight
        offset = end
    length = np.linalg.norm(vector)
    return vector / length if length else vector


def occupancy_metrics(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    p = counts[counts > 0] / total if total else np.array([])
    entropy = float(-np.sum(p * np.log(p))) if len(p) else 0.
    return {"vocabulary_utilization": float(np.mean(counts > 0)),
            "occupancy_entropy_nats": entropy,
            "occupancy_entropy_normalized": entropy / np.log(len(counts)) if len(counts) > 1 else 0.,
            "effective_visual_words": float(np.exp(entropy)) if total else 0.,
            "cluster_occupancy": counts.astype(int).tolist()}


def pair_indices(n, maximum, seed):
    total = n * (n - 1) // 2
    if not total:
        return np.empty((0, 2), dtype=int)
    rng = np.random.default_rng(seed)
    selected = rng.choice(total, min(maximum, total), replace=False)
    ends = np.cumsum(np.arange(n - 1, 0, -1, dtype=np.int64))
    first = np.searchsorted(ends, selected, side="right")
    starts = np.where(first == 0, 0, ends[np.maximum(first - 1, 0)])
    second = first + 1 + selected - starts
    return np.column_stack((first, second))


def separability(vectors, label_sets, pairs):
    """Mean cosine distance: shared-label versus disjoint-label image pairs.

    Call with weighted, L2-normalized vectors. Zero-vector pairs are excluded.
    Labels describe all categories, not only the category used by one SVM.
    """
    values = {"intra": [], "inter": []}
    excluded = 0
    for i, j in pairs:
        a, b = vectors[i], vectors[j]
        if not np.any(a) or not np.any(b):
            excluded += 1
            continue
        distance = float(np.clip(1 - np.dot(a, b), 0, 2))
        group = "intra" if label_sets[i] & label_sets[j] else "inter"
        values[group].append(distance)
    result = {f"{group}_mean_cosine_distance": float(np.mean(x)) if x else None
              for group, x in values.items()}
    result.update({f"{group}_pairs": len(x) for group, x in values.items()})
    intra, inter = result["intra_mean_cosine_distance"], result["inter_mean_cosine_distance"]
    result["inter_minus_intra"] = inter - intra if intra is not None and inter is not None else None
    result["zero_vector_pairs_excluded"] = excluded
    return result
