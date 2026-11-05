"""Shared feature maps, fitting and evaluation for F and G."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize


def transform(x, operations):
    """Explicit Hellinger feature map: k(x,y) = sqrt(x) @ sqrt(y).

    Operations execute left to right, exactly once; zero vectors stay zero.
    """
    x = np.asarray(x, dtype=np.float32).copy()
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Expected finite nonnegative histogram rows')
    for op in operations:
        if op == 'sqrt':
            x = np.sqrt(x)
        elif op in ('l1', 'l2'):
            x = normalize(x, norm=op)
        elif op != 'none':
            raise ValueError(f'Unknown feature operation: {op}')
    return x


def variants(normalization):
    return [('e_norm_sqrt', [normalization, 'sqrt']),
            ('none_sqrt', ['none', 'sqrt']),
            ('l2_sqrt', ['l2', 'sqrt']),
            ('sqrt_l2', ['sqrt', 'l2'])]


def score_metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if set(y) != {-1, 1} or not np.isfinite(scores).all():
        raise ValueError('Metrics require both classes and finite scores')
    predicted = np.where(scores >= 0, 1, -1)
    p, r, f, _ = precision_recall_fscore_support(y, predicted,
        average='binary', pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[-1, 1]).ravel()
    return dict(ap=float(average_precision_score(y, scores)), precision=float(p),
        recall=float(r), f1=float(f), accuracy=float(accuracy_score(y, predicted)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), images=len(y))


def fit(x, labels, params, seed, trainer=None):
    if trainer is None:
        from src.utils import train_svm
        trainer = train_svm
    if set(np.asarray(labels)) != {-1, 1}:
        raise ValueError('SVM training requires both labels')
    start = time.perf_counter()
    w, bias = trainer(x, np.asarray(labels), C=params['C'], max_iter=params['max_iter'],
                      tol=params['tol'], seed=seed)
    return dict(weights=np.asarray(w), bias=float(bias), fit_seconds=time.perf_counter()-start)


def evaluate_parts(features, partitions, model):
    reports = {}
    for split, labels in partitions.items():
        ids = sorted(labels)
        x = np.stack([features[i] for i in ids])
        start = time.perf_counter()
        scores = x @ model['weights'] + model['bias']
        elapsed = time.perf_counter()-start
        reports[split] = {**score_metrics([labels[i] for i in ids], scores),
                          'inference_seconds': elapsed}
    return reports


def save_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.15 * (attempt+1))


def save_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=model['weights'], bias=model['bias'],
                        metadata=json.dumps(metadata))


def load_experiment(config, project):
    """Choose E by validation AP only and load unnormalized weighted histograms."""
    from ..a_json import load, file_signature
    from ..stagea.data import manifest
    from ..stagea.results import load_centroids
    from ..stagea.stage_a import binary_labels
    from ..stageb.histograms import cached_pyramid, vector_from_pyramid
    from ..stagea.encoding import level_weights
    from ..stagec.protocol import split_digest, smoke_rows
    project = Path(project)
    digest = split_digest(project/config['split_dir'])
    rows = manifest(project/config['split_dir'])
    stage_e = project/config['stage_e_dir']
    state = load(stage_e/'run.json')
    if state.get('smoke_per_class') != config.get('smoke_per_class'):
        raise ValueError('Cannot mix smoke and full Stage E results')
    if state.get('state') != 'complete' or state.get('model_count') != 45:
        raise ValueError('A completed full Stage E with 45 models is required')
    if state['split_sha256'] != digest:
        raise ValueError('Stage E used different splits')
    with (stage_e/'metrics.csv').open(newline='', encoding='utf-8') as stream:
        candidates = [r for r in csv.DictReader(stream) if r['category'] == 'motorbike']
    if len(candidates) != 9:
        raise ValueError('Expected nine motorbike configurations from E')
    # Stable first-wins tie break in the persisted E ordering.
    selected = max(candidates, key=lambda r: float(r['val_ap']))
    model_path = stage_e/selected['model']
    if not model_path.resolve().is_relative_to((stage_e/'models').resolve()):
        raise ValueError('Stage E model path is outside models/')
    with np.load(model_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        linear_model = dict(weights=archive['weights'].copy(), bias=float(archive['bias']))
    if (metadata['split_sha256'] != digest or metadata['category'] != 'motorbike' or
        metadata['layout'] != selected['layout'] or metadata['normalization'] != selected['normalization'] or
        float(metadata['C']) != float(selected['C'])):
        raise ValueError('Stage E model and metrics disagree')
    params = {k: metadata[k] for k in ('C', 'max_iter', 'tol', 'seed')}
    centroid_config = {**config, 'seed': params['seed'], 'vocabulary_sizes': [metadata['k']]}
    centroid = load_centroids(project/config['stage_a_dir']/'centroids', centroid_config, rows, project)[metadata['k']]
    if file_signature(centroid['path']) != metadata['centroid_sha256']:
        raise ValueError('Vocabulary differs from E')
    centroid.update(k=metadata['k'], image_dir=config['image_dir'])
    rows = smoke_rows(rows, config)
    partitions = {s: binary_labels(rows, s, 'motorbike') for s in ('train', 'val', 'test')}
    for s, labels in partitions.items():
        if set(labels.values()) != {-1, 1}:
            raise ValueError(f'{s} must contain motorbike and background')
    ids = sorted(set().union(*(set(p) for p in partitions.values())))
    raw = {}
    for n, image_id in enumerate(ids, 1):
        pyramid, _ = cached_pyramid(project, project/config['stage_b_dir'], image_id,
                                    centroid, config['assignment_batch_size'])
        raw[image_id] = vector_from_pyramid(pyramid, metadata['k'], metadata['layout']).astype(np.float32) * level_weights(metadata['layout'], metadata['k'])
        print(f'Histograms {n}/{len(ids)}', flush=True)
    provenance = dict(category='motorbike', negative_class='background', k=metadata['k'],
        layout=metadata['layout'], e_normalization=metadata['normalization'], split_sha256=digest,
        centroid_sha256=metadata['centroid_sha256'], centroid_file=str(centroid['path']),
        e_model=str(model_path), e_model_sha256=file_signature(model_path),
        parameters=params, selection_split='val', test_used_for_selection=False,
        smoke_per_class=config.get('smoke_per_class'))
    return raw, partitions, provenance, linear_model


def mapped(raw, operations):
    ids = sorted(raw)
    x = transform(np.stack([raw[i] for i in ids]), operations)
    return dict(zip(ids, x))
