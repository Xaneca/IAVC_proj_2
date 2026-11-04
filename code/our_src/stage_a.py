"""Stage A: fit vocabularies in RAM and save only centroids/configuration."""
import argparse
import importlib.metadata
from pathlib import Path
import numpy as np
from .a_json import file_signature, key, load, save


def manifest(split_dir):
    rows = {}
    for split in ("train", "val", "test"):
        for path in sorted(split_dir.glob(f"*_{split}.txt")):
            category = path.stem.removesuffix(f"_{split}")
            for image_id in path.read_text().splitlines():
                image_id = image_id.strip()
                if not image_id:
                    continue
                if Path(image_id).name != image_id or image_id in (".", ".."):
                    raise ValueError("Image IDs must be plain filenames without a path")
                row = rows.setdefault(image_id, {"image_id": image_id, "split": split, "labels": []})
                if row["split"] != split:
                    raise ValueError(f"Image in different splits/categories: {image_id}")
                row["labels"].append(category)
    for row in rows.values():
        row["labels"] = sorted(set(row["labels"]))
    if not any(r["split"] == "train" for r in rows.values()):
        raise ValueError("No training images found")
    return [rows[i] for i in sorted(rows)]


def train_vocabularies(config, project, extract=None, fit=None, select=None):
    """Keep one bounded training sample in RAM; persist only centroid files."""
    project = Path(project)
    split_dir = project / config['split_dir']
    image_dir = project / config['image_dir']
    rows = manifest(split_dir)
    train_ids = {r['image_id'] for r in rows if r['split'] == 'train'}
    sizes = config['vocabulary_sizes']
    if not sizes or any(not isinstance(k, int) or k < 2 for k in sizes):
        raise ValueError('K must be an integer >= 2')
    if config['descriptors_per_image'] < 1:
        raise ValueError('descriptors_per_image must be positive')
    if select is None:
        from src.utils import loadSamplesTable
        select = loadSamplesTable
    # The supplied per-class branch has a pandas reset_index conflict.
    # Reuse its loader, then select unique images here without groupby.apply.
    table = select(split_dir, image_dir, subset='train',
                   per_class=-1, random_state=config['seed'])
    per_class = config['images_per_class']
    if per_class != -1:
        if per_class < 1:
            raise ValueError('images_per_class must be -1 or positive')
        table = table.copy()
        table['_sampling_category'] = table['category'].replace({'background': '_background'})
        table = table.sort_values('_sampling_category').drop_duplicates('filename')
        selected = sorted({image_id
                           for _, group in table.groupby('_sampling_category')
                           for image_id in group.sample(n=min(per_class, len(group)),
                                                        random_state=config['seed']).filename})
    else:
        selected = sorted(set(table.filename))
    if not selected or not set(selected) <= train_ids:
        raise ValueError('Vocabulary sample must contain only internal training images')
    versions = {}
    for name in ('numpy', 'scikit-learn', 'cyvlfeat', 'pandas'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    identity = {
        'schema': 2, 'seed': config['seed'], 'pca_dim': 0,
        'images_per_class': config['images_per_class'],
        'descriptors_per_image': config['descriptors_per_image'],
        'training_ids': sorted(train_ids), 'sample_image_ids': selected,
        'images': [{'id': i, 'bytes': (image_dir/f'{i}.jpg').stat().st_size,
                    'mtime_ns': (image_dir/f'{i}.jpg').stat().st_mtime_ns} for i in selected],
        'sources': {name: file_signature(project/'src'/name)
                    for name in ('features.py', 'train.py', 'utils.py')},
        'sampling': 'sorted unique image IDs; default_rng; without replacement',
        'versions': versions}
    output = project / config['output_dir'] / 'models/kmeans'
    paths = {k: output / f'K{k}_seed{config["seed"]}_{key({**identity, "k": k})}.json'
             for k in dict.fromkeys(sizes)}
    pending = [k for k, path in paths.items() if not path.exists()]
    for k in paths:
        if k not in pending:
            saved = load(paths[k])
            if saved.get('identity') != {**identity, 'k': k}:
                raise ValueError(f'Incompatible saved centroid configuration: {paths[k]}')
            print(f'K={k}: loading existing centroids: {paths[k]}', flush=True)
    if not pending:
        return list(paths.values())
    if extract is None:
        from src.features import extractFeatures
        extract = extractFeatures
    if fit is None:
        from src.train import train_encoder
        fit = train_encoder

    rng = np.random.default_rng(config['seed'])
    chunks = []
    for index, image_id in enumerate(selected, 1):
        # Explicitly disable the supplied function's disk cache.
        _, descriptors, _ = extract(image_dir/f'{image_id}.jpg', cache_dir=None)
        if len(descriptors):
            chosen = rng.choice(len(descriptors),
                                min(len(descriptors), config['descriptors_per_image']), replace=False)
            chunks.append(descriptors[chosen].copy())
        del descriptors
        print(f'Training sample: {index}/{len(selected)} images', flush=True)
    if not chunks:
        raise ValueError('No SIFT descriptors in selected training images')
    # Float32 only at fitting time; no descriptor/sample array is written.
    sample = np.concatenate(chunks).astype(np.float32)
    del chunks
    if len(sample) < max(pending):
        raise ValueError(f'Only {len(sample)} descriptors for K={max(pending)}')
    print(f'Sample in RAM: {sample.shape}, {sample.nbytes / 1e6:.1f} MB', flush=True)
    for k in pending:
        print(f'Fitting K={k}', flush=True)
        model = fit(sample, encoder_type='bovw', num_words=k,
                    pca_dim=0, seed=config['seed'])
        centers = np.asarray(model['words'])
        if centers.shape != (k, sample.shape[1]) or not np.isfinite(centers).all():
            raise ValueError('Invalid centroids returned by supplied trainer')
        save(paths[k], {'identity': {**identity, 'k': k},
                        'centroids': centers.tolist(), 'descriptor_dimension': sample.shape[1]})
        print(f'Saved only centroids and configuration: {paths[k]}', flush=True)
    return list(paths.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('stage_a.json'))
    parser.add_argument('--stage', choices=['vocabularies', 'all'], default='vocabularies',
                        help='Both options fit missing vocabularies only; no feature/report persistence')
    args = parser.parse_args()
    train_vocabularies(load(args.config), Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
