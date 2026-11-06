"""Stage E: global/2x2/pyramid × L1/L2/none for every target/background."""
import argparse
from pathlib import Path
from .experiment import train_models


def run(config, project):
    from ..artifact_json import load, save, file_signature
    from ..stagea.data import manifest
    from ..stagea.results import load_centroids
    from ..stagea.stage_a import binary_labels
    from ..stagea.svm import train_svm_grid
    from ..stageb.histograms import cached_pyramid, svm_vector
    from ..stagec.protocol import split_digest, smoke_rows
    project = Path(project)
    digest = split_digest(project/config['split_dir'])
    rows = manifest(project/config['split_dir'])
    previous = load(project/config['stage_b_dir']/'run.json')
    smoke = config.get('smoke_per_class')
    if previous.get('state') != 'complete' or (not smoke and previous.get('smoke_per_class') is not None):
        raise ValueError('A completed full Stage B is required')
    if not smoke and load(project/config['stage_a_dir']/'run.json').get('state') != 'complete':
        raise ValueError('A completed full Stage A is required')
    if (project/previous['config']['split_dir']).resolve() != (project/config['split_dir']).resolve():
        raise ValueError('Stage B used a different split directory')
    k = previous['encoding']['k']
    model = load_centroids(project/config['stage_a_dir']/'centroids',
                           {**config, 'vocabulary_sizes': [k]}, rows, project)[k]
    signature = file_signature(model['path'])
    if signature != previous['centroid_sha256']:
        raise ValueError('Vocabulary differs from Stage B')
    model = {**model, 'k': k, 'image_dir': config['image_dir']}
    rows = smoke_rows(rows, config)
    categories = config['categories']
    if set(categories) != {'motorbike', 'aeroplane', 'horse', 'car', 'person'} or len(categories) != 5:
        raise ValueError('Stage E requires the five target classes, each versus background')
    active = [r for r in rows if set(r['labels']) & set(categories+['background'])]
    output = project/config['output_dir']
    provenance = dict(k=k, centroid_sha256=signature, centroid_file=str(model['path']), split_sha256=digest,
                      smoke_per_class=smoke)
    state = dict(state='running', config=config, **provenance, expected_models=45,
                 test_used_for_selection=False, test_evaluated=False)
    save(output/'run.json', state)
    # Quantize each image once; all nine variants use this raw cached pyramid.
    pyramids = {}
    for n, row in enumerate(active, 1):
        pyramids[row['image_id']], _ = cached_pyramid(project, project/config['stage_b_dir'],
            row['image_id'], model, config['assignment_batch_size'])
        print(f'Histograms {n}/{len(active)}', flush=True)
    def build(layout, norm):
        choice = dict(k=k, layout=layout, normalization=norm)
        return {i: svm_vector(p, choice) for i, p in pyramids.items()}
    reports = train_models(rows, categories, build, train_svm_grid, binary_labels,
                           config, output, provenance)
    if split_digest(project/config['split_dir']) != digest:
        raise ValueError('Splits changed during Stage E')
    state.update(state='complete', model_count=len(reports), test_evaluated=True)
    save(output/'run.json', state)
    return state


def main():
    from ..artifact_json import load
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    run(load(args.config), args.project)


if __name__ == '__main__':
    main()
