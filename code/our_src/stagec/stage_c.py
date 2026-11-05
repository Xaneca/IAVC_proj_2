"""Select on validation, evaluate the held-out test, persist all artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from ..a_json import load, save, file_signature
from ..stagea.data import manifest
from ..stagea.results import load_centroids
from ..stagea.stage_a import binary_labels
from ..stagea.svm import train_svm_grid
from ..stageb.histograms import cached_pyramid, svm_vector
from .evaluation import evaluate, write_csv
from .protocol import split_digest, smoke_rows


def run(config, project):
    project = Path(project)
    rows = manifest(project/config['split_dir'])
    digest = split_digest(project/config['split_dir'])
    stage_b = project/config['stage_b_dir']
    smoke = config.get('smoke_per_class')
    if not smoke and load(project/config['stage_a_dir']/'run.json').get('state') != 'complete':
        raise ValueError('A completed full Stage A is required')
    previous = load(stage_b/'run.json')
    if previous.get('state') != 'complete' or (not smoke and previous.get('smoke_per_class') is not None):
        raise ValueError('A completed full Stage B is required; smoke is not a final model')
    if previous['config']['category'] != 'motorbike':
        raise ValueError('Expected the motorbike/background Stage B')
    if (project/previous['config']['split_dir']).resolve() != (project/config['split_dir']).resolve():
        raise ValueError('Stage B uses a different split directory')
    choice = previous['encoding']
    centroid_config = {**config, 'vocabulary_sizes': [choice['k']]}
    model = load_centroids(project/config['stage_a_dir']/'centroids', centroid_config, rows, project)[choice['k']]
    model = {**model, 'k': choice['k'], 'image_dir': config['image_dir']}
    if file_signature(model['path']) != previous['centroid_sha256']:
        raise ValueError('Vocabulary differs from Stage B')
    rows = smoke_rows(rows, config)
    output = project/config['output_dir']
    state = dict(state='running', config=config, encoding=choice, split_sha256=digest, smoke_per_class=smoke,
                 test_used_for_selection=False, test_evaluated=False)
    save(output/'run.json', state)
    write_csv(output/'split_manifest.csv', [dict(image_id=r['image_id'], split=r['split'],
              labels=';'.join(r['labels'])) for r in rows])
    features = {}
    categories = config['categories']
    active = [r for r in rows if set(r['labels']) & set(categories+['background'])]
    for n, row in enumerate(active, 1):
        pyramid, _ = cached_pyramid(project, stage_b, row['image_id'], model,
                                    config['assignment_batch_size'])
        features[row['image_id']] = svm_vector(pyramid, choice)
        print(f'Histograms {n}/{len(active)}', flush=True)
    reports = []
    for category in categories:
        folder = output/category
        folder.mkdir(parents=True, exist_ok=True)
        if category == 'motorbike':
            with np.load(stage_b/'svm_model.npz', allow_pickle=False) as archive:
                metadata = json.loads(str(archive['metadata']))
                if (metadata['encoding'] != choice or metadata['category'] != category or
                    metadata['centroid_sha256'] != previous['centroid_sha256']):
                    raise ValueError('Stage B model metadata differs from its run')
                best = dict(weights=archive['weights'].copy(), bias=float(archive['bias']), C=metadata['C'])
        else:
            result = train_svm_grid(features, binary_labels(rows, 'train', category),
                binary_labels(rows, 'val', category), dict(C=config['C_values'],
                normalization=['none'], max_iter=[config['max_iter']], tol=[config['tol']]), seed=config['seed'])
            best = result['best']
            write_csv(folder/'validation_trials.csv', result['trials'])
        metadata = dict(category=category, encoding=choice, C=best['C'],
                        centroid_sha256=previous['centroid_sha256'], split_sha256=digest,
                        centroid_file=str(model['path']), feature_normalization_already_applied=True,
                        training_split='train', selection_split='val')
        np.savez_compressed(folder/'svm_model.npz', weights=best['weights'], bias=best['bias'],
                            metadata=json.dumps(metadata))
        for split in ('train', 'val', 'test'):
            labels = binary_labels(rows, split, category)
            scores = {i: float(features[i] @ best['weights'] + best['bias']) for i in labels}
            reports.append(evaluate(labels, scores, folder/split, category, split))
        # Check the lists again after processing, before declaring success.
        manifest(project/config['split_dir'])
    write_csv(output/'aps_by_class.csv', reports)
    save(output/'aps_by_class.json', reports)
    state.update(state='complete', test_evaluated=True, metrics=reports)
    save(output/'run.json', state)
    return state


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_config or Path(__file__).with_name('config.json'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(load(args.config), args.project)


if __name__ == '__main__':
    main()
