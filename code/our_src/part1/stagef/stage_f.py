"""Fixed-parameter Hellinger comparisons for motorbike/background."""
import argparse
from pathlib import Path
import numpy as np
from .common import load_experiment, mapped, variants, fit, evaluate_parts, save_csv, save_model


def run(config, project):
    from ..artifact_json import save
    from ..stagec.protocol import split_digest
    project = Path(project)
    raw, partitions, provenance, linear = load_experiment(config, project)
    output = project/config['output_dir']
    state = dict(state='running', config=config, provenance=provenance)
    save(output/'run.json', state)
    reports = []
    experiments = [('linear_E', [provenance['e_normalization']], 'linear')]
    experiments += [(name, ops, 'hellinger') for name, ops in variants(provenance['e_normalization'])]
    for name, operations, kernel in experiments:
        features = mapped(raw, operations)
        if kernel == 'linear':
            model = linear  # Preserve the exact selected E baseline.
            model['fit_seconds'] = 0.
        else:
            ids = sorted(partitions['train'])
            model = fit(np.stack([features[i] for i in ids]),
                [partitions['train'][i] for i in ids], provenance['parameters'], provenance['parameters']['seed'])
        metrics = evaluate_parts(features, partitions, model)
        metadata = dict(**provenance, variant=name, kernel=kernel, operations=operations,
                        feature_map='explicit_sqrt' if kernel == 'hellinger' else 'identity',
                        training_split='train')
        save_model(output/'models'/f'{name}.npz', model, metadata)
        record = dict(variant=name, kernel=kernel, operations=' -> '.join(operations),
                      C=provenance['parameters']['C'], fit_seconds=model['fit_seconds'],
                      model=f'models/{name}.npz')
        for split, values in metrics.items():
            record.update({f'{split}_{k}': v for k, v in values.items()})
        reports.append(record)
        save_csv(output/'metrics.csv', reports)
    best = max((r for r in reports if r['kernel'] == 'hellinger'), key=lambda r: r['val_ap'])
    state.update(state='complete', best_hellinger_variant=best['variant'],
                 best_model=best['model'], best_validation_ap=best['val_ap'], test_evaluated=True)
    if split_digest(project/config['split_dir']) != provenance['split_sha256']:
        raise ValueError('Splits changed during F')
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
