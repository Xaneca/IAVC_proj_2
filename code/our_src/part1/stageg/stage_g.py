"""AP learning curves for fixed linear and best Hellinger classifiers."""
import argparse
import json
from pathlib import Path
import numpy as np
from .sampling import subset
from ..stagef.common import load_experiment, mapped, fit, evaluate_parts, save_csv


def summarize(records):
    summaries = []
    for kernel in ('linear', 'hellinger'):
        for fraction in sorted({r['fraction'] for r in records}):
            group = [r for r in records if r['kernel'] == kernel and r['fraction'] == fraction]
            aps = np.array([r['test_ap'] for r in group])
            summaries.append(dict(kernel=kernel, fraction=fraction, train_images=group[0]['train_images'],
                seeds=len(group), test_ap_mean=float(aps.mean()), test_ap_std=float(aps.std(ddof=1)) if len(aps)>1 else 0.,
                val_ap_mean=float(np.mean([r['val_ap'] for r in group]))))
    return summaries


def plot(summaries, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for kernel in ('linear', 'hellinger'):
        group = [r for r in summaries if r['kernel'] == kernel]
        ax.errorbar([r['train_images'] for r in group], [r['test_ap_mean'] for r in group],
            yerr=[r['test_ap_std'] for r in group], marker='o', capsize=3, label=kernel)
    ax.set(xlabel='Number of training images (motorbike + background)', ylabel='Test AP (mean ± sample SD)',
           ylim=(0, 1.02), title='Fixed vocabulary — stratified training subsets')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config, project):
    from ..artifact_json import load, save, file_signature
    from ..stagec.protocol import split_digest
    project = Path(project)
    fractions, seeds = config['fractions'], config['seeds']
    if (not fractions or len(set(fractions)) != len(fractions) or
        any(not 0 < f <= 1 for f in fractions) or len(seeds) != 10 or len(set(seeds)) != 10):
        raise ValueError('Expected unique fractions in (0,1] and ten distinct seeds')
    raw, partitions, provenance, _ = load_experiment(config, project)
    stage_f = project/config['stage_f_dir']
    previous = load(stage_f/'run.json')
    if previous.get('state') != 'complete' or previous['provenance'] != provenance:
        raise ValueError('A completed F with the same E model and splits is required')
    path = stage_f/previous['best_model']
    if not path.resolve().is_relative_to((stage_f/'models').resolve()):
        raise ValueError('F model path outside models/')
    with np.load(path, allow_pickle=False) as archive:
        hellinger = json.loads(str(archive['metadata']))
    if (hellinger['kernel'] != 'hellinger' or
        hellinger['variant'] != previous['best_hellinger_variant'] or
        any(hellinger.get(k) != v for k, v in provenance.items())):
        raise ValueError('Invalid selected Hellinger model')
    output = project/config['output_dir']
    state = dict(state='running', config=config, provenance=provenance,
                 hellinger_model_sha256=file_signature(path), vocabulary_refitted=False,
                 subset_scope='SVM train only; vocabulary fixed from full internal train')
    save(output/'run.json', state)
    samples = {f'{fraction}:{seed}': subset(partitions['train'], fraction, seed)
               for fraction in fractions for seed in seeds}
    save(output/'train_subsets.json', samples)
    records = []
    for kernel, operations in [('linear', [provenance['e_normalization']]),
                               ('hellinger', hellinger['operations'])]:
        features = mapped(raw, operations)
        for fraction in sorted(fractions):
            for seed in seeds:
                ids = samples[f'{fraction}:{seed}']
                model = fit(np.stack([features[i] for i in ids]),
                    [partitions['train'][i] for i in ids], provenance['parameters'], seed)
                metrics = evaluate_parts(features, {s: partitions[s] for s in ('val', 'test')}, model)
                record = dict(kernel=kernel, fraction=fraction, seed=seed, train_images=len(ids),
                    train_positives=sum(partitions['train'][i]==1 for i in ids),
                    train_negatives=sum(partitions['train'][i]==-1 for i in ids),
                    operations=' -> '.join(operations), C=provenance['parameters']['C'],
                    fit_seconds=model['fit_seconds'], split_sha256=provenance['split_sha256'])
                for split, values in metrics.items():
                    record.update({f'{split}_{k}': v for k, v in values.items()})
                records.append(record)
                save_csv(output/'metrics.csv', records)
                print(f"{kernel}: fraction={fraction}, seed={seed}, n={len(ids)}, AP={record['test_ap']:.4f}", flush=True)
    summary = summarize(records)
    save_csv(output/'metrics_summary.csv', summary)
    plot(summary, output/'ap_vs_train_images.png')
    if split_digest(project/config['split_dir']) != provenance['split_sha256']:
        raise ValueError('Splits changed during G')
    state.update(state='complete', experiments=len(records), test_used_for_selection=False)
    save(output/'run.json', state)
    return state


def main():
    from ..artifact_json import load
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(load(args.config), args.project)


if __name__ == '__main__':
    main()
