"""Nested multilabel training subsets with explicit background policy."""
from pathlib import Path
import numpy as np
from .common import save
from .classification import search, persist_binary


def multilabel_subsets(rows, categories, counts, seed):
    """N anchors per label, unioned without dropping secondary labels.

    Background receives N anchors too. Other target categories are OvR negatives.
    Co-occurring labels can make actual positives >N; persist actual counts.
    Each class permutation is fixed within seed, making the union nested.
    """
    train = [r for r in rows if r['split'] == 'train']
    rng = np.random.default_rng(seed)
    pools = {c: rng.permutation(sorted(r['image_id'] for r in train if c in r['labels'])).tolist()
             for c in categories}
    for count in counts:
        if count != 'all' and (not isinstance(count,int) or count < 1 or any(len(p)<count for p in pools.values())):
            raise ValueError(f'Not enough training anchors per category for N={count}')
        selected = set().union(*(set(p if count == 'all' else p[:count]) for p in pools.values()))
        active = [r for r in train if r['image_id'] in selected]
        if any(not any(c in r['labels'] for r in active) or all(c in r['labels'] for r in active) for c in categories):
            raise ValueError('Few-shot subset lacks positive/negative examples')
        yield count, active, {c:dict(positives=sum(c in r['labels'] for r in active),
            negatives=sum(c not in r['labels'] for r in active)) for c in categories}


def subsets(rows, categories, counts, seed):
    """Independent binary nested prefixes; all original training background negatives."""
    train = [r for r in rows if r['split']=='train']
    negatives = sorted((r for r in train if 'background' in r['labels']), key=lambda r:r['image_id'])
    if not negatives:
        raise ValueError('No training background negatives')
    for category in categories:
        if category == 'background':
            continue
        # Independent per-category generators keep membership stable if order changes.
        import hashlib
        salt = int.from_bytes(hashlib.sha256(category.encode()).digest()[:4], 'little')
        rng = np.random.default_rng(np.random.SeedSequence([seed, salt]))
        pool = sorted((r for r in train if category in r['labels']), key=lambda r:r['image_id'])
        permutation = rng.permutation(len(pool))
        for count in counts:
            if count != 'all' and (not isinstance(count,int) or count<1 or count>len(pool)):
                raise ValueError(f'Not enough training positives for {category}, N={count}')
            chosen = permutation if count=='all' else permutation[:count]
            positives = [pool[i] for i in chosen]
            yield category, count, positives+negatives, dict(positives=len(positives),negatives=len(negatives))


def run(representation, flipped, rows, config, project, output, trial_cache=None, final=False):
    """C/D share singleton trials; training AP includes augmented duplicates."""
    from .common import read
    output = Path(output)
    records = []
    for seed in config['fewshot_seeds']:
        for category, count, selected, counts in subsets(rows, config['categories'], config['positive_counts'], seed):
            for augmentation in config.get('augmentations', ['none','horizontal_flip']):
                destination = output/category/f'seed_{seed}'/str(count)/augmentation
                if (destination/'complete.json').exists():
                    marker = read(destination/'complete.json')
                    from .common import file_hash
                    if not all((destination/p).is_file() and file_hash(destination/p)==sha for p,sha in marker['artifacts'].items()):
                        raise ValueError('Few-shot artifacts changed; preserve and inspect before resuming')
                    records.append(marker['record']); continue
                features = dict(representation['features'])
                active = selected + [r for r in rows if r['split']!='train']
                if augmentation=='horizontal_flip':
                    if flipped is None:
                        raise ValueError('Flipped training features required')
                    for r in selected:
                        image_id = r['image_id']+'::horizontal_flip'
                        features[image_id] = flipped['features'][r['image_id']]
                        active.append(dict(r,image_id=image_id))
                elif augmentation!='none':
                    raise ValueError('Unsupported augmentation')
                cfg = dict(config,categories=[category,'background'])
                result = search(features,active,cfg,project,seed,True,trial_cache)
                reports = persist_binary(features,active,result,destination,project,final,curves=False)
                record = dict(category=category,seed=seed,n_positive_requested=count,augmentation=augmentation,
                    representation=representation.get('layer'),feature_identity=representation.get('identity'),
                    image_ids=[r['image_id'] for r in selected],actual_counts=counts,
                    negative_policy='all original training background images; fixed across N and seeds',
                    training_vectors=sum(r['split']=='train' for r in active),
                    training_ap_includes_augmented_duplicates=augmentation=='horizontal_flip',
                    C=result['selected']['C_by_category'][category],
                    train_ap=reports[category]['train']['ap'],validation_ap=reports[category]['val']['ap'],
                    test_ap=reports[category].get('test',{}).get('ap'),
                    trials=[dict(t['per_class'][0],representation=representation.get('layer'),augmentation=augmentation,
                        seed=seed,n_positive_requested=count,feature_identity=representation.get('identity')) for t in result['trials']])
                from .common import file_hash
                artifacts={str(p.relative_to(destination)):file_hash(p) for p in destination.rglob('*') if p.is_file()}
                save(destination/'complete.json',dict(record=record,artifacts=artifacts))
                records.append(record)
    save(output/'results.json',records)
    summary=[]
    categories=[c for c in config['categories'] if c!='background']
    for count in config['positive_counts']:
        for aug in config.get('augmentations',['none','horizontal_flip']):
            values=[r for r in records if r['n_positive_requested']==count and r['augmentation']==aug]
            seed_maps=[np.mean([r['validation_ap'] for r in values if r['seed']==s]) for s in config['fewshot_seeds']]
            summary.append(dict(n=count,augmentation=aug,mean=float(np.mean(seed_maps)),
                std=float(np.std(seed_maps,ddof=1)) if len(seed_maps)>1 else 0.,seeds=len(seed_maps),
                per_class={c:dict(mean=float(np.mean([r['validation_ap'] for r in values if r['category']==c])),
                    std=float(np.std([r['validation_ap'] for r in values if r['category']==c],ddof=1)) if len(seed_maps)>1 else 0.) for c in categories}))
    save(output/'summary.json',summary)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for aug in config.get('augmentations',['none','horizontal_flip']):
        values=[r for r in summary if r['augmentation']==aug]
        ax.errorbar(range(len(values)),[r['mean'] for r in values],yerr=[r['std'] for r in values],label=aug,marker='o')
    ax.set(xticks=range(len(config['positive_counts'])),xticklabels=config['positive_counts'],
        xlabel='Positive original training images per binary category',ylabel='Validation mAP (target/background)')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'learning_curve.png',dpi=150); plt.close(fig)
    # Regularization across seeds for each N/category; no cross-category score mixing.
    for category in categories:
        for count in config['positive_counts']:
            fig,ax=plt.subplots()
            for aug in config.get('augmentations',['none','horizontal_flip']):
                selected=[r for r in records if r['category']==category and r['n_positive_requested']==count and r['augmentation']==aug]
                grid=sorted(config['C_values'])
                for metric in ('train_ap','validation_ap'):
                    values=[[t[metric] for r in selected for t in r['trials'] if t['C']==c] for c in grid]
                    ax.errorbar(grid,[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) if len(v)>1 else 0 for v in values],label=f'{aug} {metric}',marker='o')
            ax.set(xscale='log',xlabel='SVM C',ylabel='AP',ylim=(0,1.02),title=f'{category}: N={count}')
            ax.legend(); fig.tight_layout(); fig.savefig(output/f'regularization_{category}_{count}.png',dpi=150); plt.close(fig)
    return records

