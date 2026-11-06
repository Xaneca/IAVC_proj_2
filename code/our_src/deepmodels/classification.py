"""Shared C selection via the existing binary SVM grid; true multilabel evaluation."""
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import (average_precision_score, accuracy_score, f1_score, confusion_matrix,
    multilabel_confusion_matrix, precision_recall_fscore_support)
from .common import module, npz, save, read, digest, file_hash


def partitions(rows, category, background_only=False):
    return {split: {r['image_id']: (1 if category in r['labels'] else -1)
        for r in rows if r['split'] == split and (not background_only or
            category in r['labels'] or 'background' in r['labels'])}
        for split in ('train', 'val', 'test')}


def search(features, rows, config, project, seed, background_only=False, trial_cache=None):
    """Independent category C selection by validation AP; lowest C wins ties.

    Optional content-addressed singleton trials are shared between C and D.
    Neither test features nor test labels participate in trial identities.
    """
    trainer = module('svm', project).train_svm_grid
    categories = [c for c in config['categories'] if c != 'background']
    fitted, trials = [], []
    for c_value in sorted(config['C_values']):
        models, records = {}, []
        for c in categories:
            parts = partitions(rows, c, background_only)
            params = dict(C=[c_value], normalization=['none'], max_iter=[config['max_iter']], tol=[config['tol']])
            cache = None
            if trial_cache is not None:
                import hashlib
                h = hashlib.sha256()
                for split in ('train', 'val'):
                    for image_id, label in sorted(parts[split].items()):
                        h.update(f'{split}:{image_id}:{label}'.encode())
                        h.update(np.asarray(features[image_id], dtype=np.float32).tobytes())
                identity = digest(dict(data=h.hexdigest(), category=c, params=params, seed=seed,
                    trainer=file_hash(module('svm', project).__file__), provider=file_hash(Path(project)/'src/utils.py')))
                cache = Path(trial_cache)/identity
            if cache is not None and cache.with_suffix('.json').exists():
                record = read(cache.with_suffix('.json'))
                if file_hash(cache.with_suffix('.npz')) != record['sha256']:
                    raise ValueError('Corrupt cached SVM trial')
                with np.load(cache.with_suffix('.npz'), allow_pickle=False) as z:
                    best = dict(record['trial'], weights=z['weights'].copy(), bias=float(z['bias']))
            else:
                result = trainer(features, parts['train'], parts['val'], params, seed=seed)
                best = result['best']
                if cache is not None:
                    npz(cache.with_suffix('.npz'), weights=best['weights'], bias=np.array(best['bias']))
                    save(cache.with_suffix('.json'), dict(trial=result['trials'][0],sha256=file_hash(cache.with_suffix('.npz'))))
            models[c] = best
            records.append(dict(category=c, **{k:v for k,v in best.items() if k not in ('weights','bias')}))
        trial = dict(C=float(c_value), train_map=float(np.mean([r['train_ap'] for r in records])),
            validation_map=float(np.mean([r['validation_ap'] for r in records])),
            classifier_training_time=sum(r['fit_seconds'] for r in records), per_class=records)
        fitted.append(models)
        trials.append(trial)
    indices = {c:max(range(len(trials)), key=lambda i: fitted[i][c]['validation_ap']) for c in categories}
    selected = dict(C_by_category={c:trials[i]['C'] for c,i in indices.items()},
        validation_map=float(np.mean([fitted[i][c]['validation_ap'] for c,i in indices.items()])),
        train_map=float(np.mean([fitted[i][c]['train_ap'] for c,i in indices.items()])),
        criterion='independent validation AP per category; lowest C wins ties', test_used_for_selection=False)
    return dict(models={c:fitted[i][c] for c,i in indices.items()}, trials=trials, selected=selected,
                protocol='target_vs_background' if background_only else 'multilabel_one_vs_rest')


def persist_binary(features, rows, result, output, project, final=False, curves=True):
    """Persist the controlled protocol without evaluating other-target negatives."""
    output = Path(output)
    reports = {}
    for c, model in result['models'].items():
        destination = output/c
        npz(destination/'model.npz', weights=model['weights'], bias=np.array(model['bias']))
        reports[c] = {}
        for split, labels in partitions(rows, c, True).items():
            if split == 'test' and not final:
                continue
            start = time.perf_counter()
            scores = {i:float(features[i] @ model['weights'] + model['bias']) for i in labels}
            seconds = time.perf_counter()-start
            if curves:
                report = module('evaluation',project).evaluate(labels,scores,destination/split,c,split)
            else:
                report = dict(ap=float(average_precision_score(list(labels.values()),list(scores.values()))),
                    images=len(labels),positives=sum(v==1 for v in labels.values()),negatives=sum(v==-1 for v in labels.values()))
            report['inference_seconds'] = seconds
            truth=np.array(list(labels.values()))
            predicted=np.where(np.array(list(scores.values()))>=0,1,-1)
            precision, recall, f1, _=precision_recall_fscore_support(truth,predicted,labels=[1],zero_division=0)
            report.update(precision=float(precision[0]),recall=float(recall[0]),f1=float(f1[0]),
                confusion_matrix=confusion_matrix(truth,predicted,labels=[-1,1]).tolist(),
                confusion_definition='[TN, FP; FN, TP]; target/background membership only')
            reports[c][split] = report
    save(output/'search.json', dict(trials=result['trials'],selected=result['selected'],protocol=result['protocol']))
    save(output/'metrics.json', reports)
    if curves:
        categories=list(result['models'])
        weights=np.stack([result['models'][c]['weights'] for c in categories])
        bias=np.array([result['models'][c]['bias'] for c in categories])
        npz(output/'model.npz',weights=weights,bias=bias,categories=np.array(categories))
        for split in ('train','val','test') if final else ('train','val'):
            active=[r for r in rows if r['split']==split]
            scores=np.stack([features[r['image_id']] for r in active])@weights.T+bias
            npz(output/(split+'_predictions.npz'),image_ids=np.array([r['image_id'] for r in active]),
                labels=np.array([[c in r['labels'] for c in categories] for r in active]),
                categories=np.array(categories),scores=scores,predictions=scores>=0,
                protocol=np.array('independently fitted target/background models; full-image scores for qualitative analysis'))
        plot_search(result['trials'], output/'regularization.png')
    return reports


def metrics(rows, categories, scores, forced_single=False):
    y = np.array([[c in r['labels'] for c in categories] for r in rows], dtype=int)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite decision scores')
    if any(len(set(y[:, j])) != 2 for j in range(y.shape[1])):
        raise ValueError('Each evaluated category needs positive and negative images')
    if forced_single:
        predicted = np.eye(len(categories), dtype=int)[np.argmax(scores, axis=1)]
    else:
        predicted = (scores >= 0).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, predicted, average=None, zero_division=0)
    aps = {c:float(average_precision_score(y[:, j], scores[:, j])) for j,c in enumerate(categories)}
    return dict(accuracy=float(accuracy_score(y, predicted)), accuracy_definition='exact label-set match',
        top1_label_membership_accuracy=float(np.mean(y[np.arange(len(y)), np.argmax(scores, axis=1)])),
        macro_f1=float(f1_score(y, predicted, average='macro', zero_division=0)),
        per_class_ap=aps, map=float(np.mean([v for c,v in aps.items() if c != 'background'])),
        per_class_precision=dict(zip(categories, map(float,p))), per_class_recall=dict(zip(categories,map(float,r))),
        per_class_f1=dict(zip(categories,map(float,f))),
        confusion_matrix=multilabel_confusion_matrix(y,predicted).tolist(),
        confusion_definition='one 2x2 [TN,FP;FN,TP] matrix per category',
        images=len(y), forced_single_prediction=forced_single), predicted


def persist_scores(features, rows, result, output, config, project, final=False):
    output = Path(output)
    categories = list(result['models'])
    npz(output/'model.npz', weights=np.stack([result['models'][c]['weights'] for c in categories]),
        bias=np.array([result['models'][c]['bias'] for c in categories]), categories=np.array(categories))
    save(output/'search.json', dict(selected=result['selected'], trials=result['trials'], protocol=result['protocol']))
    reports = {}
    for split in ('train','val','test') if final else ('train','val'):
        active = [r for r in rows if r['split'] == split]
        start = time.perf_counter()
        scores = np.stack([[features[r['image_id']] @ result['models'][c]['weights'] + result['models'][c]['bias']
                            for c in categories] for r in active])
        seconds = time.perf_counter()-start
        report, predictions = metrics(active, categories, scores)
        report['inference_time'] = seconds
        reports[split] = report
        npz(output/(split + '_predictions.npz'), image_ids=np.array([r['image_id'] for r in active]),
            labels=np.array([[c in r['labels'] for c in categories] for r in active]),
            scores=scores, predictions=predictions, categories=np.array(categories))
        # Reuse existing AP/PR/ranking persistence; title convention is background,
        # so write only the genuinely matching target/background subsets here.
        binary = {}
        for j,c in enumerate(categories):
            labels = partitions(active,c,True)[split]
            subset_scores = {r['image_id']:float(scores[i,j]) for i,r in enumerate(active) if r['image_id'] in labels}
            binary[c] = module('evaluation',project).evaluate(labels,subset_scores,output/'binary_background'/split/c,c,split)['ap']
        reports[split]['binary_background_ap_on_ovr_model'] = binary
    save(output/'metrics.json', reports)
    plot_search(result['trials'], output/'regularization.png')
    return reports


def plot_search(trials, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for key in ('train_map','validation_map'):
        ax.plot([r['C'] for r in trials],[r[key] for r in trials],marker='o',label=key)
    ax.set(xscale='log',xlabel='SVM C',ylabel='mAP (target categories)',ylim=(0,1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(destination,dpi=150); plt.close(fig)
