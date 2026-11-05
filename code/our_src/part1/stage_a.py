"""Stage A metrics and linear SVM baseline; in-memory local features/encodings."""
import argparse
import importlib.metadata
from datetime import datetime, timezone
import platform
from pathlib import Path
import time
import numpy as np
from src.features import extractFeatures
from src.encoders import KMeansPredictor
from .data import manifest
from .encoding import LAYOUTS, encode_image, feature_dimension, level_weights
from .metrics import evaluate_representation
from .results import load_centroids, recover_centroids, write_csv
from .svm import train_svm_grid
from ..a_json import load, save, file_signature


def binary_labels(rows, split, category):
    result = {}
    for row in rows:
        if row['split'] != split:
            continue
        labels = set(row['labels'])
        if category in labels and 'background' in labels:
            raise ValueError('An image is both target and designated background')
        if category in labels:
            result[row['image_id']] = 1
        elif 'background' in labels:
            result[row['image_id']] = -1
    return result


def run(config, project, revision=None, limit=None):
    project = Path(project)
    output = project/config['output_dir']
    # Smoke output is isolated so it never overwrites a full experiment.
    if limit is not None:
        output = output/'smoke'
    rows = manifest(project/config['split_dir'])
    model_folder = project/config['output_dir']/'centroids'
    if revision:
        recover_centroids(project, project/config['output_dir'], revision)
    models = load_centroids(model_folder,config,rows,project)
    active = [r for r in rows if r['split'] in ('train','val')]
    if limit is not None:
        # Include both target and background in each split for a useful smoke test.
        selected = set()
        for split in ('train','val'):
            for category in config['baseline_categories']+['background']:
                selected.update(r['image_id'] for r in [r for r in active if r['split']==split and category in r['labels']][:limit])
        active = [r for r in active if r['image_id'] in selected]
    if not active:
        raise ValueError('No images for this experiment')
    output.mkdir(parents=True,exist_ok=True)
    run_info = {'started_utc':datetime.now(timezone.utc).isoformat(), 'config':config,
                'smoke_limit_per_category_split':limit, 'state':'running',
                'platform':platform.platform(), 'test_evaluated':False,
                'image_ids':[r['image_id'] for r in active],
                'centroids':{str(k):file_signature(m['path']) for k,m in models.items()},
                'versions':{p:importlib.metadata.version(p) for p in
                            ('numpy','scikit-learn','pandas','cyvlfeat')},
                'code_signatures':{p.name:file_signature(p) for p in Path(__file__).parent.glob('*.py')},
                'stability':'not_evaluated_single_seed',
                'timing_note':'shared extraction/assignment/count times must not be summed over layouts',
                'separability_note':'cosine distance after level weights and L2; intra=shared label, inter=disjoint labels'}
    save(output/'run.json',run_info)
    # normal and pyramid_L0 are aliases, sharing one array in RAM.
    features = {}
    for k in models:
        for name in LAYOUTS:
            if name != 'pyramid_L0':
                features[k,name] = np.empty((len(active),feature_dimension(name,k)),dtype=np.uint32)
        features[k,'pyramid_L0'] = features[k,'normal']
    predictors = {k:KMeansPredictor(m['centroids']) for k,m in models.items()}
    stats = {k:[] for k in models}
    extraction_seconds = []
    start_run = time.perf_counter()
    for index,row in enumerate(active):
        start = time.perf_counter()
        points,desc,shape = extractFeatures(project/config['image_dir']/f"{row['image_id']}.jpg",cache_dir=None)
        extraction_seconds.append(time.perf_counter()-start)
        desc = desc.astype(np.float32)
        for k,predictor in predictors.items():
            vectors,measurements = encode_image(points,desc,shape,predictor,k,config['assignment_batch_size'])
            stats[k].append(measurements)
            for name,vector in vectors.items():
                features[k,name][index] = vector
        del points,desc
        print(f"Encoded {index+1}/{len(active)} images, all K/layouts",flush=True)
    metric_rows,svm_rows,best = [],[],{}
    for k,model in models.items():
        for name in LAYOUTS:
            matrix = features[k,name]
            metric_rows.extend(evaluate_representation(matrix,active,k,name,stats[k],model,
                              extraction_seconds,config['seed'],config['separability_max_pairs']))
            write_csv(output/'metrics.csv',metric_rows)
            # Apply pyramid weights; API handles the requested normalization.
            weighted = matrix.astype(np.float32)*level_weights(name,k)
            mapping = {r['image_id']:weighted[i] for i,r in enumerate(active)}
            for category in config['baseline_categories']:
                result = train_svm_grid(mapping,binary_labels(active,'train',category),
                            binary_labels(active,'val',category),config['svm_grid'],seed=config['seed'])
                for trial in result['trials']:
                    svm_rows.append({'category':category,'k':k,'layout':name,'seed':config['seed'],
                                     'selected_within_encoding':trial['C']==result['best']['C'] and
                                       trial['normalization']==result['best']['normalization'] and
                                       trial['max_iter']==result['best']['max_iter'] and trial['tol']==result['best']['tol'],
                                     'test_ap':None, **trial})
                candidate = result['best']
                if category not in best or candidate['validation_ap']>best[category]['validation_ap']:
                    best[category]={key:value for key,value in candidate.items() if key not in ('weights','bias')}
                    best[category].update(k=k,layout=name)
                write_csv(output/'svm_ap.csv',svm_rows)
                print(f"K={k} {name} {category}: best validation AP={candidate['validation_ap']:.4f}",flush=True)
    run_info.update(state='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-start_run,best_validation=best)
    save(output/'run.json',run_info)
    print(f'Complete: {output}/metrics.csv and svm_ap.csv',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('stage_a.json'))
    parser.add_argument('--recover-centroids-from',help='Explicit local Git commit hash with historical centroid files')
    parser.add_argument('--smoke-per-category',type=int,help='Bounded run; outputs go to stageA/smoke')
    args=parser.parse_args()
    if args.smoke_per_category is not None and args.smoke_per_category<1:
        parser.error('--smoke-per-category must be positive')
    config = load(args.config)
    project = Path(__file__).resolve().parents[2]
    try:
        run(config,project,args.recover_centroids_from,args.smoke_per_category)
    except (Exception,KeyboardInterrupt) as error:
        output = project/config['output_dir']
        if args.smoke_per_category is not None:
            output = output/'smoke'
        status_path = output/'run.json'
        if status_path.exists():
            status = load(status_path)
            status.update(state='failed',error=repr(error),stopped_utc=datetime.now(timezone.utc).isoformat())
            save(status_path,status)
        raise


if __name__=='__main__':
    main()
