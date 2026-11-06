"""Six global BoVW/VLAD/Fisher experiments, five target/background classifiers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from src.features import extractFeatures
from our_src.part1.artifact_json import load, save, file_signature
from our_src.part1.stagea.data import manifest
from our_src.part1.stagea.stage_a import binary_labels
from our_src.part1.stagea.svm import train_svm_grid
from our_src.part1.stagec.protocol import split_digest, smoke_rows
from our_src.part1.stagec.evaluation import evaluate
from our_src.part1.stagef.common import save_csv, save_model, score_metrics
from .encoders import create, encode, dimension
from .models import prepare
from .provenance import identity, DenseSIFT, resources
from .reporting import consolidate
from our_src.part1.artifact_json import key

EXPERIMENTS = [('E1','bovw',64,'l2'), ('E2','vlad',64,'intra_l2'), ('E3','fv',64,'power_l2'),
               ('E4','bovw',128,'l2'), ('E5','vlad',128,'intra_l2'), ('E6','fv',128,'power_l2')]


def run(config, project):
    project = Path(project)
    experiments=[r for r in EXPERIMENTS if r[1] in config.get('encoders',['bovw','vlad','fv'])]
    if not experiments: raise ValueError('No supported encoder selected')
    if config['vocabulary_sizes'] != [64,128] or set(config['categories']) != {'motorbike','aeroplane','horse','car','person'} or len(config['categories']) != 5:
        raise ValueError('Stage H requires K=64/128 and the five target classes')
    if config['images_per_class'] != -1 or config.get('pca_dim',0) != 0:
        raise ValueError('Stage H uses the full internal train sampling protocol, without PCA')
    if not set(config['curve_splits']) <= {'train','val','test'}:
        raise ValueError('Invalid curve split')
    digest = split_digest(project/config['split_dir'])
    all_rows = manifest(project/config['split_dir'])
    active = smoke_rows(all_rows, config)
    output = project/config['output_dir']
    provenance = identity(config, project, active, digest)
    baseline_reports=[]
    if config.get('stage_id')=='I':
        from .reporting import compatible_baseline
        baseline_reports=compatible_baseline(project/config['baseline_dir'],provenance)
        provenance['baseline_metrics_sha256']=file_signature(project/config['baseline_dir']/'metrics.csv')
    run_id = key(provenance)
    if (output/'run.json').exists():
        if not (output/'manifest.json').exists() or load(output/'manifest.json')['run_id'] != run_id:
            raise ValueError('Existing results have different/unknown provenance; preserve them and use --output-dir for a new run')
        previous=load(output/'run.json')
        if previous.get('state')=='complete':
            artifacts=load(output/'complete.json')['artifacts']
            if not all((output/p).is_file() and file_signature(output/p)==sha for p,sha in artifacts.items()):
                raise ValueError('Completed artifact checksum changed; preserve and inspect this run')
            print(f'Reuse completed H/I run {run_id}',flush=True)
            return previous
    save(output/'manifest.json',dict(run_id=run_id,**provenance))
    extractor=DenseSIFT(project,config)
    state = dict(run_id=run_id, state='running', started_utc=datetime.now(timezone.utc).isoformat(), config=config,
        split_sha256=digest, smoke_per_class=config.get('smoke_per_class'), pca_dim=0,
        tiling='global', test_used_for_selection=False, test_evaluated=False)
    save(output/'run.json', state)
    try:
        centers, gmms = prepare(config, project, all_rows, active, output, digest, extractor)
        preparation_sift_seconds=extractor.extraction_seconds
        preparation_cache_seconds=extractor.cache_read_seconds
        initial_hits,initial_misses=extractor.hits,extractor.misses
        encoders = {}
        provenance = {}
        for name, kind, k, norm in experiments:
            params = {'words': centers[k]['centroids']} if kind != 'fv' else gmms[k]['params']
            encoders[name] = create(kind, params, k)
            path = centers[k]['path'] if kind != 'fv' else gmms[k]['path']
            provenance[name] = dict(encoder_model=str(path), encoder_sha256=file_signature(path),
                encoder_fit_seconds=centers[k]['fit_seconds'] if kind!='fv' else gmms[k]['metadata']['fit_seconds'])
        ids = [r['image_id'] for r in active]
        index = {image_id:n for n,image_id in enumerate(ids)}
        reports, trials, selected = [], [], []
        # Disk-backed temporary encodings keep RAM bounded, while SIFT stays in RAM.
        with tempfile.TemporaryDirectory(dir=output, prefix='features_') as temporary:
            matrices = {}
            try:
                matrices = {name: np.lib.format.open_memmap(Path(temporary)/f'{name}.npy',mode='w+',
                    dtype=np.float32,shape=(len(ids),dimension(kind,k))) for name,kind,k,_ in experiments}
                encoding_seconds = {name:0. for name,_,_,_ in experiments}
                extraction_seconds = 0.
                for n, row in enumerate(active):
                    start = time.perf_counter()
                    _, descriptors, _ = extractor(project/config['image_dir']/f'{row["image_id"]}.jpg',cache_dir=None)
                    extraction_seconds += time.perf_counter()-start
                    descriptors = descriptors.astype(np.float32)
                    for name,kind,k,_ in experiments:
                        start = time.perf_counter()
                        matrices[name][n] = encode(encoders[name], descriptors, kind, k)
                        encoding_seconds[name] += time.perf_counter()-start
                    print(f"{config.get('stage_id','H/I')} encoding {n+1}/{len(active)}: {len(experiments)} experiments", flush=True)
                for name,kind,k,norm in experiments:
                    features = {image_id: matrices[name][index[image_id]] for image_id in ids}
                    for category in config['categories']:
                        partitions = {s:binary_labels(active,s,category) for s in ('train','val','test')}
                        result = train_svm_grid(features,partitions['train'],partitions['val'],
                            dict(C=config['C_values'],normalization=['none'],max_iter=[config['max_iter']],tol=[config['tol']]),seed=config['seed'])
                        best = result['best']
                        trials.extend(dict(experiment=name,encoder=kind,k=k,category=category,**r) for r in result['trials'])
                        save_csv(output/'validation_trials.csv', trials)
                        model_path = output/'models'/f'{category}__{name}.npz'
                        metadata = dict(experiment=name,encoder=kind,k=k,normalization=norm,pca_dim=0,
                            dimension=dimension(kind,k),tiling='global',category=category,negative_class='background',
                            C=best['C'],max_iter=config['max_iter'],tol=config['tol'],seed=config['seed'],
                            split_sha256=digest,smoke_per_class=config.get('smoke_per_class'),
                            training_split='train',selection_split='val',test_used_for_selection=False,
                            svm_fit_seconds=best['fit_seconds'],
                            svm_search_fit_seconds=sum(t['fit_seconds'] for t in result['trials']),
                            run_id=run_id,
                            **provenance[name])
                        save_model(model_path,best,metadata)
                        selected.append((metadata,partitions,model_path))
                    del features
                # Every model/C is frozen before looking at test metrics.
                for metadata, partitions, model_path in selected:
                    name, category = metadata['experiment'],metadata['category']
                    with np.load(model_path,allow_pickle=False) as archive:
                        weights,bias = archive['weights'],float(archive['bias'])
                    record = dict(**metadata,model=model_path.relative_to(output).as_posix(),
                        encoding_seconds_shared=encoding_seconds[name],sift_seconds_shared=extractor.extraction_seconds-preparation_sift_seconds,
                        sift_access_seconds_shared=extraction_seconds,
                        sift_cache_read_seconds_shared=extractor.cache_read_seconds-preparation_cache_seconds,
                        sift_cache_hits=extractor.hits-initial_hits,sift_cache_misses=extractor.misses-initial_misses,
                        encoder_preparation_sift_seconds_shared=preparation_sift_seconds)
                    for split, labels in partitions.items():
                        split_ids = sorted(labels)
                        start = time.perf_counter()
                        scores = np.stack([matrices[name][index[i]] for i in split_ids])@weights+bias
                        elapsed = time.perf_counter()-start
                        metrics = score_metrics([labels[i] for i in split_ids],scores)
                        record.update({f'{split}_{key}':value for key,value in metrics.items()})
                        record[f'{split}_inference_seconds'] = elapsed
                        if split in config['curve_splits']:
                            curve_dir = output/'curves'/category/name/split
                            evaluate(labels,dict(zip(split_ids,scores)),curve_dir,category,split)
                            ranking = output/'rankings'/category/name/f'{split}.csv'
                            ranking.parent.mkdir(parents=True,exist_ok=True)
                            (curve_dir/'ranking.csv').replace(ranking)
                    reports.append(record)
                    save_csv(output/'metrics.csv',reports)
                    print(f'H {category}/{name}: val AP={record["val_ap"]:.4f}, test AP={record["test_ap"]:.4f}',flush=True)
            finally:
                for matrix in matrices.values():
                    matrix.flush()
                    matrix._mmap.close()
        for category in config['categories']:
            best = max((r for r in reports if r['category']==category),key=lambda r:r['val_ap'])
            for record in reports:
                if record['category']==category:
                    record['selected_by_validation'] = record['experiment']==best['experiment']
        save_csv(output/'metrics.csv',reports)
        if split_digest(project/config['split_dir']) != digest:
            raise ValueError('Splits changed during Stage H')
        comparison_records=baseline_reports+reports
        consolidate(output,comparison_records,len(active),dict(run_id=run_id,stage_id=config.get('stage_id','H/I'),smoke_per_class=config.get('smoke_per_class'),split_sha256=digest))
        artifacts={str(p.relative_to(output)):file_signature(p) for p in output.rglob('*') if p.is_file() and p.name not in ('run.json','complete.json','preflight.json')}
        save(output/'complete.json',dict(run_id=run_id,artifacts=artifacts))
        state.update(state='complete',model_count=len(selected),test_evaluated=True,
                     completed_utc=datetime.now(timezone.utc).isoformat())
        save(output/'run.json',state)
        return state
    except BaseException as error:
        state.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
        save(output/'run.json',state)
        raise


def main(stage="H"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parents[1]/('stageh' if stage=='H' else 'stagei')/'config.json')
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--smoke-per-class',type=int)
    parser.add_argument('--output-dir',help='New output directory relative to --project; historical results are protected')
    parser.add_argument('--baseline-dir',help='Completed Stage H result directory for the Fisher comparison')
    parser.add_argument('--sift-cache',help='Optional source/image-addressed cache using provided Dense SIFT API')
    parser.add_argument('--preflight',action='store_true',help='Resource estimate only; no training or extraction')
    args = parser.parse_args()
    config = load(args.config)
    if args.output_dir: config['output_dir']=args.output_dir
    if args.baseline_dir: config['baseline_dir']=args.baseline_dir
    if args.sift_cache: config['sift_cache_dir']=args.sift_cache
    if args.smoke_per_class is not None:
        if args.smoke_per_class < 2:
            parser.error('--smoke-per-class must be at least 2')
        config.update(smoke_per_class=args.smoke_per_class,output_dir=str(Path(config['output_dir'])/'smoke'))
        if config.get('centroid_cache_dir'): config['centroid_cache_dir']=str(Path(config['centroid_cache_dir'])/'smoke')
        if config.get('baseline_dir'): config['baseline_dir']=str(Path(config['baseline_dir'])/'smoke')
    if not args.smoke_per_class:
        rows=manifest(args.project/config['split_dir'])
        preflight=resources(rows,config,args.project)
        save(args.project/config['output_dir']/'preflight.json',preflight)
        print(json.dumps(preflight,indent=2),flush=True)
        if args.preflight: return
        if not preflight['feasible']:
            raise SystemExit('Full benchmark not started: insufficient resource headroom; see preflight.json')
    run(config,args.project)


if __name__=='__main__':
    main()
