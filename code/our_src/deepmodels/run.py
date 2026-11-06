"""Independent JSON-configured Part 4 runner. Test evaluation requires --final."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .common import dataset, digest, file_hash, image_path, module, read, save, npz, l2
from .features import Backbone, extract
from .classification import search, persist_scores, plot_search, partitions, persist_binary


def signature(project, config, rows, smoke, final):
    images = {r['image_id']:file_hash(image_path(project,config,r['image_id'])) for r in rows}
    files = {p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    files['src/deep_models.py'] = file_hash(Path(project)/'src/deep_models.py')
    files['src/utils.py'] = file_hash(Path(project)/'src/utils.py')
    for name in ('data','svm','evaluation','protocol'):
        m = module(name,project)
        files[name] = file_hash(m.__file__)
    import importlib.metadata as metadata
    versions = {}
    for package in ('numpy','scikit-learn','torch','torchvision','clip','Pillow'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return dict(config=config, rows=rows, image_sha256=images, source_sha256=files,
                versions=versions, smoke=smoke, final=final)


def classify(rep, rows, config, project, output, final):
    result = search(rep['features'],rows,config,project,config['seed'])
    metrics = persist_scores(rep['features'],rows,result,output,config,project,final)
    # Matching Part 1 task: fit separate category/background models, not OvR
    # models evaluated only on a smaller test subset.
    binary = search(rep['features'],rows,config,project,config['seed'],background_only=True)
    persist_binary(rep['features'],rows,binary,Path(output)/'matching_classical_protocol',project,final)
    save(Path(output)/'representation.json',{k:v for k,v in rep.items() if k not in ('features',)})
    return metrics


def report(root, project, split_sha, smoke):
    """Consolidate only completed, compatible results; never invent baselines."""
    records = []
    for path in (root/'classification').glob('*/metrics.json'):
        if not (path.parent/'complete.json').exists():
            continue
        meta = read(path.parent/'representation.json')
        for split, metrics in read(path).items():
            records.append(dict(method=path.parent.name,supervision='frozen pretrained + task labels',split=split,
                protocol='multilabel_one_vs_rest',map=metrics['map'],accuracy=metrics['accuracy'],
                macro_f1=metrics['macro_f1'],dimension=meta['dimension'],storage_bytes=meta['storage_bytes'],
                feature_extraction_time=sum(sum(t.values()) for t in meta['timings'].values()),
                inference_time=metrics['inference_time']))
        binary = read(path.parent/'matching_classical_protocol/metrics.json')
        binary_search=read(path.parent/'matching_classical_protocol/search.json')
        for split in read(path):
            records.append(dict(method=path.parent.name,supervision='frozen pretrained + task labels',split=split,
                protocol='target_vs_background',map=float(np.mean([v[split]['ap'] for v in binary.values()])),
                dimension=meta['dimension'],per_class_ap={c:v[split]['ap'] for c,v in binary.items()},
                selected_C=binary_search['selected']['C_by_category'],
                descriptor_bytes_per_image=meta.get('descriptor_bytes_per_image'),
                cache_storage_bytes_shared=meta['storage_bytes'],
                feature_extraction_time_shared=sum(sum(t.values()) for t in meta['timings'].values()),
                extraction_cost_scope=meta.get('extraction_cost_scope'),
                timing_components_shared={k:sum(t.get(k,0.) for t in meta['timings'].values()) for k in next(iter(meta['timings'].values()))},
                classifier_training_time=sum(r['fit_seconds'] for t in binary_search['trials'] for r in t['per_class']),
                inference_time=sum(v[split]['inference_seconds'] for v in binary.values())))
    for path in (root/'clip').glob('*/metrics.json'):
        for split, metrics in read(path).items():
            records.append(dict(method='clip_'+path.parent.name,supervision='image-text pretraining; zero task labels',
                split=split,protocol='full_dataset_classwise_text_retrieval',map=metrics['map'],accuracy=metrics['accuracy'],
                per_class_ap=metrics['per_class_ap'],macro_f1=metrics['macro_f1'],dimension=512,inference_time=metrics['similarity_time'],
                prediction_policy='single highest-similarity category including background'))
    excluded = []
    # Classical results differ in schemas and may be incomplete. Import only
    # canonical H rows with matching split hashes and every target category.
    import csv
    for stage in ('stageA','stageB','stageC','stageD','stageE','stageF','stageG','stageH'):
        state_path = Path(project)/'results'/'part1'/stage/'run.json'
        if not state_path.exists():
            state_path = Path(project)/stage/'run.json'  # Legacy checkout layout
        if not state_path.exists():
            excluded.append(dict(stage=stage,reason='no full run manifest')); continue
        state = read(state_path)
        path = state_path.parent/'metrics.csv'
        if smoke or state.get('state')!='complete' or state.get('split_sha256')!=split_sha or state.get('smoke_per_class') is not None:
            excluded.append(dict(stage=stage,reason='incomplete, smoke, or incompatible split provenance')); continue
        if stage != 'stageH' or not path.exists():
            excluded.append(dict(stage=stage,reason='no compatible multi-category representation comparison schema')); continue
        if state.get('dataset_image_sha256') != read(root/'manifest.json')['image_sha256']:
            excluded.append(dict(stage=stage,reason='dataset image-content provenance unavailable or incompatible')); continue
        values = list(csv.DictReader(path.open(encoding='utf-8-sig')))
        for experiment in sorted({r['experiment'] for r in values}):
            group = [r for r in values if r['experiment']==experiment]
            if {r['category'] for r in group} != {'motorbike','aeroplane','horse','car','person'} or len(group)!=5:
                continue
            if any(float(r.get('C',0)) not in (.1,1,10) or r.get('normalization') not in ('l2','intra_l2','power_l2') for r in group):
                excluded.append(dict(stage=stage,experiment=experiment,reason='incompatible SVM grid/normalization')); continue
            for split in ('train','val','test'):
                if split=='test' and not read(root/'manifest.json')['final']:
                    continue
                if all(r.get(split+'_ap') for r in group):
                    records.append(dict(method='classical_'+experiment,supervision='task labels',protocol='target_vs_background',
                        split=split,map=float(np.mean([float(r[split+'_ap']) for r in group])),
                        per_class_ap={r['category']:float(r[split+'_ap']) for r in group},
                        selected_C={r['category']:float(r['C']) for r in group},dimension=int(group[0]['dimension']),source=str(path),timing_comparison='not assumed comparable across hardware/runs'))
    save(root/'comparison.json',dict(records=records,excluded_classical=excluded,
        caveats=['Compare mAP only within the same protocol.', 'Feature size alone does not explain AP.',
                 'Shared multi-layer extraction time/storage must not be added across ResNet layers.']))
    expected={'BoVW_K64':64,'BoVW_K128':128,'VLAD_K64':8192,'VLAD_K128':16384,
        'Fisher_K64':16448,'Fisher_K128':32896,
        'resnet18_layer2':128,'resnet18_layer3':256,'resnet18_layer4':512,
        'dinov2_vits14_global':384,'dinov2_vits14_patch_mean':384,
        'dinov2_vits14_spatial2x2':1536,'clip_vit_b_32_global':512}
    aliases={'BoVW_K64':'classical_E1','VLAD_K64':'classical_E2','Fisher_K64':'classical_E3',
        'BoVW_K128':'classical_E4','VLAD_K128':'classical_E5','Fisher_K128':'classical_E6'}
    wide=[]
    for name,dimension in expected.items():
        matched={r['split']:r for r in records if r['method']==aliases.get(name,name) and r['protocol']=='target_vs_background'}
        validation=matched.get('val',{})
        wide.append(dict(representation=name,dimension=dimension,protocol='target_vs_background',
            validation_map=validation.get('map'),test_map=matched.get('test',{}).get('map'),
            validation_ap_per_category=validation.get('per_class_ap'),test_ap_per_category=matched.get('test',{}).get('per_class_ap'),
            selected_C=validation.get('selected_C'),extraction_time_shared=validation.get('feature_extraction_time_shared'),
            descriptor_bytes_per_image=4*dimension,cache_storage_bytes_shared=validation.get('cache_storage_bytes_shared'),
            status='EXECUTED technical smoke' if matched and smoke else 'EXECUTED' if matched else 'PENDING / no compatible completed result'))
    save(root/'comparison_table.json',wide)
    module('evaluation',project).write_csv(root/'comparison_table.csv',wide)
    if records:
        import matplotlib.pyplot as plt
        for protocol in ('multilabel_one_vs_rest','target_vs_background'):
            selected=[r for r in records if r['split']=='val' and r['protocol']==protocol]
            if not selected:
                continue
            fig,ax=plt.subplots(figsize=(10,5))
            ax.barh([r['method'] for r in selected],[r['map'] for r in selected])
            ax.set(xlabel='Validation mAP',xlim=(0,1),title=protocol)
            fig.tight_layout(); fig.savefig(root/(protocol+'_comparison.png'),dpi=150); plt.close(fig)
    if records:
        module('evaluation',project).write_csv(root/'comparison.csv',
            [{key:r.get(key,'') for key in sorted(set().union(*(r.keys() for r in records)))} for r in records])


def execute(args):
    project = args.project.resolve()
    config = read(args.config)
    rows, split_sha = dataset(project,config,args.smoke)
    if args.smoke:
        config['positive_counts'] = [1,'all']
        config['fewshot_seeds'] = [0,1,2,3,4]
        config['visual_examples'] = 1
        config['occlusion_grid'] = 2
    identity = signature(project,config,rows,args.smoke,args.final)
    experiment_id = digest(identity)[:20]
    base = args.output.resolve() if args.output else project/config['output_dir']
    root = base/('smoke' if args.smoke else 'runs')/experiment_id
    if args.dry_run:
        print(f'Validated {len(rows)} images; split SHA256 {split_sha}; output {root}')
        print('Stages: '+','.join(args.stages)+'; final test evaluation: '+str(args.final))
        return
    root.mkdir(parents=True,exist_ok=True)
    save(root/'manifest.json',dict(experiment_id=experiment_id,split_sha256=split_sha,**identity))
    save(root/'config.json',config)
    registry = read(root/'representations.json') if (root/'representations.json').exists() else {}
    backbones = {}
    def backbone(name):
        if name not in backbones:
            backbones[name] = Backbone(project,name,config)
        return backbones[name]
    def representation(name):
        if name not in registry:
            raise ValueError(f'Missing representation {name}; run A and B first with the same options')
        meta = registry[name]
        features = {}
        for image_id,path in meta['files'].items():
            if file_hash(base/path)!=meta['file_sha256'][path]:
                raise ValueError(f'Feature checksum changed: {base/path}')
            with np.load(base/path,allow_pickle=False) as z:
                features[image_id] = z[meta['layer']+'_l2'].copy()
        return dict(meta,features=features)
    def register(representations):
        for name,rep in representations.items():
            cache = base/'features'/rep['metadata']['model']/rep['identity']/'manifest.json'
            meta = {k:v for k,v in rep.items() if k!='features'}
            meta.update({k:read(cache)[k] for k in ('files','file_sha256')})
            registry[name] = meta
        save(root/'representations.json',registry)
    stage_handlers = {}
    def ab(stage):
        model = 'resnet18' if stage=='A' else 'dinov2_vits14'
        reps = extract(backbone(model),rows,project,config,base)
        register(reps)
        for name,rep in reps.items():
            destination=root/'classification'/name
            if not (destination/'complete.json').exists():
                classify(rep,rows,config,project,destination,args.final)
                save(destination/'complete.json',dict(state='complete',feature_identity=rep['identity']))
    def choose():
        candidates = []
        for p in (root/'classification').glob('*/matching_classical_protocol/metrics.json'):
            if p.parents[1].name.startswith(('resnet18_', 'dinov2_')):
                candidates.append((float(np.mean([v['val']['ap'] for v in read(p).values()])),p.parents[1].name))
        if len(candidates)!=6:
            raise ValueError('Run all three ResNet and three DINO representations before C')
        name = sorted(candidates,key=lambda x:(-x[0],x[1]))[0][1]
        return name, candidates
    def c():
        from .fewshot import run
        name,candidates=choose()
        rep=representation(name)
        save(root/'fewshot/selection.json',dict(representation=name,validation_metric='target/background mAP',
            reason='highest validation mAP; lexical ascending representation name breaks ties',
            candidates=candidates,test_used_for_selection=False))
        selected_backbone=backbone(rep['metadata']['model'])
        if selected_backbone.metadata!=rep['metadata']:
            raise ValueError('Few-shot backbone provenance mismatch')
        flips=extract(selected_backbone,rows,project,config,base,flip=True)[name]
        run(rep,flips,rows,config,project,root/'fewshot',root/'svm_trials',args.final)
    def d():
        from .fewshot import run
        name=read(root/'fewshot/selection.json')['representation']
        rep=representation(name)
        extended=dict(config,C_values=config['regularization_C_values'],augmentations=['none'])
        run(rep,None,rows,extended,project,root/'regularization',root/'svm_trials',args.final)
    def e():
        from .clip_experiments import run
        b=backbone('clip_vit_b_32')
        reps=extract(b,rows,project,config,base)
        register(reps)
        run(b,reps['clip_vit_b_32_global'],rows,config,project,root/'clip',args.final)
        destination=root/'classification/clip_vit_b_32_global'
        classify(reps['clip_vit_b_32_global'],rows,config,project,destination,args.final)
        save(destination/'complete.json',dict(state='complete'))
    def f():
        from .visual_analysis import run,load_predictions
        split='test' if args.final else 'val'
        cnn_candidates=[(float(np.mean([v['val']['ap'] for v in read(root/'classification'/name/'matching_classical_protocol/metrics.json').values()])),name)
            for name in registry if name.startswith('resnet18_')]
        if len(cnn_candidates)!=3:
            raise ValueError('Stage F needs A/B/E outputs')
        cnn=max(sorted(cnn_candidates),key=lambda x:x[0])[1]
        dino=max(sorted((float(np.mean([v['val']['ap'] for v in read(root/'classification'/name/'matching_classical_protocol/metrics.json').values()])),name) for name in registry if name.startswith('dinov2_')),key=lambda x:x[0])[1]
        names={'cnn':cnn,'dino':dino}
        predictions={k:load_predictions(root/'classification'/v/'matching_classical_protocol'/(split+'_predictions.npz')) for k,v in names.items()}
        predictions['clip']=load_predictions(root/'clip/selected'/(split+'_predictions.npz'))
        models={}
        scorers={}
        for key,name in names.items():
            models[key]=backbone(registry[name]['metadata']['model'])
            if models[key].metadata!=registry[name]['metadata']:
                raise ValueError('Visual-analysis backbone differs from cached feature provenance')
            with np.load(root/'classification'/name/'matching_classical_protocol/model.npz',allow_pickle=False) as z:
                weights,bias,categories=z['weights'].copy(),z['bias'].copy(),z['categories'].tolist()
            layer=registry[name]['layer']
            def factory(category,weights=weights,bias=bias,categories=categories,layer=layer):
                if category=='background':
                    return lambda values:-(l2(values[layer])@weights.T+bias).max(axis=1)
                j=categories.index(category)
                return lambda values:l2(values[layer])@weights[j]+bias[j]
            scorers[key]=factory
        models['clip']=backbone('clip_vit_b_32')
        if models['clip'].metadata!=registry['clip_vit_b_32_global']['metadata']:
            raise ValueError('CLIP visual-analysis backbone differs from cached feature provenance')
        with np.load(root/'clip/selected/text_embeddings.npz',allow_pickle=False) as z:
            text=z['embeddings'].mean(axis=1); categories=z['categories'].tolist()
        scorers['clip']=lambda category:lambda values:l2(values['global'])@text[categories.index(category)]
        save(root/'visual_analysis/models.json',dict(representations=names,svm_protocol='target_vs_background',
            clip_mode='validation-selected global template',selection_split='val',
            background_svm_score='negative maximum target margin; derived score, not a trained background classifier'))
        run(predictions,models,scorers,rows,project,config,root/'visual_analysis')
    stage_handlers.update(A=lambda:ab('A'),B=lambda:ab('B'),D=d,C=c,E=e,F=f)
    state_path=root/'run.json'
    state=read(state_path) if state_path.exists() else dict(stages={})
    state.update(state='running',experiment_id=experiment_id,started_utc=datetime.now(timezone.utc).isoformat(),split_sha256=split_sha,
        technical_smoke=bool(args.smoke),final_test_evaluation=args.final)
    state.pop('error',None)
    save(state_path,state)
    def verify(stage):
        marker=root/'manifests'/(stage+'.json')
        if not marker.exists():
            raise ValueError(f'Part 4 Stage {stage} must complete first with the same configuration/options')
        completed=read(marker)
        if not all((root/p).is_file() and file_hash(root/p)==sha for p,sha in completed['artifacts'].items()):
            raise ValueError(f'Stage {stage} artifacts changed; preserve this run and inspect before resuming')
    try:
        for stage in args.stages:
            for dependency in {'A':[], 'B':[], 'C':['A','B'], 'D':['C'], 'E':[], 'F':['A','B','E']}[stage]:
                verify(dependency)
            marker=root/'manifests'/(stage+'.json')
            if marker.exists():
                completed=read(marker)
                if all((root/p).is_file() and file_hash(root/p)==sha for p,sha in completed['artifacts'].items()):
                    state['stages'][stage]='complete'
                    save(state_path,state)
                    print(f'{stage}: reuse completed artifacts',flush=True); continue
                raise ValueError(f'Stage {stage} artifacts changed; preserve this run and inspect before resuming')
            print(f'Part 4 Stage {stage}',flush=True)
            stage_handlers[stage]()
            # Each stage owns its outputs; shared classification manifests are
            # scoped to its representations to permit later B/D execution.
            folders = {'A':[root/'classification'/n for n in registry if n.startswith('resnet18_')],
                'B':[root/'classification'/n for n in registry if n.startswith('dinov2_')], 'C':[root/'fewshot'],
                'D':[root/'regularization'], 'E':[root/'clip',root/'classification/clip_vit_b_32_global'], 'F':[root/'visual_analysis']}[stage]
            artifacts={str(p.relative_to(root)):file_hash(p) for folder in folders for p in folder.rglob('*') if p.is_file()}
            save(marker,dict(state='complete',artifacts=artifacts))
            state['stages'][stage]='complete'; save(state_path,state)
        report(root,project,split_sha,args.smoke)
        state['state']='complete' if set(state['stages'])==set('ABCDEF') else 'partial'
    except BaseException as exc:
        state.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        raise
    finally:
        for b in backbones.values():
            b.close()
        state['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(state_path,state)
        print(f'Outputs: {root}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output',type=Path,help='Override feature-cache and experiment-output base')
    parser.add_argument('--stages',nargs='+',choices=list('ABCDEF'),default=list('ABCDEF'))
    parser.add_argument('--smoke',type=int,metavar='IMAGES_PER_CLASS')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--final',action='store_true',help='Evaluate selected configurations on test; keep disabled during development')
    args=parser.parse_args()
    execute(args)

if __name__=='__main__':
    main()
