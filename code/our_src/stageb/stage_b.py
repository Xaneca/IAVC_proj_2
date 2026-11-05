"""Stage B: select encoding, cache three-level histograms, tune C and rank."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from ..a_json import load,save,file_signature
from ..part1.data import manifest
from ..part1.results import load_centroids,write_csv
from ..part1.stage_a import binary_labels
from ..part1.svm import train_svm_grid
from .selection import select_encoding
from .histograms import cached_pyramid,svm_vector
from .statistics import class_statistics
from .ranking import ranking_rows,save_relevant_visual_words


def run(config,project,k=None,layout=None,normalization=None,smoke=None,render=True):
    project=Path(project)
    choice=select_encoding(project/config['stage_a_dir'],config['category'],k,layout,normalization)
    rows=manifest(project/config['split_dir'])
    centroid_config={**config,'vocabulary_sizes':[choice['k']]}
    models=load_centroids(project/config['stage_a_dir']/'centroids',centroid_config,rows,project)
    model={**models[choice['k']],'k':choice['k'],'image_dir':config['image_dir']}
    active=[r for r in rows if r['split'] in ('train','val')]
    output=project/config['output_dir']
    if smoke is not None:
        output=output/'smoke'
        ids=set()
        for split in ('train','val'):
            for category in (config['category'],'background'):
                ids.update(r['image_id'] for r in [r for r in active if r['split']==split and category in r['labels']][:smoke])
        active=[r for r in active if r['image_id'] in ids]
    state={'state':'running','encoding':choice,'config':config,'test_evaluated':False,
           'centroid_file':str(model['path']),'centroid_sha256':file_signature(model['path']),
           'smoke_per_class':smoke,'started_utc':datetime.now(timezone.utc).isoformat()}
    save(output/'run.json',state)
    pyramids,features={},{}
    for i,row in enumerate(active,1):
        pyramid,_=cached_pyramid(project,output,row['image_id'],model,config['assignment_batch_size'])
        pyramids[row['image_id']]=pyramid
        features[row['image_id']]=svm_vector(pyramid,choice)
        print(f'Histograms {i}/{len(active)}',flush=True)
    # These features already have the winning encoding's normalization.
    # Set the grid normalization to none to avoid transforming twice.
    result=train_svm_grid(features,binary_labels(active,'train',config['category']),
        binary_labels(active,'val',config['category']),
        dict(C=config['C_values'],normalization=['none'],max_iter=[config['max_iter']],tol=[config['tol']]),
        seed=config['seed'])
    best=result['best']
    write_csv(output/'svm_ap.csv',[{**r,'encoding_normalization':choice['normalization'],
        'selected':r['C']==best['C'],'k':choice['k'],'layout':choice['layout']} for r in result['trials']])
    stats=class_statistics(features,active)
    write_csv(output/'histograms/metrics.csv',stats)
    save(output/'histograms/metrics_definition.json',{
        'mean_variance':'per component of actual SVM inputs; population variance ddof=0',
        'chi_squared':'0.5 * sum((p-q)^2/(p+q)), zero/zero terms=0; L1 histogram rows',
        'reference':'class prototype computed from internal train only',
        'feature_encoding':choice,'multilabel':'each image contributes to each of its categories'})
    ranked=[]
    for split in config['ranking_splits']:
        if split not in ('train','val'):raise ValueError('Ranking only supports internal train/val')
        ranked.extend(ranking_rows(features,binary_labels(active,split,config['category']),split,best))
    write_csv(output/'ranking/ranking.csv',ranked)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'svm_model.npz').open('wb') as stream:
        np.savez_compressed(stream,weights=best['weights'],bias=best['bias'],
                            metadata=json.dumps({'encoding':choice,'C':best['C'],'category':config['category'],
                                                  'centroid_sha256':file_signature(model['path'])}))
    images=[]
    if render:
        images=save_relevant_visual_words(project,output,ranked,pyramids,choice,best,
                                          model['centroids'],config['image_dir'],config['extreme_images_per_end'])
    state.update(state='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                 best_C=best['C'],validation_ap=best['validation_ap'],visualizations=images,
                 image_count=len(active),histogram_layout='pyramid_L2',input_dimension=len(best['weights']))
    save(output/'run.json',state)
    print(f'Stage B complete: {output}',flush=True)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--k',type=int)
    parser.add_argument('--layout')
    parser.add_argument('--normalization',choices=['none','l1','l2'])
    parser.add_argument('--smoke-per-class',type=int)
    parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args()
    if args.smoke_per_class is not None and args.smoke_per_class<1:
        parser.error('--smoke-per-class must be positive')
    try:
        run(load(args.config),Path(__file__).resolve().parents[2],args.k,args.layout,args.normalization,
            args.smoke_per_class,not args.no_render)
    except ValueError as error:
        parser.exit(2,str(error)+'\n')


if __name__=='__main__':main()
