"""Frozen Stage B test evaluation: no fitting or hyperparameter selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from ..artifact_json import load,save,key,file_signature
from ..stagea.data import manifest
from ..stagea.stage_a import binary_labels
from ..stagec.protocol import split_digest
from ..stagec.evaluation import evaluate
from ..stagef.common import score_metrics,save_csv
from .histograms import cached_pyramid,svm_vector


def select_cases(records, count=2):
    """Deterministic illustrative selection after scoring; no model tuning."""
    rules={
        'TP':lambda r:r['label']==1 and r['score']>=0,
        'FN':lambda r:r['label']==1 and r['score']<0,
        'FP':lambda r:r['label']==-1 and r['score']>=0,
        'Hard Negative':lambda r:r['label']==-1 and r['score']<0}
    selected={}
    for group,eligible in rules.items():
        candidates=[r for r in records if eligible(r)]
        candidates.sort(key=lambda r:((r['score'] if group=='FN' else -r['score']),r['image_id']))
        selected[group]=candidates[:count]
    return selected


def selected_centroid(project,config,state,metadata):
    """Resolve the exact frozen vocabulary by hash, never by a new score."""
    expected=metadata['centroid_sha256']
    if state['centroid_sha256']!=expected: raise ValueError('Stage B state/model vocabulary mismatch')
    matches=[p for p in (project/config['stage_a_dir']/'centroids').glob('*.json') if file_signature(p)==expected]
    if not matches: raise ValueError('Exact trained vocabulary is unavailable; do not substitute/retrain it')
    path=matches[0]; data=load(path); source=(project/'src/features.py').read_bytes()
    equivalent={hashlib.sha256(source).hexdigest(),hashlib.sha256(source.replace(b'\r\n',b'\n')).hexdigest()}
    if data['identity']['sources']['features.py'] not in equivalent:
        raise ValueError('SIFT source differs from vocabulary training beyond line endings')
    rows=manifest(project/config['split_dir'])
    train={r['image_id'] for r in rows if r['split']=='train'}
    if set(data['identity']['training_ids'])!=train or not set(data['identity']['sample_image_ids'])<=train:
        raise ValueError('Vocabulary training membership differs from historical training')
    centers=np.asarray(data['centroids'],dtype=np.float32); k=metadata['encoding']['k']
    if centers.shape!=(k,128) or not np.isfinite(centers).all(): raise ValueError('Invalid frozen vocabulary')
    return dict(path=path,centroids=centers,k=k,image_dir=config['image_dir'])


def figures(project,config,output,report,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm=np.array(report['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(5,4)); image=ax.imshow(cm,cmap='Blues')
    for (y,x),v in np.ndenumerate(cm): ax.text(x,y,str(v),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['background','motorbike'],yticklabels=['background','motorbike'],
        xlabel='Prediction (margin >= 0)',ylabel='Ground truth',title='Stage B — frozen test model')
    fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(output/'confusion_matrix.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(11,15))
    for axes_row,(group,records) in zip(axes,cases.items()):
        for i,axis in enumerate(axes_row):
            axis.axis('off')
            if i>=len(records): axis.set_title(f'{group}: no further eligible example'); continue
            r=records[i]
            with Image.open(project/config['image_dir']/(r['image_id']+'.jpg')) as im: axis.imshow(im.convert('RGB'))
            axis.set_title(f"{group}: {r['image_id']} | rank {r['rank']}\nGT: {', '.join(r['labels'])} | predicted: {r['prediction']}\nSVM margin: {r['score']:.6f}",fontsize=10)
    fig.suptitle('Stage B — deterministic test cases, decision threshold 0',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(output/'test_cases.png',dpi=140); plt.close(fig)


def latex_table(report,cases):
    lines=[r'\begin{tabular}{lr}',r'\hline',r'Metric & Test \\',r'\hline']
    for field in ('ap','accuracy','precision','recall','f1','images'):
        value=report[field]; lines.append(f'{field} & {value:.6f} '+r'\\')
    lines.extend([r'\hline',r'\end{tabular}'])
    case_lines=[r'\begin{tabular}{llrr}',r'\hline',r'Group & Image & Score & Rank \\',r'\hline']
    for group,rows in cases.items():
        for r in rows: case_lines.append(f"{group} & {r['image_id']} & {r['score']:.6f} & {r['rank']} "+r'\\')
    case_lines.extend([r'\hline',r'\end{tabular}'])
    return '\n'.join(lines)+'\n','\n'.join(case_lines)+'\n'


def notebook(stage_b,output,report,cases):
    """Persist a report notebook with executed display outputs; no inference cells."""
    import base64
    relative=output.relative_to(stage_b).as_posix()
    cells=[]
    def markdown(text): cells.append(dict(cell_type='markdown',id=f'cell-{len(cells)}',metadata={},source=text.splitlines(True)))
    def code(text,outputs): cells.append(dict(cell_type='code',id=f'cell-{len(cells)}',metadata={},source=text.splitlines(True),execution_count=len(cells),outputs=outputs))
    markdown('# Stage B — frozen model test evaluation\n\nTraining and validation artifacts are preserved. K/layout/normalization/C were frozen before test scoring. '
        'Cases use test labels only for qualitative error analysis, never for selection.\n\n'
        f"Test AP: **{report['ap']:.6f}**; precision **{report['precision']:.6f}**, recall **{report['recall']:.6f}**, F1 **{report['f1']:.6f}**.\n\n"
        f'Run: `{relative}`. All figures and LaTeX tables are generated by the evaluation command.\n')
    code("from pathlib import Path\nimport json\nfrom IPython.display import display, Image\ncandidates = [Path.cwd(), Path.cwd()/'code/results/part1/stageB', Path.cwd()/'results/part1/stageB']\nstage_b = next(p for p in candidates if (p/'svm_model.npz').exists())\nrun = stage_b / "+repr(relative)+"\nmetrics = json.loads((run/'metrics.json').read_text())\nprint(json.dumps(metrics, indent=2))\n",
        [dict(output_type='stream',name='stdout',text=json.dumps(report,indent=2).splitlines(True))])
    for name in ('precision_recall.png','confusion_matrix.png','test_cases.png'):
        code(f"display(Image(filename=str(run / {name!r})))\n",[dict(output_type='display_data',metadata={},data={'image/png':base64.b64encode((output/name).read_bytes()).decode()})])
    code("print((run/'metrics.tex').read_text())\nprint((run/'cases.tex').read_text())\n",[dict(output_type='stream',name='stdout',text=((output/'metrics.tex').read_text()+(output/'cases.tex').read_text()).splitlines(True))])
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python')),cells=cells)
    target=stage_b/'report.ipynb'
    if target.exists():
        archive=stage_b/'report_history'/(file_signature(target)+'.ipynb'); archive.parent.mkdir(exist_ok=True)
        if not archive.exists(): archive.write_bytes(target.read_bytes())
    target.write_text(json.dumps(nb,indent=1),encoding='utf-8')


def evaluate_test(config,project,stage_b=None):
    project=Path(project).resolve(); stage_b=Path(stage_b or project/config['output_dir']).resolve()
    state=load(stage_b/'run.json')
    if state.get('state')!='complete' or state.get('smoke_per_class') is not None: raise ValueError('Requires completed full Stage B training/validation')
    model_path=stage_b/'svm_model.npz'
    with np.load(model_path,allow_pickle=False) as z:
        weights=z['weights'].copy(); bias=float(z['bias']); metadata=json.loads(str(z['metadata']))
    if metadata['encoding']!=state['encoding'] or float(metadata['C'])!=float(state['best_C']): raise ValueError('Frozen model/state mismatch')
    # No test labels enter any choice above; there is no fit/grid search in this function.
    centroid=selected_centroid(project,config,state,metadata)
    digest=split_digest(project/config['split_dir']); rows=manifest(project/config['split_dir'])
    labels=binary_labels(rows,'test',metadata['category']); by_id={r['image_id']:r for r in rows}
    sources={str(p.relative_to(project)):file_signature(p) for p in [Path(__file__),Path(__file__).with_name('histograms.py'),project/'src/features.py',project/'src/encoders.py',project/'our_src/part1/stagea/encoding.py']}
    provenance=dict(frozen_model_sha256=file_signature(model_path),training_state_sha256=file_signature(stage_b/'run.json'),
        centroid_sha256=file_signature(centroid['path']),split_sha256=digest,encoding=metadata['encoding'],C=metadata['C'],
        image_sha256={i:file_signature(project/config['image_dir']/(i+'.jpg')) for i in sorted(labels)},sources=sources,
        selection_split='val',evaluation_split='test',test_used_for_selection=False,threshold=0.,
        versions={p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','scikit-image','cyvlfeat')})
    identity=key(provenance); output=stage_b/'test_evaluations'/identity
    if (output/'complete.json').exists():
        marker=load(output/'complete.json')
        if not all((output/p).is_file() and file_signature(output/p)==sha for p,sha in marker['artifacts'].items()): raise ValueError('Test artifacts changed')
        print(f'Reuse complete frozen Stage B test evaluation: {output}',flush=True); return load(output/'metrics.json')
    if output.exists() and any(output.iterdir()) and not (output/'manifest.json').exists(): raise ValueError('Unknown existing evaluation directory')
    save(output/'manifest.json',provenance)
    scores={}; seconds=0.
    for n,image_id in enumerate(sorted(labels),1):
        start=time.perf_counter()
        pyramid,_=cached_pyramid(project,output,image_id,centroid,config['assignment_batch_size'])
        vector=svm_vector(pyramid,metadata['encoding'])
        if vector.shape!=weights.shape: raise ValueError('Frozen feature/model dimensions differ')
        scores[image_id]=float(vector@weights+bias); seconds+=time.perf_counter()-start
        if n%25==0 or n==len(labels): print(f'Stage B frozen test {n}/{len(labels)}',flush=True)
    evaluate(labels,scores,output,metadata['category'],'test')
    report=score_metrics([labels[i] for i in scores],list(scores.values()))
    report.update(confusion_matrix=[[report['tn'],report['fp']],[report['fn'],report['tp']]],threshold=0.,
        k=metadata['encoding']['k'],layout=metadata['encoding']['layout'],normalization=metadata['encoding']['normalization'],C=metadata['C'],
        frozen_model_sha256=provenance['frozen_model_sha256'],test_used_for_selection=False,
        extraction_encoding_and_prediction_seconds=seconds,run_id=identity)
    ranking=sorted(scores,key=lambda i:(-scores[i],i))
    records=[dict(image_id=i,label=labels[i],labels=by_id[i]['labels'],score=scores[i],rank=n,
        prediction=metadata['category'] if scores[i]>=0 else 'background') for n,i in enumerate(ranking,1)]
    cases=select_cases(records); save(output/'cases.json',cases); save(output/'metrics.json',report)
    figures(project,config,output,report,cases)
    metrics_tex,cases_tex=latex_table(report,cases)
    (output/'metrics.tex').write_text(metrics_tex); (output/'cases.tex').write_text(cases_tex)
    notebook(stage_b,output,report,cases)
    artifacts={str(p.relative_to(output)):file_signature(p) for p in output.rglob('*') if p.is_file() and p.name!='complete.json'}
    save(output/'complete.json',dict(run_id=identity,artifacts=artifacts))
    assert file_signature(model_path)==provenance['frozen_model_sha256']
    assert file_signature(stage_b/'run.json')==provenance['training_state_sha256']
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--stage-b-dir',type=Path)
    args=parser.parse_args(); evaluate_test(load(args.config),args.project,args.stage_b_dir)


if __name__=='__main__': main()
