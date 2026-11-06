"""Fixed prior prompts; average cosine similarity, never test-tuned."""
from pathlib import Path
import time
import numpy as np
from .common import l2, npz, save, module, read
from .classification import metrics, partitions


def retrieval(labels,scores,destination,category,split,project):
    """Reuse historical rankings/AP, correcting its binary-background plot title."""
    report=module('evaluation',project).evaluate(labels,scores,destination,category,split)
    report.update(protocol='full_dataset_classwise_text_retrieval',
        negative_definition='all images without this category',
        ap_definition='sklearn non-interpolated AP on cosine similarities')
    save(destination/'ap.json',report)
    import csv
    import matplotlib.pyplot as plt
    with (destination/'precision_recall.csv').open() as stream:
        curve=list(csv.DictReader(stream))
    fig,ax=plt.subplots()
    ax.step([float(r['recall']) for r in curve],[float(r['precision']) for r in curve],where='post',label=f"AP={report['ap']:.4f}")
    ax.set(xlabel='Recall',ylabel='Precision',title=f'{category} / all non-target images: {split}',xlim=(0,1),ylim=(0,1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(destination/'precision_recall.png',dpi=150); plt.close(fig)


def run(backbone, representation, rows, config, project, output, final=False):
    import clip
    torch = backbone.torch
    output = Path(output)
    categories = config['categories']
    reports = {}
    conditions = [('single',config['prompt_templates'][:1]),('multiple',config['prompt_templates'])] + [(f'template_{i}',[t]) for i,t in enumerate(config['prompt_templates'])]
    for mode, templates in conditions:
        destination = output/mode
        destination.mkdir(parents=True,exist_ok=True)
        prompts = [[template.format(config['class_descriptions'][c]) for template in templates] for c in categories]
        start = time.perf_counter()
        tokens = clip.tokenize([p for group in prompts for p in group]).to(backbone.device)
        with torch.inference_mode():
            text = backbone.model.encode_text(tokens).float()
            text = text / text.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        backbone.sync()
        text = text.cpu().numpy().reshape(len(categories),len(templates),-1)
        text_seconds = time.perf_counter()-start
        npz(destination/'text_embeddings.npz', embeddings=text,categories=np.array(categories),prompts=np.array(prompts))
        save(destination/'prompts.json',dict(prompts=prompts,aggregation='mean of cosine similarities; no renormalization of mean text',selection='fixed prior knowledge, before evaluation'))
        reports[mode] = {}
        for split in ('train','val'):
            active = [r for r in rows if r['split']==split]
            x = np.stack([representation['features'][r['image_id']] for r in active])
            start = time.perf_counter()
            scores = x @ text.mean(axis=1).T
            seconds = time.perf_counter()-start
            report,predictions = metrics(active,categories,scores,forced_single=True)
            report.update(text_encoding_time=text_seconds,similarity_time=seconds)
            npz(destination/(split+'_predictions.npz'),image_ids=np.array([r['image_id'] for r in active]),
                scores=scores,predictions=predictions,labels=np.array([[c in r['labels'] for c in categories] for r in active]),categories=np.array(categories))
            for j,c in enumerate(categories):
                labels = {r['image_id']:1 if c in r['labels'] else -1 for r in active}
                retrieval(labels,dict(zip(labels,map(float,scores[:,j]))),
                    destination/'retrieval'/split/c,c,split,project)
            reports[mode][split] = report
        save(destination/'metrics.json',reports[mode])
    candidates=[dict(index=i,template=t,validation_map=reports[f'template_{i}']['val']['map']) for i,t in enumerate(config['prompt_templates'])]
    selected=max(candidates,key=lambda r:r['validation_map'])
    save(output/'template_selection.json',dict(selected=selected,trials=candidates,
        criterion='global validation mAP over five targets; first template wins ties',test_used_for_selection=False))
    import shutil
    source=output/f"template_{selected['index']}"
    destination=output/'selected'
    shutil.copytree(source,destination,dirs_exist_ok=True)
    selected_prompts=read(destination/'prompts.json')
    selected_prompts.update(selection='global template selected exclusively by validation target mAP',
        selection_record='../template_selection.json')
    save(destination/'prompts.json',selected_prompts)
    import copy
    reports['selected']=copy.deepcopy(reports[f"template_{selected['index']}"])
    if final:
        # All choices are frozen above before any test labels/scores are evaluated.
        for mode in ('single','multiple','selected'):
            destination=output/mode
            with np.load(destination/'text_embeddings.npz',allow_pickle=False) as z:
                text=z['embeddings'].copy()
            active=[r for r in rows if r['split']=='test']
            x=np.stack([representation['features'][r['image_id']] for r in active])
            start=time.perf_counter(); scores=x@text.mean(axis=1).T; seconds=time.perf_counter()-start
            report,predictions=metrics(active,categories,scores,forced_single=True)
            report.update(similarity_time=seconds,text_encoding_time=0.,text_encoding_reused=True)
            reports[mode]['test']=report
            npz(destination/'test_predictions.npz',image_ids=np.array([r['image_id'] for r in active]),
                scores=scores,predictions=predictions,labels=np.array([[c in r['labels'] for c in categories] for r in active]),categories=np.array(categories))
            for j,c in enumerate(categories):
                labels={r['image_id']:1 if c in r['labels'] else -1 for r in active}
                retrieval(labels,dict(zip(labels,map(float,scores[:,j]))),destination/'retrieval/test'/c,c,'test',project)
            save(destination/'metrics.json',reports[mode])
    save(output/'comparison.json',reports)
    return reports
