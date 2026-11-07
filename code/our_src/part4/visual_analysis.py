"""Prediction-driven panels and model-agnostic spatial occlusion sensitivity."""
from pathlib import Path
import numpy as np
from PIL import Image
from .common import image_path, l2, npz, save


def select(predictions, maximum, max_total=15):
    sets=[set(p) for p in predictions.values()]
    if any(s!=sets[0] for s in sets):
        raise ValueError('Prediction image IDs differ across models')
    ids = sorted(sets[0])
    groups = {key:[] for key in ('all_correct','all_wrong','cnn_only','dino_only','clip_success','clip_failure','background','confident_mistake','multilabel')}
    candidates={g:[] for g in groups}
    for i in ids:
        p = {name:predictions[name][i] for name in predictions}
        if any(set(v['truth'])!=set(p['cnn']['truth']) for v in p.values()):
            raise ValueError('Ground truth differs for the same image ID')
        correct = {name:v['prediction'] in v['truth'] for name,v in p.items()}
        flags = dict(all_correct=all(correct.values()),all_wrong=not any(correct.values()),
            cnn_only=correct['cnn'] and not correct['dino'],dino_only=correct['dino'] and not correct['cnn'],
            clip_success=correct['clip'] and not (correct['cnn'] and correct['dino']),
            clip_failure=not correct['clip'] and (correct['cnn'] or correct['dino']),
            background='background' in p['cnn']['truth'] and not all(correct.values()),
            confident_mistake=not all(correct.values()),multilabel=len(p['cnn']['truth'])>1)
        for group,flag in flags.items():
            if flag:
                candidates[group].append(i)
    # Confidence ranks are calculated within each model; margins and cosines
    # are never compared as if their numeric scales were interchangeable.
    ranks={}
    for name in predictions:
        ordered=sorted(ids,key=lambda i:(-float(max(predictions[name][i].get('scores',[0]))),i))
        ranks[name]={i:j for j,i in enumerate(ordered)}
    used=set(); category_counts={}
    for group in groups:
        while candidates[group] and len(groups[group])<maximum and len(used)<max_total:
            image_id=min(candidates[group],key=lambda i:(
                sum(category_counts.get(c,0) for c in predictions['cnn'][i]['truth']),
                min(ranks[n][i] for n in ranks if group!='confident_mistake' or predictions[n][i]['prediction'] not in predictions[n][i]['truth']),i))
            candidates[group].remove(image_id)
            if image_id in used:
                continue
            groups[group].append(image_id); used.add(image_id)
            for c in predictions['cnn'][image_id]['truth']:
                category_counts[c]=category_counts.get(c,0)+1
    return groups


def load_predictions(path):
    with np.load(path,allow_pickle=False) as z:
        categories = z['categories'].tolist()
        return {str(i):dict(prediction=('background' if 'background' not in categories and max(s)<0 else categories[int(np.argmax(s))]),
            truth=([c for c,y in zip(categories,l) if y] or ['background']),scores=s.tolist(),categories=categories,
            predicted_label_set=([categories[int(np.argmax(s))]] if 'background' in categories else [c for c,v in zip(categories,s) if v>=0] or ['background']))
            for i,s,l in zip(z['image_ids'],z['scores'],z['labels'])}


def encode_as_extracted(backbone, images, batch_size):
    """Encode at the extraction batch size; cuDNN convolutions vary slightly with it."""
    padded = images + [images[-1]]*(batch_size-len(images))
    values,_ = backbone.encode(padded)
    return {k:v[:len(images)] for k,v in values.items()}


def occlusion(backbone, image, score, grid, expected=None, batch_size=1):
    """Score decrease after mean-RGB occlusion. Sensitivity, not causal proof."""
    baseline = score(encode_as_extracted(backbone,[image],batch_size))
    if expected is not None and not np.allclose(baseline,expected,atol=5e-4,rtol=5e-4):
        raise ValueError('Occlusion baseline differs from the saved classifier score')
    width,height = image.size
    heat = np.zeros((grid,grid),dtype=np.float32)
    mean = tuple(np.asarray(image).mean(axis=(0,1)).astype(int))
    for y in range(grid):
        images = []
        for x in range(grid):
            changed = image.copy()
            changed.paste(mean,(x*width//grid,y*height//grid,(x+1)*width//grid,(y+1)*height//grid))
            images.append(changed)
        heat[y] = baseline - score(encode_as_extracted(backbone,images,batch_size))
    return heat


def run(predictions, backbones, scorers, rows, project, config, output):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    groups = select(predictions,config['visual_examples'],config.get('visual_max_total',15))
    save(output/'selection.json',dict(groups=groups,correctness='top-scoring target belongs to true label set; SVM predicts background if all margins <0; CLIP includes background prompt',
        explanation='mean-RGB occlusion score decrease; sensitivity is not causal evidence; input includes original crop preprocessing'))
    selected_records=[]
    for group, ids in groups.items():
        for image_id in ids:
            with Image.open(image_path(project,config,image_id)) as im:
                image = im.convert('RGB')
            truth=predictions['cnn'][image_id]['truth']
            target=sorted(c for c in truth if c!='background')[0] if truth!=['background'] else 'background'
            selected_records.append(dict(image_id=image_id,selection_reason=group,ground_truth=truth,
                ground_truth_target=target,target_policy='lexically first annotated target; background only when no targets',
                models={n:dict(v[image_id],top1_membership=v[image_id]['prediction'] in truth,
                    exact_multilabel_correct=set(v[image_id].get('predicted_label_set',[v[image_id]['prediction']]))==set(truth)) for n,v in predictions.items()}))
            for mode in ('ground_truth','predicted'):
                render_case(image,image_id,group,mode,target,predictions,backbones,scorers,config,output)
            if config.get('dino_patch_visualization') and hasattr(backbones['dino'],'patch_similarity'):
                similarity=backbones['dino'].patch_similarity(image)
                npz(output/f'{group}_{image_id}_patch_cls.npz',cosine_similarity=similarity)
                # Show the actual crop, avoiding an incorrect overlay on uncropped pixels.
                b=backbones['dino']; tensor=b.preprocess(image).cpu().numpy().transpose(1,2,0)
                crop=np.clip(tensor*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
                fig,ax=plt.subplots(); ax.imshow(crop)
                overlay=ax.imshow(similarity,extent=(0,crop.shape[1],crop.shape[0],0),alpha=.5,cmap='coolwarm',vmin=-1,vmax=1)
                fig.colorbar(overlay,ax=ax,label='Patch-CLS cosine'); ax.set_title('Representation similarity; not SVM attribution')
                fig.savefig(output/f'{group}_{image_id}_patch_cls.png',dpi=140); plt.close(fig)
    save(output/'cases.json',selected_records)


def render_case(image,image_id,group,mode,target,predictions,backbones,scorers,config,output):
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,4,figsize=(16,4))
    axes[0].imshow(image); axes[0].set_title('Truth: '+', '.join(predictions['cnn'][image_id]['truth']))
    for axis,name in zip(axes[1:],('cnn','dino','clip')):
        prediction = target if mode=='ground_truth' else predictions[name][image_id]['prediction']
        saved=predictions[name][image_id]
        expected=(saved['scores'][saved['categories'].index(prediction)] if prediction in saved['categories'] else -max(saved['scores']))
        heat = occlusion(backbones[name],image,scorers[name](prediction),config['occlusion_grid'],expected,config['batch_size'])
        npz(output/f'{group}_{image_id}_{mode}_{name}.npz',occlusion_score_decrease=heat,
            image_id=np.array(image_id),category=np.array(prediction),mode=np.array(mode),baseline_score=np.array(expected),
            score_type=np.array('cosine similarity' if name=='clip' else 'SVM margin / negative max margin for background'))
        axis.imshow(image)
        limit=max(float(np.abs(heat).max()),1e-8)
        overlay=axis.imshow(heat,cmap='coolwarm',alpha=.5,extent=(0,image.width,image.height,0),vmin=-limit,vmax=limit)
        fig.colorbar(overlay,ax=axis,fraction=.035,label='Cosine decrease' if name=='clip' else 'Margin decrease')
        axis.set_title(f'{name}: {prediction}\n{mode}; signed score decrease')
    for axis in axes:
        axis.axis('off')
    fig.tight_layout(); fig.savefig(output/f'{group}_{image_id}_{mode}.png',dpi=140); plt.close(fig)
