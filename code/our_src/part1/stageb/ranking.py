"""Rank decision scores and save figures from RelevantVisualWordsDisplay."""
from pathlib import Path
from unittest.mock import patch
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.features import extractFeatures, standarizeImage
from src.encoders import BoVWEncoder
from src.visualization import RelevantVisualWordsDisplay
from .histograms import svm_vector


def word_contributions(input_vector, weights, k):
    x,w=np.asarray(input_vector),np.asarray(weights)
    if x.shape!=w.shape or x.ndim!=1 or len(x)%k:
        raise ValueError('Encoding and SVM weights must have aligned spatial word blocks')
    return (x*w).reshape(-1,k).sum(axis=0)


def display_weights(contributions, global_histogram):
    """Compensate the provided display's fixed Hellinger map exactly.

    Pass K weights, not the concatenated spatial vector (the display folds
    that vector in an incompatible order). Its plotted word scores then equal
    the word contributions of the actual weighted/normalized SVM input.
    """
    h=np.sqrt(np.asarray(global_histogram,dtype=np.float64))
    h/=np.sqrt(np.sum(h*h)+1e-10)
    return np.divide(contributions,h,out=np.zeros_like(h),where=h>0)


def ranking_rows(features, labels, split, model):
    ids=sorted(labels)
    scores=np.stack([features[i] for i in ids])@model['weights']+model['bias']
    order=sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))
    return [{'split':split,'rank':rank,'image_id':ids[i],
             'label':labels[ids[i]],'decision_score':float(scores[i])}
            for rank,i in enumerate(order,1)]


def save_relevant_visual_words(project,output,rows,pyramids,choice,model,centroids,image_dir,ends=3):
    output=Path(output)/'ranking/relevantVisualWords'
    output.mkdir(parents=True,exist_ok=True)
    k=choice['k']
    encoder=BoVWEncoder(dict(type='bovw',num_words=k,pca_dim=0,words=centroids))
    saved=[]
    for split in sorted({r['split'] for r in rows}):
        ranked=[r for r in rows if r['split']==split]
        for role,selected in (('highest',ranked[:ends]),('lowest',ranked[-ends:])):
            for row in selected:
                image_id=row['image_id']
                image_path=Path(project)/image_dir/f'{image_id}.jpg'
                # Extract from the original path, exactly as the feature cache:
                # passing an already resized image could normalize it twice.
                points,descriptors,_=extractFeatures(image_path,cache_dir=None)
                image=np.clip(standarizeImage(image_path),0,1)
                hist,words,distances=encoder.encode(descriptors,with_dist=True)
                if not len(words):continue
                vector=svm_vector(pyramids[image_id],choice)
                contribution=word_contributions(vector,model['weights'],k)
                weights=display_weights(contribution,hist)
                class Adapter:
                    def get_visual_words(self,_image):
                        return hist,words,distances,points
                display=RelevantVisualWordsDisplay(image,weights,Adapter())
                present=set(words.tolist())
                candidate_ranks=[i for i,word in enumerate(display.perm) if int(word) in present]
                rank=candidate_ranks[0] if role=='highest' else candidate_ranks[-1]
                with patch('matplotlib.pyplot.show',lambda:None):
                    display.plot(rank,figsize=(14,8))
                figure=plt.gcf()
                for axis in figure.axes:
                    axis.set_title(textwrap.fill(axis.get_title(),55),fontsize=10)
                figure.suptitle(f"{split} {image_id}: decision score {row['decision_score']:.4f} ({role})",fontsize=13)
                figure.tight_layout(rect=(.02,.02,.98,.94))
                path=output/f'{split}_{role}_{image_id}.png'
                figure.savefig(path,dpi=140)
                plt.close(figure)
                saved.append(str(path))
    return saved
