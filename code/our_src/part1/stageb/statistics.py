"""Per-component mean/variance and class chi-squared distances."""
import numpy as np


def l1_rows(x):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('Chi-squared requires finite nonnegative histograms')
    total=x.sum(axis=1,keepdims=True)
    return np.divide(x,total,out=np.zeros_like(x),where=total>0)


def chi_squared(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if np.any(a<0) or np.any(b<0) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Chi-squared requires nonnegative finite inputs')
    den=a+b
    return .5*np.sum(np.divide((a-b)**2,den,out=np.zeros_like(den),where=den>0),axis=-1)


def class_statistics(features, rows):
    """Stats describe actual SVM input vectors; variance ddof=0.

    Chi-squared is calculated after L1 normalization. Validation images are
    compared to train-only class prototypes, never validation prototypes.
    Multi-label images contribute once to each relevant class.
    """
    categories=sorted({c for row in rows for c in row['labels']})
    ids=[r['image_id'] for r in rows]
    x=np.stack([features[i] for i in ids])
    prob=l1_rows(x)
    prototypes={}
    for category in categories:
        indices=[i for i,r in enumerate(rows) if r['split']=='train' and category in r['labels']]
        if indices:prototypes[category]=prob[indices].mean(axis=0)
    result=[]
    for split in ('train','val'):
        for category in categories:
            indices=[i for i,r in enumerate(rows) if r['split']==split and category in r['labels']]
            if not indices:continue
            selected=x[indices]
            mean,variance=selected.mean(axis=0,dtype=np.float64),selected.var(axis=0,dtype=np.float64,ddof=0)
            for component,(m,v) in enumerate(zip(mean,variance)):
                result.append({'record_type':'feature','split':split,'class':category,
                               'n_images':len(indices),'component':component,'mean':float(m),'variance':float(v)})
            distances=chi_squared(prob[indices],prototypes[category]) if category in prototypes else None
            result.append({'record_type':'class_summary','split':split,'class':category,'n_images':len(indices),
                           'chi2_to_train_class_mean':float(distances.mean()) if distances is not None else None,
                           'chi2_to_train_class_variance':float(distances.var()) if distances is not None else None,
                           'zero_histograms':int(np.count_nonzero(prob[indices].sum(axis=1)==0))})
    for i,a in enumerate(sorted(prototypes)):
        for b in sorted(prototypes)[i+1:]:
            result.append({'record_type':'class_pair','split':'train','class':a,'other_class':b,
                           'chi2_between_train_class_means':float(chi_squared(prototypes[a],prototypes[b]))})
    return result
