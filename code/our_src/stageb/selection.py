"""Select an encoding only from completed Stage A validation results."""
import csv
from pathlib import Path
from ..a_json import load
from ..stagea.encoding import LAYOUTS


def select_encoding(stage_a, category, k=None, layout=None, normalization=None):
    if (k is None) != (layout is None):
        raise ValueError('Explicit selection requires both --k and --layout')
    if k is not None:
        if k not in (32,64,128,256,512,1024) or layout not in LAYOUTS:
            raise ValueError('Unsupported explicit encoding')
        if normalization not in (None,'none','l1','l2'):
            raise ValueError('Unsupported normalization')
        return dict(k=k,layout=layout,normalization=normalization or 'l2',
                    source='explicit_override_not_Stage_A_winner',validation_ap=None)
    root=Path(stage_a)
    if not (root/'run.json').exists() or load(root/'run.json').get('state')!='complete':
        raise ValueError('Stage A has no completed full run. Finish Stage A or provide explicit --k and --layout; smoke results are not used.')
    with (root/'svm_ap.csv').open(encoding='utf-8',newline='') as stream:
        candidates=[r for r in csv.DictReader(stream) if r['category']==category]
    if not candidates:
        raise ValueError(f'No Stage A validation results for {category}')
    best=max(candidates,key=lambda r:float(r['validation_ap']))
    result=dict(k=int(best['k']),layout=best['layout'],normalization=best['normalization'],
                source='Stage_A_validation_AP',validation_ap=float(best['validation_ap']))
    if result['layout'] not in LAYOUTS or result['normalization'] not in ('none','l1','l2'):
        raise ValueError('Unsupported Stage A encoding configuration')
    return result
