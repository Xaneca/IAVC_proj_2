"""Mandatory historical split check shared by C, D and E."""
import hashlib
import json
from pathlib import Path


def smoke_rows(rows, config):
    limit = config.get('smoke_per_class')
    if limit is None:
        return rows
    if not isinstance(limit, int) or limit < 2:
        raise ValueError('Smoke requires at least two images per class/split')
    if Path(config['output_dir']).name != 'smoke':
        raise ValueError('Smoke output must end in /smoke')
    ids = set()
    for split in ('train', 'val', 'test'):
        for category in ('motorbike', 'aeroplane', 'horse', 'car', 'person', 'background'):
            ids.update(r['image_id'] for r in [r for r in rows
                if r['split'] == split and category in r['labels']][:limit])
    return [r for r in rows if r['image_id'] in ids]


def split_digest(split_dir):
    reference = Path(__file__).with_name('split_reference')
    expected = {p.name: sorted(set(p.read_text().split())) for p in reference.glob('*.txt')}
    actual = {p.name: sorted(set(p.read_text().split())) for p in Path(split_dir).glob('*.txt')
              if p.stem.endswith(('_train', '_val', '_test'))}
    if not expected or expected != actual:
        raise ValueError('Historical split reference missing or splits changed')
    return hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
