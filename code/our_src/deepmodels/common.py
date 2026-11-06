"""Adapters to existing split, SVM, persistence and AP implementations."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, project):
    """Resolve shared modules in the remote repository and legacy checkouts."""
    candidates = {
        'data': ['our_src.part1.stagea.data', 'our_src.part1.data'],
        'svm': ['our_src.part1.stagea.svm', 'our_src.part1.svm'],
        'evaluation': ['our_src.part1.stagec.evaluation', 'our_src.stagec.evaluation'],
        'protocol': ['our_src.part1.stagec.protocol', 'our_src.stagec.protocol'],
    }
    sys.path.insert(0, str(project))
    for candidate in candidates[name]:
        try:
            return importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            if exc.name != candidate and not candidate.startswith(exc.name + '.'):
                raise
    raise ImportError(f'Existing {name} utility unavailable; install the previous stages')


def provided(project):
    """Load the professor module without changing it or masking dependencies."""
    path = Path(project)/'src/deep_models.py'
    if not path.is_file():
        raise FileNotFoundError(f'Professor-provided module missing: {path}')
    spec = importlib.util.spec_from_file_location('assignment_provided_deep_models', path)
    result = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(result)
    except ImportError as exc:
        raise RuntimeError(f'Cannot import provided deep_models.py: {exc}. Install its requirements (including OpenAI CLIP and torchinfo).') from exc
    return result


def dataset(project, config, smoke=None):
    split_dir = Path(project)/config['split_dir']
    sha = module('protocol', project).split_digest(split_dir)
    rows = module('data', project).manifest(split_dir)
    categories = sorted({c for r in rows for c in r['labels']})
    if set(categories) != set(config['categories']):
        raise ValueError(f'Configured classes differ from dataset: {categories}')
    if any('background' in r['labels'] and len(r['labels']) > 1 for r in rows):
        raise ValueError('Background overlaps target labels; review protocol')
    if smoke is not None:
        if smoke < 2:
            raise ValueError('Smoke requires >=2 images/category/split')
        chosen = set()
        for split in ('train', 'val', 'test'):
            for c in categories:
                matches = [r for r in rows if r['split'] == split and c in r['labels']]
                chosen.update(r['image_id'] for r in matches[:smoke])
        rows = [r for r in rows if r['image_id'] in chosen]
    for row in rows:
        if not image_path(project, config, row['image_id']).is_file():
            raise FileNotFoundError(image_path(project, config, row['image_id']))
    return rows, sha


def image_path(project, config, image_id):
    return Path(project)/config['image_dir']/(image_id if Path(image_id).suffix else image_id + '.jpg')


def l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
