"""Small result files; encoding arrays and local descriptors remain in RAM."""
import csv
from pathlib import Path
import re
import subprocess
import numpy as np
from ..artifact_json import load, save


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def recover_centroids(project, output, revision):
    """Explicitly relocate previously computed results from local Git history."""
    if not re.fullmatch(r'[0-9a-fA-F]{7,40}', revision):
        raise ValueError('Recovery revision must be a Git commit hash')
    repo = Path(project).parent
    prefix = 'code/data/stage_a_json/models/kmeans'
    command = ['git', '-C', str(repo)]
    names = subprocess.check_output(command + ['ls-tree','-r','--name-only',revision,'--',prefix], text=True).splitlines()
    if not names:
        raise ValueError('No centroid results in requested revision')
    for name in names:
        if name.endswith('.json'):
            destination = Path(output)/'centroids'/Path(name).name
            if not destination.exists():
                raw = subprocess.check_output(command + ['show', f'{revision}:{name}'])
                import json
                data = json.loads(raw)
                data['recovered_from_commit'] = revision
                save(destination, data)


def load_centroids(folder, config, rows, project):
    train_ids = {r['image_id'] for r in rows if r['split']=='train'}
    historical_times = {}
    # Match the exact result filename, not merely K: avoids attributing another
    # sample's timing to this model. Original fit timing excluded JSON IO.
    pattern = re.compile(r"Fitting K=(\d+)\s+Training encoder of type 'bovw' completed in ([0-9.]+) seconds\.\s+Saved only centroids and configuration:[^\r\n]*?(K\d+_seed\d+_[a-f0-9]+\.json)")
    for log in sorted((Path(project)/'data/stage_a_json/logs').glob('*.stdout.log')):
        for _, seconds, name in pattern.findall(log.read_text(encoding='utf-8',errors='replace')):
            historical_times[name] = (float(seconds),str(log.relative_to(project)))
    models = {}
    for path in sorted(Path(folder).glob('*.json')):
        data = load(path)
        ident = data['identity']
        k = ident['k']
        if k not in config['vocabulary_sizes'] or ident['seed'] != config['seed']:
            continue
        if set(ident['training_ids']) != train_ids or not set(ident['sample_image_ids']) <= train_ids:
            raise ValueError(f'Centroid protocol does not match internal training split: {path}')
        if ident.get('pca_dim',0) != 0:
            raise ValueError('This Stage A expects raw SIFT centroids, without PCA')
        # Protect against mixing a modified extractor with historical vocabularies.
        from ..artifact_json import file_signature
        if ident.get('sources',{}).get('features.py') != file_signature(Path(project)/'src/features.py'):
            raise ValueError('SIFT implementation differs from centroid training')
        for item in ident.get('images',[]):
            stat = (Path(project)/config['image_dir']/f"{item['id']}.jpg").stat()
            if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
                raise ValueError(f"Training image changed since centroid fit: {item['id']}")
        centers = np.asarray(data['centroids'], dtype=np.float32)
        if centers.shape != (k,128) or not np.isfinite(centers).all():
            raise ValueError(f'Invalid centroid shape/values: {path}')
        if k in models:
            raise ValueError(f'More than one compatible vocabulary for K={k}; select a unique result')
        seconds = data.get('fit_seconds')
        origin = 'centroid_metadata' if seconds is not None else 'not_recorded'
        if seconds is None and path.name in historical_times:
            seconds,origin = historical_times[path.name]
        models[k] = {'centroids':centers, 'path':path, 'fit_seconds':seconds,'fit_timing_source':origin}
    missing = set(config['vocabulary_sizes']) - set(models)
    if missing:
        raise ValueError(f'Missing K={sorted(missing)} in {folder}; fit or recover those centroids first')
    return {k:models[k] for k in config['vocabulary_sizes']}
