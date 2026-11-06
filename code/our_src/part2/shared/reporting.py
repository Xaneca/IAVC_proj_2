"""Consolidated H/I results; all Fisher mathematics remains in src.encoders."""
import csv
import json
from pathlib import Path
import numpy as np
from our_src.part1.artifact_json import save,load,file_signature
from our_src.part1.stagef.common import save_csv


def consolidate(output, records, images, manifest):
    output = Path(output)
    table = []
    for name in sorted({r['experiment'] for r in records}):
        group = [r for r in records if r['experiment'] == name]
        first = group[0]
        d = first['dimension']
        row = dict(experiment=name, encoder=first['encoder'], k=first['k'], dimension=d,
            descriptor_bytes_per_image=4*d, descriptor_bytes_all_images=4*d*images,
            feature_storage='uncompressed float32; temporary matrices removed after evaluation',
            images=images, seed=first['seed'], protocol='target_vs_background',
            validation_map=float(np.mean([r['val_ap'] for r in group])),
            test_map=float(np.mean([r['test_ap'] for r in group])),
            sift_extraction_seconds_shared=first['sift_seconds_shared'],
            sift_cache_read_seconds_shared=first.get('sift_cache_read_seconds_shared',0.),
            sift_access_seconds_shared=first.get('sift_access_seconds_shared',first['sift_seconds_shared']),
            sift_cache_hits=first.get('sift_cache_hits',0),sift_cache_misses=first.get('sift_cache_misses',images),
            encoder_preparation_sift_seconds_shared=first.get('encoder_preparation_sift_seconds_shared',0.),
            encoder_fit_seconds=first['encoder_fit_seconds'],
            encoding_seconds_per_image=first['encoding_seconds_shared']/images,
            selected_svm_fit_seconds=sum(r['svm_fit_seconds'] for r in group),
            svm_search_fit_seconds=sum(r['svm_search_fit_seconds'] for r in group),
            classification_inference_seconds=sum(r[s+'_inference_seconds'] for r in group for s in ('train','val','test')))
        for r in group:
            c=r['category']
            row.update({f'validation_ap_{c}':r['val_ap'],f'test_ap_{c}':r['test_ap'],f'C_{c}':r['C']})
        table.append(row)
    save_csv(output/'comparison.csv',table)
    save(output/'comparison.json',dict(manifest=manifest,records=table,
        timing_caveats=['SIFT is shared across six encodings; do not sum repeated rows.',
            'Encoder fitting may reuse a historical K-means fit; inspect encoder metadata/hardware.',
            'Encoding time excludes image extraction; inference includes matrix assembly.',
            'SIFT cache reads are reported separately; cache-miss extraction includes the provided cache write when enabled.'],
        dimensionality='Provided FisherVectorEncoder/skimage returns K*(2D+1); no components removed.'))
    lines=['# Part 2 — '+manifest.get('stage_id','H/I')+' shared comparison', '',
        'TECHNICAL SMOKE ONLY' if manifest['smoke_per_class'] else 'FULL DATASET RUN', '',
        '| Encoder | K | Dimension | Val mAP | Test mAP | Encoding s/image | Feature bytes/image |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['encoder']} | {r['k']} | {r['dimension']} | {r['validation_map']:.6f} | {r['test_map']:.6f} | {r['encoding_seconds_per_image']:.6f} | {r['descriptor_bytes_per_image']} |")
    lines += ['', 'Per-category validation/test AP, selected C and all timing/storage fields are in comparison.csv.', '',
        'VLAD K128 has 16384 components; Fisher K64 has 16448: +64 (+0.390625%).',
        'This is the supplied library variant, not exactly equal-dimensional 2KD Fisher.', '',
        'Compared with hard-assigned first-order VLAD residuals, Fisher uses soft GMM assignments and encodes '
        'first/second-order deviations; this implementation also retains mixture-weight statistics. '
        'Additional costs include diagonal-GMM estimation, posterior computation for each descriptor/component, '
        'and storing/classifying the larger vector. Similar final dimensions do not imply equal encoding costs.', '',
        'SIFT extraction is shared across methods. Encoder-fit times and hardware provenance must be inspected '
        'before comparing timings. A smoke run is not scientific evidence of full-data accuracy.']
    (output/('STAGE_I_REPORT.md' if manifest.get('stage_id')=='I' else 'STAGE_H_REPORT.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return table


def compatible_baseline(folder, current):
    folder=Path(folder)
    previous=load(folder/'manifest.json')
    if load(folder/'run.json').get('state')!='complete': raise ValueError('Stage H must complete first')
    for relative,sha in load(folder/'complete.json')['artifacts'].items():
        if not (folder/relative).is_file() or file_signature(folder/relative)!=sha:
            raise ValueError('Stage H baseline artifact changed')
    for name in ('split_sha256','image_sha256','rows','versions','sources'):
        if previous[name]!=current[name]: raise ValueError('Incompatible Stage H baseline: '+name)
    for name in ('vocabulary_sizes','C_values','categories','seed','pca_dim','images_per_class','descriptors_per_image','max_iter','tol'):
        if previous['config'][name]!=current['config'][name]: raise ValueError('Incompatible Stage H baseline setting: '+name)
    with (folder/'metrics.csv').open() as stream: rows=list(csv.DictReader(stream))
    # CSV -> typed metric records for the single consolidated reporting engine.
    for row in rows:
        for name,value in list(row.items()):
            if name in ('experiment','encoder','normalization','tiling','category','negative_class','split_sha256','training_split','selection_split','encoder_model','encoder_sha256','model','run_id','test_used_for_selection','selected_by_validation','smoke_per_class'): continue
            if value!='':
                try: row[name]=float(value)
                except ValueError: pass
        row['dimension']=int(row['dimension']); row['k']=int(row['k']); row['seed']=int(row['seed'])
    return rows
