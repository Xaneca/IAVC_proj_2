"""Focused contracts for frozen test evaluation; no hyperparameter search."""
from pathlib import Path
import tempfile
import unittest
import json
import hashlib
import numpy as np
from .test_evaluation import select_cases,selected_centroid,latex_table
from unittest.mock import patch


class FrozenTestContracts(unittest.TestCase):
    def test_cases_use_zero_threshold_and_hard_negative_order(self):
        records=[dict(image_id=str(i),label=y,score=s,rank=i+1,labels=['motorbike' if y==1 else 'background'],prediction='motorbike' if s>=0 else 'background')
            for i,(y,s) in enumerate([(1,4),(1,0),(1,-4),(1,-1),(-1,8),(-1,2),(-1,-.01),(-1,-.1),(-1,-5)])]
        groups=select_cases(records)
        self.assertEqual([r['score'] for r in groups['TP']],[4,0])
        self.assertEqual([r['score'] for r in groups['FN']],[-4,-1])
        self.assertEqual([r['score'] for r in groups['FP']],[8,2])
        self.assertEqual([r['score'] for r in groups['Hard Negative']],[-.01,-.1])

    def test_missing_groups_do_not_invent_examples(self):
        self.assertEqual(select_cases([]),{'TP':[],'FN':[],'FP':[],'Hard Negative':[]})
        one=dict(image_id='a',label=-1,score=-.2,rank=1,labels=['background'],prediction='background')
        result=select_cases([one]); self.assertEqual(result['Hard Negative'],[one]); self.assertEqual(result['FN'],[])

    def test_exact_vocabulary_checksum_and_training_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'src').mkdir(); (root/'stageA/centroids').mkdir(parents=True)
            content=b'# identical source\r\n'; (root/'src/features.py').write_bytes(content)
            data=dict(identity=dict(training_ids=['train'],sample_image_ids=['train'],sources={'features.py':hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()}),centroids=np.zeros((2,128)).tolist())
            p=root/'stageA/centroids/model.json'; p.write_text(json.dumps(data)); sha=hashlib.sha256(p.read_bytes()).hexdigest()
            cfg=dict(stage_a_dir='stageA',split_dir='splits',image_dir='images')
            meta=dict(centroid_sha256=sha,encoding=dict(k=2)); state=dict(centroid_sha256=sha)
            with patch('our_src.part1.stageb.test_evaluation.manifest',return_value=[dict(image_id='train',split='train')]):
                self.assertEqual(selected_centroid(root,cfg,state,meta)['centroids'].shape,(2,128))
                with self.assertRaises(ValueError): selected_centroid(root,cfg,dict(centroid_sha256='wrong'),meta)

    def test_latex_exports_metrics_and_available_cases(self):
        report=dict(ap=.5,accuracy=.6,precision=.7,recall=.8,f1=.75,images=12)
        metrics,cases=latex_table(report,select_cases([]))
        self.assertIn('0.500000',metrics); self.assertIn('tabular',cases)


if __name__=='__main__': unittest.main()
