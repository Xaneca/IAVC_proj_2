"""Contract/integration tests; synthetic features are never benchmark results."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from . import common, classification, fewshot, features, visual_analysis

PROJECT = Path(__file__).resolve().parents[2]


def fixture():
    rows=[]; vectors={}
    for split in ('train','val','test'):
        for category,vector in [('cat',[1,0,0]),('dog',[0,1,0]),('background',[0,0,1])]:
            for n in range(6):
                image_id=f'{split}_{category}_{n}'
                rows.append(dict(image_id=image_id,split=split,labels=[category]))
                vectors[image_id]=np.array(vector,dtype=np.float32)
    return rows,vectors


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.rows,self.vectors=fixture()
        self.config=dict(categories=['cat','dog','background'],C_values=[.1,1,10,100,1000],
            max_iter=20000,tol=1e-4,seed=42)

    def test_exact_binary_subsets_nested_fixed_background(self):
        rows=list(self.rows)
        rows[0]=dict(rows[0],labels=['cat','dog'])
        previous={}; negatives={}
        first=list(fewshot.subsets(rows,self.config['categories'],[1,5,'all'],3))
        self.assertEqual(first,list(fewshot.subsets(rows,self.config['categories'],[1,5,'all'],3)))
        for c,n,selected,counts in first:
            positive={r['image_id'] for r in selected if c in r['labels']}
            negative={r['image_id'] for r in selected if 'background' in r['labels']}
            self.assertEqual(len(positive),counts['positives'])
            if n!='all': self.assertEqual(len(positive),n)
            self.assertTrue(previous.get(c,set())<=positive)
            self.assertEqual(negative,negatives.get(c,negative))
            self.assertTrue(all(r['split']=='train' and (c in r['labels'] or 'background' in r['labels']) for r in selected))
            previous[c]=positive; negatives[c]=negative
        with self.assertRaises(ValueError): list(fewshot.subsets(rows,['cat'],[20],0))

    def test_dino_pooling_order_grid_and_l2(self):
        import torch
        tokens=torch.tensor([[[1.],[2.],[3.],[4.],[5.],[6.],[7.],[8.]]])
        values=features.pool_patches(tokens,(28,56),14)
        np.testing.assert_allclose(values['spatial2x2'],[[1.5,3.5,5.5,7.5]])
        np.testing.assert_allclose(values['patch_mean'],[[4.5]])
        self.assertEqual(features.DIMENSIONS['dinov2_vits14'],{'global':384,'patch_mean':384,'spatial2x2':1536})
        with self.assertRaises(ValueError): features.patch_grid(tokens,(28,28),14)
        x=common.l2(np.array([[0,0],[3,4]],dtype=np.float32))
        np.testing.assert_allclose(x,[[0,0],[.6,.8]])

    def test_independent_C_selection_and_trial_reuse(self):
        calls=[]
        def trainer(features,train,val,params,seed):
            c=params['C'][0]; category=next(i for i,v in train.items() if v==1).split('_')[1]
            calls.append((category,c))
            trial=dict(C=c,train_ap=.95,validation_ap=float(c==(.1 if category=='cat' else 10)),
                fit_seconds=.1,validation_inference_seconds=.01,train_images=len(train),validation_images=len(val))
            return dict(best=dict(trial,weights=np.ones(3),bias=0.),trials=[trial])
        actual=common.module('svm',PROJECT)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d, patch.object(actual,'train_svm_grid',trainer):
            first=classification.search(self.vectors,self.rows,dict(self.config,C_values=[.1,1,10]),PROJECT,0,True,d)
            self.assertEqual(first['selected']['C_by_category'],{'cat':.1,'dog':10})
            self.assertEqual(len(calls),6)
            extended=classification.search(self.vectors,self.rows,self.config,PROJECT,0,True,d)
            self.assertEqual(len(calls),10)
            self.assertEqual(len(extended['trials']),5)
            self.assertTrue(all('train_ap' in r and 'validation_ap' in r for t in extended['trials'] for r in t['per_class']))

    def test_signed_occlusion_same_score(self):
        class Encoder:
            def encode(self,images):
                return {'global':np.array([[np.asarray(im).mean()] for im in images])},{}
        a=np.zeros((4,4,3),dtype=np.uint8); a[:2,:2]=255
        heat=visual_analysis.occlusion(Encoder(),Image.fromarray(a),lambda values:values['global'][:,0],2)
        self.assertGreater(heat[0,0],0); self.assertLess(heat[1,1],0)

    def test_callable_preprocessing_identity_survives_process_addresses(self):
        def make_transform():
            def convert(image):
                return image
            return convert
        first,second=make_transform(),make_transform()
        self.assertNotEqual(repr(first),repr(second))
        self.assertEqual(features.preprocessing_description(first),features.preprocessing_description(second))
        class Transform:
            def __repr__(self):
                return 'CenterCrop(size=(224, 224)); Normalize(mean=(0.4, 0.5, 0.6))'
        self.assertEqual(features.preprocessing_description(Transform()),repr(Transform()))

    def test_real_shared_svm_selects_without_accessing_test_features(self):
        # Exercise professor train_svm and existing train_svm_grid together.
        vectors={i:v for i,v in self.vectors.items() if not i.startswith('test_')}
        result=classification.search(vectors,self.rows,self.config,PROJECT,42)
        self.assertEqual(len(result['trials']),5)
        self.assertEqual(result['selected']['validation_map'],1.)
        self.assertEqual(result['selected']['C_by_category']['cat'],.1)
        binary=classification.search(vectors,self.rows,self.config,PROJECT,42,True)
        self.assertEqual(binary['protocol'],'target_vs_background')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            reports=classification.persist_scores(vectors,self.rows,result,d,self.config,PROJECT,False)
            self.assertEqual(set(reports),{'train','val'})
            self.assertFalse((Path(d)/'test_predictions.npz').exists())
            with np.load(Path(d)/'model.npz',allow_pickle=False) as model:
                self.assertEqual(model['weights'].shape,(2,3))

    def test_multilabel_metrics_and_clip_single_prediction_are_explicit(self):
        rows=[dict(labels=['cat','dog']),dict(labels=['background']),dict(labels=['cat'])]
        scores=np.array([[2,3],[-2,-3],[1,-1]])
        report,predicted=classification.metrics(rows,['cat','dog'],scores)
        self.assertEqual(report['accuracy'],1.)
        self.assertEqual(np.asarray(report['confusion_matrix']).shape,(2,2,2))
        single,_=classification.metrics(rows,['cat','dog'],scores,True)
        self.assertLess(single['accuracy'],1.)
        self.assertEqual(single['top1_label_membership_accuracy'],2/3)

    def test_nested_multilabel_fewshot_and_no_split_leakage(self):
        rows=list(self.rows)
        rows[0]=dict(rows[0],labels=['cat','dog'])
        previous=set()
        for n,selected,counts in fewshot.multilabel_subsets(rows,self.config['categories'],[1,5,'all'],0):
            ids={r['image_id'] for r in selected}
            self.assertTrue(previous<=ids)
            self.assertTrue(all(r['split']=='train' for r in selected))
            self.assertGreaterEqual(counts['cat']['positives'],1)
            previous=ids
        with self.assertRaises(ValueError):
            list(fewshot.multilabel_subsets(rows,self.config['categories'],[20],0))

    def test_real_image_flip_cache_and_image_change_invalidation(self):
        class Encoder:
            name='resnet18'; metadata={'test_double':True}
            calls=[]
            def encode(self,images):
                self.calls.extend(np.asarray(i).copy() for i in images)
                return {k:np.ones((len(images),d),dtype=np.float32) for k,d in features.DIMENSIONS[self.name].items()},dict(backbone_seconds=.01)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            root=Path(d); (root/'images').mkdir()
            a=np.zeros((4,6,3),dtype=np.uint8); a[:,0]=255
            Image.fromarray(a).save(root/'images/a.png')
            Image.fromarray(a).save(root/'images/b.png')
            rows=[dict(image_id='a.png',split='train',labels=['cat']),dict(image_id='b.png',split='val',labels=['cat'])]
            encoder=Encoder(); config=dict(image_dir='images',batch_size=2)
            result=features.extract(encoder,rows,root,config,root/'cache',True)
            self.assertEqual(set(result['resnet18_layer2']['features']),{'a.png'})
            self.assertTrue(np.array_equal(encoder.calls[-1],a[:,::-1]))
            count=len(encoder.calls)
            features.extract(encoder,rows,root,dict(config,C_values=[1000]),root/'cache',True)
            self.assertEqual(len(encoder.calls),count)
            Image.fromarray(255-a).save(root/'images/a.png')
            features.extract(encoder,rows,root,config,root/'cache',True)
            self.assertEqual(len(encoder.calls),count+1)

    def test_visual_case_selection_uses_actual_predictions(self):
        predictions={name:{'a':dict(truth=['cat'],prediction=p)} for name,p in [('cnn','cat'),('dino','dog'),('clip','cat')]}
        groups=visual_analysis.select(predictions,2)
        self.assertEqual(groups['cnn_only'],['a'])
        self.assertEqual(groups['all_correct'],[])

    def test_clip_mean_similarity_and_persistence_contract(self):
        # A boundary double tests orchestration, not real CLIP availability.
        import torch
        from .clip_experiments import run
        class Model:
            def encode_text(self,tokens):
                return torch.tensor([[1.,0,0],[0,1,0],[0,0,1],[1.,1,0],[0,1,1],[1.,0,1]])[tokens]
        class Backbone:
            device='cpu'; model=Model()
            def sync(self): pass
        b=Backbone(); b.torch=torch
        cfg=dict(self.config,prompt_templates=['a {}','the {}'],class_descriptions={c:c for c in self.config['categories']})
        fake=types.SimpleNamespace(tokenize=lambda prompts:torch.tensor([self.config['categories'].index(p.split()[-1])+(3 if p.startswith('the ') else 0) for p in prompts]))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d, patch.dict(sys.modules,{'clip':fake}):
            results=run(b,dict(features={i:v for i,v in self.vectors.items() if not i.startswith('test_')}),self.rows,cfg,PROJECT,d,False)
            self.assertEqual(results['multiple']['val']['map'],1.)
            self.assertEqual(results['single']['val']['map'],1.)
            self.assertFalse((Path(d)/'single/test_predictions.npz').exists())
            self.assertEqual(common.read(Path(d)/'template_selection.json')['selected']['index'],0)
            with np.load(Path(d)/'multiple/val_predictions.npz') as z:
                # First cat: average cosines (1 + 1/sqrt(2))/2, not normalized mean text.
                self.assertAlmostEqual(float(z['scores'][0,0]),(1+1/np.sqrt(2))/2,places=6)

    @unittest.skipUnless((PROJECT/'data/images').is_dir() and (PROJECT/'data/splits_internal').is_dir(), 'Requires the separately supplied dataset and historical split setup')
    def test_historical_manifest_and_smoke_use_original_images(self):
        config=common.read(Path(__file__).with_name('config.json'))
        rows,sha=common.dataset(PROJECT,config,2)
        self.assertEqual(len(sha),64)
        self.assertGreater(len(rows),0)
        for split in ('train','val','test'):
            for category in config['categories']:
                self.assertGreaterEqual(sum(r['split']==split and category in r['labels'] for r in rows),2)

    @unittest.skipUnless((PROJECT/'data/images').is_dir() and (PROJECT/'data/splits_internal').is_dir(), 'Requires the separately supplied dataset and historical split setup')
    def test_full_runner_and_resume_with_explicit_backbone_double(self):
        """Actual images/splits/SVM/metrics, fake backbones: integration only."""
        import torch
        from . import run as runner
        config=common.read(Path(__file__).with_name('config.json'))
        rows,_=common.dataset(PROJECT,config,2)
        config.update(C_values=[.1],fewshot_seeds=[0],positive_counts=[1,'all'],visual_examples=1,occlusion_grid=2)
        class TextModel:
            def encode_text(self,tokens):
                return torch.nn.functional.one_hot(tokens,num_classes=512).float()
        class Double:
            calls=0
            def __init__(self,project,name,config):
                self.name=name; self.metadata=dict(model=name,integration_test_double=True)
                self.torch=torch; self.device='cpu'; self.model=TextModel()
            def encode(self,images):
                Double.calls+=1
                # Image-dependent descriptors, unrelated to ground-truth labels.
                x=np.stack([np.asarray(im.resize((4,4))).flatten()/255 for im in images]).astype(np.float32)
                return {layer:np.tile(x,(1,(d+47)//48))[:,:d] for layer,d in features.DIMENSIONS[self.name].items()},dict(backbone_seconds=0.)
            def sync(self): pass
            def close(self): pass
        fake=types.SimpleNamespace(tokenize=lambda prompts:torch.arange(len(prompts))%6)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            root=Path(d); cfg=root/'config.json'; common.save(cfg,config)
            args=Namespace(project=PROJECT,config=cfg,output=root/'output',smoke=None,final=False,
                dry_run=False,stages=list('ABCDEF'))
            with (patch.object(runner,'Backbone',Double),patch.object(runner,'dataset',return_value=(rows,'fixture_split_sha')),
                    patch.dict(sys.modules,{'clip':fake})):
                runner.execute(args)
                state=next((root/'output/runs').glob('*/run.json'))
                self.assertEqual(common.read(state)['state'],'complete')
                before=Double.calls
                runner.execute(args)
                self.assertEqual(Double.calls,before)
                self.assertEqual(set(common.read(state)['stages']),set('ABCDEF'))


if __name__=='__main__':
    unittest.main()
