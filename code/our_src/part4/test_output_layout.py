"""Lightweight output-layout/resume checks; no model loading or inference."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import run as runner
from .common import digest, file_hash, read, save


class OutputLayoutTests(unittest.TestCase):
    def test_stage_paths_and_representation_ownership(self):
        for mode in ('runs', 'smoke'):
            root=Path('results/part4')/mode/'example'
            for stage in 'ABCDEF':
                self.assertEqual(runner.stage_output(root,stage),Path('results/part4')/('stage'+stage)/mode/'example')
            for name,stage in [('resnet18_layer2','A'),('dinov2_vits14_global','B'),('clip_vit_b_32_global','E')]:
                self.assertEqual(runner.classification_output(root,name),runner.stage_output(root,stage)/'classification'/name)
        with self.assertRaises(ValueError): runner.stage_output(root,'invalid')

    def test_resume_checks_artifacts_in_stage_directory_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); cfg=base/'config.json'; save(cfg,{'output_dir':'results/part4'})
            identity={'layout_fixture':True}; run_id=digest(identity)[:20]
            root=base/'results/part4/runs'/run_id
            artifact=runner.stage_output(root,'A')/'classification/resnet18_layer2/metrics.json'
            save(artifact,{'fixture':True})
            save(root/'manifests/A.json',{'artifacts':{artifact.relative_to(base/'results/part4').as_posix():file_hash(artifact)}})
            args=Namespace(project=base,config=cfg,output=None,smoke=None,final=False,dry_run=False,stages=['A'])
            with patch.object(runner,'dataset',return_value=([],'fixture')), patch.object(runner,'signature',return_value=identity), patch.object(runner,'report'), patch.object(runner,'Backbone',side_effect=AssertionError('Model loading forbidden')):
                runner.execute(args)
                self.assertEqual(read(root/'run.json')['stages']['A'],'complete')
                save(artifact,{'fixture':'changed'})
                with self.assertRaisesRegex(ValueError,'artifacts changed'): runner.execute(args)


    def test_minimal_ab_smoke_writes_and_resumes_stage_outputs(self):
        # Exercise the real runner and manifests with tiny IO-only doubles.
        # No checkpoints, extraction, SVM fitting or inference are performed.
        class Double:
            def __init__(self, project, name, config): self.name=name
            def close(self): pass
        def extract(backbone, rows, project, config, base):
            metadata={'model':backbone.name}
            save(base/'features'/backbone.name/'fixture/manifest.json',{'files':{},'file_sha256':{}})
            return {backbone.name+'_global':{'metadata':metadata,'identity':'fixture','features':{}}}
        def classify(rep, rows, config, project, output, final):
            save(output/'metrics.json',{'fixture_only':True})
        for smoke in (None,2):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as directory:
                base=Path(directory); cfg=base/'config.json'; save(cfg,{'output_dir':'results/part4'})
                args=Namespace(project=base,config=cfg,output=None,smoke=smoke,final=False,dry_run=False,stages=['A','B'])
                identity={'smoke':smoke}; run_id=digest(identity)[:20]
                root=base/'results/part4'/('smoke' if smoke else 'runs')/run_id
                with patch.object(runner,'dataset',return_value=([],'fixture')), patch.object(runner,'signature',return_value=identity), patch.object(runner,'Backbone',Double), patch.object(runner,'extract',side_effect=extract) as extraction, patch.object(runner,'classify',side_effect=classify), patch.object(runner,'report'):
                    runner.execute(args)
                    self.assertEqual(extraction.call_count,2)
                    for stage,name in [('A','resnet18_global'),('B','dinov2_vits14_global')]:
                        destination=runner.stage_output(root,stage)/'classification'/name
                        self.assertTrue((destination/'metrics.json').is_file())
                        marker=read(root/'manifests'/(stage+'.json'))
                        self.assertTrue(marker['artifacts'])
                        self.assertTrue(all((base/'results/part4'/path).is_file() for path in marker['artifacts']))
                    runner.execute(args)
                    self.assertEqual(extraction.call_count,2)
                    self.assertEqual(read(root/'run.json')['stages'],{'A':'complete','B':'complete'})
                    args.stages=['F']
                    with self.assertRaisesRegex(ValueError,'Stage E must complete first'): runner.execute(args)


if __name__=='__main__': unittest.main()
