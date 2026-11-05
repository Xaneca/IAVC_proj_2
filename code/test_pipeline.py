import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import run_part_1 as pipeline


class PipelineTests(unittest.TestCase):
    def test_plan_order_and_existing_dependencies(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            for stage, (_, relative) in pipeline.STAGES.items():
                config = dict(split_dir='data/splits_internal', output_dir=f'stage{stage}')
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(config))
            plan = pipeline.build_plan(root, Path(sys.executable), 'A', 'G')
            self.assertEqual([item[0] for item in plan], list('ABCDEFG'))
            with self.assertRaises(FileNotFoundError):
                pipeline.build_plan(root, Path(sys.executable), 'E', 'G')
            for stage in 'AB':
                path = root/f'stage{stage}/run.json'
                path.parent.mkdir()
                path.write_text(json.dumps(dict(state='complete')))
            self.assertEqual([r[0] for r in pipeline.build_plan(root, Path(sys.executable), 'E', 'G')], list('EFG'))
            (root/'stageB/run.json').write_text(json.dumps(dict(state='complete', smoke_per_class=2)))
            with self.assertRaises(ValueError):
                pipeline.build_plan(root, Path(sys.executable), 'E', 'G')

    def test_failure_stops_chain_and_records_log(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            protocol = types.ModuleType('our_src.stagec.protocol')
            protocol.split_digest = lambda _: 'fixture'
            child = Mock()
            child.stdout = iter(['training failed\n'])
            child.wait.return_value = 7
            plan = [('A', ['fake', 'A'], {'output_dir': 'stageA'}),
                    ('B', ['fake', 'B'], {'output_dir': 'stageB'})]
            with patch.dict(sys.modules, {'our_src.stagec.protocol': protocol}), patch.object(sys, 'argv',
                ['run_part_1.py', '--project', str(root), '--python', sys.executable]), patch.object(pipeline,
                'build_plan', return_value=plan), patch.object(pipeline.subprocess, 'Popen', return_value=child) as popen, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
                    pipeline.main()
            self.assertEqual(popen.call_count, 1)
            log = next((root/'pipeline_logs').glob('*/pipeline.json'))
            state = json.loads(log.read_text())
            self.assertEqual(state['state'], 'failed')
            self.assertEqual(len(state['stages']), 1)
            self.assertEqual(state['stages'][0]['state'], 'failed')
            self.assertIn('training failed', (log.parent/'A.log').read_text())


if __name__ == '__main__':
    unittest.main()
