"""Sequential A-G runner. Full experiments start only when this is executed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

STAGES = {
    'A': ('our_src.stagea.stage_a', 'our_src/stagea/stage_a.json'),
    'B': ('our_src.stageb.stage_b', 'our_src/stageb/config.json'),
    'C': ('our_src.stagec.stage_c', 'our_src/stagec/config.json'),
    'D': ('our_src.staged.stage_d', 'our_src/staged/config.json'),
    'E': ('our_src.stagee.stage_e', 'our_src/stagee/config.json'),
    'F': ('our_src.stagef.stage_f', 'our_src/stagef/config.json'),
    'G': ('our_src.stageg.stage_g', 'our_src/stageg/config.json'),
}
DEPENDENCIES = {'A': [], 'B': ['A'], 'C': ['A', 'B'], 'D': ['A', 'B'],
                'E': ['A', 'B'], 'F': ['E'], 'G': ['E', 'F']}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def complete(project, stage, config):
    state = read(project/config['output_dir']/'run.json')
    if state.get('state') != 'complete' or state.get('smoke_per_class') is not None or state.get('smoke_limit_per_category_split') is not None:
        raise ValueError(f'Stage {stage} has no completed full run')
    return state


def build_plan(project, python, start, end):
    letters = list(STAGES)
    if letters.index(start) > letters.index(end):
        raise ValueError('--start must come before --end')
    selected = letters[letters.index(start):letters.index(end)+1]
    configs = {s: read(project/STAGES[s][1]) for s in letters}
    reference = project/configs['A']['split_dir']
    for s, config in configs.items():
        if (project/config['split_dir']).resolve() != reference.resolve():
            raise ValueError(f'Stage {s} uses a different split directory')
        if config.get('smoke_per_class') is not None or Path(config['output_dir']).name == 'smoke':
            raise ValueError(f'Stage {s} default config is a smoke config')
    # Check that dependency paths are the outputs that this runner will produce.
    for s in letters[1:]:
        for key, source in [('stage_a_dir', 'A'), ('stage_b_dir', 'B'),
                            ('stage_e_dir', 'E'), ('stage_f_dir', 'F')]:
            if key in configs[s] and (project/configs[s][key]).resolve() != (project/configs[source]['output_dir']).resolve():
                raise ValueError(f'Stage {s}: {key} does not match Stage {source} output')
    plan = []
    for s in selected:
        for dependency in DEPENDENCIES[s]:
            if dependency not in selected:
                complete(project, dependency, configs[dependency])
        module, config_path = STAGES[s]
        plan.append((s, [str(python), '-u', '-m', module, '--config', str(project/config_path)], configs[s]))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--start', choices=list(STAGES), default='A')
    parser.add_argument('--end', choices=list(STAGES), default='G')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print commands without running experiments')
    parser.add_argument('--smoke', action='store_true', help='Run the small real-image A-G integration chain')
    parser.add_argument('--per-class', type=int, default=2)
    args = parser.parse_args()
    project = args.project.resolve()
    python = args.python or project/'.venv-stage-a/Scripts/python.exe'
    if not python.is_file():
        parser.error(f'Python not found: {python}; use --python')
    # Validate canonical image memberships even for a dry-run; never regenerate splits.
    sys.path.insert(0, str(project))
    from our_src.stagec.protocol import split_digest
    digest = split_digest(project/'data/splits_internal')
    if args.smoke:
        if args.per_class < 2:
            parser.error('--per-class must be at least 2')
        plan = [('SMOKE', [str(python), '-u', '-m', 'our_src.smoke_stages',
                           '--per-class', str(args.per_class), '--project', str(project)], None)]
    else:
        plan = build_plan(project, python, args.start, args.end)
    for stage, command, _ in plan:
        print(f'{stage}: {subprocess.list2cmdline(command)}', flush=True)
    if args.dry_run:
        print('Dry-run OK: no experiments started.', flush=True)
        return 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_dir = project/'pipeline_logs'/timestamp
    log_dir.mkdir(parents=True)
    status_path = log_dir/'pipeline.json'
    status = dict(state='running', mode='smoke' if args.smoke else 'full',
                  split_sha256=digest, started_utc=datetime.now(timezone.utc).isoformat(), stages=[])
    def record():
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    record()
    current = None
    try:
        for stage, command, config in plan:
            if split_digest(project/'data/splits_internal') != digest:
                raise ValueError('Historical splits changed during the pipeline')
            current = dict(stage=stage, state='running', command=command,
                           started_utc=datetime.now(timezone.utc).isoformat())
            status['stages'].append(current)
            record()
            print(f'\nStarting {stage}; log: {log_dir/stage}.log', flush=True)
            with (log_dir/f'{stage}.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
                try:
                    for line in child.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    exit_code = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            if exit_code:
                raise RuntimeError(f'Stage {stage} failed with exit code {exit_code}; see {log_dir/stage}.log')
            if config is not None:
                complete(project, stage, config)
            else:
                summary = read(project/'smoke_summary.json')
                if any(summary.get(s, {}).get('state') != 'complete' for s in 'ABCDEFG'):
                    raise ValueError('Smoke chain did not complete every stage')
            current.update(state='complete', finished_utc=datetime.now(timezone.utc).isoformat())
            record()
        if split_digest(project/'data/splits_internal') != digest:
            raise ValueError('Splits changed during the pipeline')
        status['state'] = 'complete'
    except BaseException as error:
        status.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        if current is not None and current['state'] == 'running':
            current['state'] = status['state']
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        record()
    print(f'Pipeline complete. Logs: {log_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
