"""Run A-H on a tiny real-image sample without changing historical splits."""
import argparse
import csv
from pathlib import Path
from .artifact_json import load, save
from .stagea.stage_a import run as run_a
from .stageb.stage_b import run as run_b
from .stagec.stage_c import run as run_cd
from .stagee.stage_e import run as run_e
from .stagef.stage_f import run as run_f
from .stageg.stage_g import run as run_g
from .stageh.stage_h import run as run_h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class', type=int, default=2)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--stages', nargs='+', choices=list('abcdefgh'), default=list('abcdefgh'))
    args = parser.parse_args()
    project = args.project
    summary_path = project/'smoke_summary.json'
    summary = load(summary_path) if summary_path.exists() else {}
    if args.per_class < 2:
        parser.error('--per-class must be at least 2')
    for stage in ('a', 'b'):
        if stage not in args.stages:
            continue
        output = f'results/part1/stage{stage.upper()}/smoke'
        print(f'Running Stage {stage.upper()} real-data SMOKE', flush=True)
        if stage == 'a':
            config = load(Path(__file__).parent/'stagea/config.json')
            config['vocabulary_sizes'] = [32]
            save(project/output/'smoke_config.json', config)
            run_a(config, project, limit=args.per_class)
            state = load(project/output/'run.json')
        else:
            previous = load(project/'results/part1/stageA/smoke/run.json')
            if previous.get('state') != 'complete':
                raise ValueError('Stage A smoke must finish before B smoke')
            with (project/'results/part1/stageA/smoke/svm_ap.csv').open(newline='', encoding='utf-8') as stream:
                candidates = [r for r in csv.DictReader(stream) if r['category'] == 'motorbike']
            best = max(candidates, key=lambda r: float(r['validation_ap']))
            config = load(Path(__file__).parent/'stageb/config.json')
            config['stage_a_dir'] = 'results/part1/stageA/smoke'
            save(project/output/'smoke_config.json', config)
            state = run_b(config, project, k=int(best['k']), layout=best['layout'],
                          normalization=best['normalization'], smoke=args.per_class, render=False)
        summary[stage.upper()] = dict(state=state['state'], output_dir=output,
                                      smoke_per_class=args.per_class)
        save(summary_path, summary)
        print(f'Stage {stage.upper()}: {state["state"]}', flush=True)
    for stage, runner in [('c', run_cd), ('d', run_cd), ('e', run_e), ('f', run_f), ('g', run_g), ('h', run_h)]:
        if stage not in args.stages:
            continue
        config = load(Path(__file__).parent/f'stage{stage}'/'config.json')
        config.update(smoke_per_class=args.per_class, output_dir=f'results/part1/stage{stage.upper()}/smoke',
                      stage_b_dir='results/part1/stageB/smoke', stage_a_dir='results/part1/stageA/smoke')
        if stage in ('f', 'g'):
            config['stage_e_dir'] = 'results/part1/stageE/smoke'
        if stage == 'g':
            config.update(stage_f_dir='results/part1/stageF/smoke', fractions=[.5, 1.])
        save(project/config['output_dir']/'smoke_config.json', config)
        print(f'Running Stage {stage.upper()} real-data SMOKE', flush=True)
        state = runner(config, project)
        summary[stage.upper()] = dict(state=state['state'], output_dir=config['output_dir'],
                                      smoke_per_class=args.per_class)
        save(project/'smoke_summary.json', summary)
        print(f'Stage {stage.upper()}: {state["state"]}', flush=True)


if __name__ == '__main__':
    main()
