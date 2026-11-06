"""Part 2 H/I orchestration; shared implementation, separate stage outputs."""
import argparse
from pathlib import Path
import subprocess
import sys
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages',nargs='+',choices=['H','I'],default=['H','I'])
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--per-class',type=int,default=2)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--sift-cache')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.threads<1: parser.error('--threads must be positive')
    project=Path(__file__).resolve().parents[2]
    environment=dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): environment[name]=str(args.threads)
    for stage in args.stages:
        command=[sys.executable,'-B','-m',f'our_src.part2.stage{stage.lower()}.stage_{stage.lower()}']
        if args.smoke: command+=['--smoke-per-class',str(args.per_class)]
        if args.preflight: command+=['--preflight']
        if args.sift_cache: command+=['--sift-cache',args.sift_cache]
        print('Part 2 '+stage+': '+subprocess.list2cmdline(command),flush=True)
        result=subprocess.run(command,cwd=project,env=environment)
        if result.returncode: raise SystemExit(result.returncode)


if __name__=='__main__': main()
