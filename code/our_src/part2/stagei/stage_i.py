"""Part 2 stage entry point using the shared, supplied-encoder integration."""
from ..shared.comparison import run as _run, main as _main


def run(config, project):
    return _run(dict(config,stage_id='I',encoders=['fv']),project)


def main():
    _main('I')


if __name__=='__main__': main()
