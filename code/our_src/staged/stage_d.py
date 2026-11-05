from pathlib import Path
from ..stagec.stage_c import main

if __name__ == '__main__':
    main(Path(__file__).with_name('config.json'))
