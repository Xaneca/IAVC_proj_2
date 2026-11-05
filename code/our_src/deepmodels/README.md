# Part 4 - frozen representations

Canonical source: `code/our_src/deepmodels/`, a sibling of the remote repository's
`code/our_src/part1/`. Entry point: `code/run_deepmodels.py`. No external editing
workspace or machine-specific Python path is required. All commands below start
at the Git repository root (the directory containing `code/`). The entry point
resolves its default project directory from the installed package location.

## Install on Linux

Python 3.12 and Git are required. Create an isolated Part 4 environment; the
professor's requirements/environment files and Part 1 remain unchanged.

```bash
python3.12 -m venv .venv-part4
source .venv-part4/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r code/requirements-deepmodels.txt
python code/run_deepmodels.py --help
```

For CUDA, install the matching torch/torchvision pair for the target hardware
instead of the CPU command, then use the same requirements file. The recorded
real smoke ran on Windows/Python 3.12/CPU; Linux dependency installation and
runtime have not been executed in this audit. The portable requirements pin the
versions used in that smoke (CLIP also pins its Git commit). First installation
needs package-index/GitHub access. No cyvlfeat is needed for Part 4's shared
manifest, SVM or evaluation adapters; it remains a separate Part 1 dependency.

Windows equivalent:

```powershell
py -3.12 -m venv .venv-part4
.\.venv-part4\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r code/requirements-deepmodels.txt
python code/run_deepmodels.py --help
```

## Dataset and immutable splits

Obtain the original assignment images from the course materials separately and
place them under `code/data/images/` as `<image_id>.jpg`. They are deliberately
not committed. Bootstrap the internal split directory from the checked-in
historical references; do not regenerate random splits or overwrite different
existing split files. On Linux, from the Git root:

```bash
python - <<'PY'
from pathlib import Path
import shutil
base = Path('code')
reference = next(p for p in (
    base / 'our_src/part1/stagec/split_reference',
    base / 'our_src/stagec/split_reference',
) if p.is_dir())
target = base / 'data/splits_internal'
target.mkdir(parents=True, exist_ok=True)
for source in reference.glob('*.txt'):
    destination = target / source.name
    if destination.exists():
        assert set(destination.read_text().split()) == set(source.read_text().split()), destination
    else:
        shutil.copyfile(source, destination)
PY
```

The same Python code can be saved temporarily and run on Windows. The reference
folder belongs to existing Part 1 and is reused, not duplicated. The adapter also
supports the older checkout's Part 1 module locations.

## Configuration and checkpoints

`config.json` contains unchanged experimental settings. `--config` accepts an
alternate JSON file. `image_dir` and `split_dir` are resolved relative to
`--project` (default: this repository's `code/`); absolute paths are also accepted.
`output_dir` defaults to `deepmodels` beneath that project directory. `--output`
overrides the base containing both feature caches and experiment outputs and is
resolved relative to the invoking shell. It can point to external storage.

ResNet uses torchvision official weights/transforms, DINOv2 uses the professor's
Torch Hub loader plus our official evaluation preprocessing correction, and CLIP
uses OpenAI's loader/tokenizer. All remain frozen. First model load can download
checkpoints. Torch Hub/torchvision caches can be moved with `TORCH_HOME`; OpenAI
CLIP's provided loader uses `~/.cache/clip`. Prepopulate these on offline machines.
Checkpoint files and generated embeddings are not repository source. Actual
weight hashes, preprocessing, software versions and hardware are recorded per
experiment; models are never replaced with random weights.

## Execution

Data/path validation only (no inference, no training):

```bash
python code/run_deepmodels.py --smoke 2 --dry-run
```

On the destination machine, when explicitly ready to run experiments:

```bash
python code/run_deepmodels.py --stages A
python code/run_deepmodels.py --stages B
python code/run_deepmodels.py --stages C
python code/run_deepmodels.py --stages D
python code/run_deepmodels.py --stages E
python code/run_deepmodels.py --stages F
# Equivalent complete development benchmark, with no test scoring:
python code/run_deepmodels.py
# Separate final evaluation, only after fixing all selections:
python code/run_deepmodels.py --final
```

A/B precede C; C precedes D; A/B/E precede F. Default order is A-B-C-D-E-F.
Keep config, output base, smoke/final options and hardware/thread settings
consistent for partial runs. Use `--smoke 2` for a future 33-image technical run;
it still performs real model inference, so it was NOT executed in this audit.
Existing implementation tests can be run later from `code/`:

```bash
cd code
python -m unittest our_src.deepmodels.test_integration -v
```

Dataset-dependent integration tests skip if the separately supplied images and
internal splits are absent. No tests depend on a former editing workspace.

## Protocol, outputs and status

See [PROTOCOL.md](PROTOCOL.md) for representations, unchanged grids/seeds, exact-N
sampling, validation-only selection and metric definitions. [AUDIT.md](AUDIT.md),
[CHANGE_REPORT.md](CHANGE_REPORT.md) and [VALIDATION.md](VALIDATION.md) retain the
historical implementation/validation record. Historical logs and the old local
smoke harness are archival evidence, not files required for clone execution.

Outputs default to `code/deepmodels/`: content-addressed `features/`, distinct
`runs/` and technical `smoke/`, with manifests/checksums, raw/L2 features,
per-category searches/metrics, few-shot and regularization curves, CLIP prompt
searches, signed maps and protocol-separated comparison tables. Keep this entire
generated tree outside Git. Do not sum shared ResNet/DINO extraction costs across
representations. Missing compatible classical or final-test results remain null.

IMPLEMENTED and previously TESTED: A-F, including a real pretrained 33-image
smoke. This repository-integration audit performs only static checks, shared-module
imports and CLI help. No benchmark, inference, checkpoint download or new
scientific performance validation was performed. Full benchmark and final test
reporting remain pending.
