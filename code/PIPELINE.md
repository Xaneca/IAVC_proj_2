# Part 1 pipeline: A-H

From the Git repository root, enter `code/`. The canonical runner is
`run_part_1.py` (`run_part_1.ps1` on Windows); modules live beneath
`our_src/part1/stagea` through `stageh`. A-G follow the fetched repository layout;
H preserves our local BoVW/VLAD/Fisher comparison extension.

The full pipeline uses the existing stage configurations and unchanged historical
splits. All newly generated Part 1 outputs use `results/part1/stageA` through
`stageH`. Previous local `stageA` through `stageH` directories are retained as
historical artifacts and are not overwritten or silently merged with upstream
results. Stage H remains independently runnable and uses its own configurations.

## Validate paths only (no experiments)

Windows PowerShell, from `code/`, with the intended environment active:

```powershell
./run_part_1.ps1 -DryRun
```

Linux, from `code/`, with the intended Part 1 environment active:

```bash
python run_part_1.py --python "$(command -v python)" --dry-run
```

Part 1 needs its documented cyvlfeat/SIFT dependencies. Choose an environment
containing those dependencies before running experiments. CLI path validation
prints commands without launching the stages.

## Future execution, only when explicitly ready

```powershell
./run_part_1.ps1
./run_part_1.ps1 -Start E
./run_part_1.ps1 -Start H -End H
./run_part_1.ps1 -Smoke -PerClass 2
```

```bash
python run_part_1.py --python "$(command -v python)"
python run_part_1.py --python "$(command -v python)" --start H --end H
```

Smoke still runs actual extraction/training; it was NOT run during repository
synchronization. It uses the nested `results/part1/stageX/smoke` directories,
K=32 in A, K=64/128 in H and the existing seed/fraction protocol. H reuses or fits
its encoder models using training images only. Main pipeline logs are retained
under `pipeline_logs/`; stop-on-failure behavior is preserved.

For frozen representations (Part 4 A-F), use `run_deepmodels.py` and see
`our_src/deepmodels/README.md`. This is separate from the classical A-H pipeline.
