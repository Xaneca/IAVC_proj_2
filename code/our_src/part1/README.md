# Part 1 pipeline: A-G

From the Git repository root, enter `code/`. The canonical runner is
`our_src/part1/run.py` (`run_part_1.ps1` on Windows); modules live beneath
`our_src/part1/stagea` through `stageg`. Stages H and I are in Part 2; see
[`../part2/README.md`](../part2/README.md).

The full pipeline uses the existing stage configurations and unchanged historical
splits. All newly generated Part 1 outputs use `results/part1/stageA` through
`stageG`. Historical outputs remain in their original locations and are not
overwritten or silently merged with upstream results.

## Validate paths only (no experiments)

Windows PowerShell, from `code/`, with the intended environment active:

```powershell
./run_part_1.ps1 -DryRun
```

Linux, from `code/`, with the intended Part 1 environment active:

```bash
python -m our_src.part1.run --python "$(command -v python)" --dry-run
```

Part 1 needs its documented cyvlfeat/SIFT dependencies. Choose an environment
containing those dependencies before running experiments. CLI path validation
prints commands without launching the stages.

## Future execution, only when explicitly ready

```powershell
./run_part_1.ps1
./run_part_1.ps1 -Start E
./run_part_1.ps1 -Smoke -PerClass 2
```

```bash
python -m our_src.part1.run --python "$(command -v python)"
python -m our_src.part1.run --python "$(command -v python)" --start E
```

Smoke still runs actual extraction/training; it was NOT run during repository
synchronization. It uses the nested `results/part1/stageX/smoke` directories,
K=32 in A and the existing seed/fraction protocol. Main pipeline logs are retained
under `pipeline_logs/`; stop-on-failure behavior is preserved.

## Frozen Stage B test evaluation (no training or C selection)

```bash
python -m our_src.part1.stageb.stage_b --evaluate-test
python -m unittest our_src.part1.stageb.test_stage_b our_src.part1.stageb.test_frozen_test
```

Requires the existing Stage B model, run metadata, matching Stage A centroids,
original splits and dataset. Test artifacts use content-addressed
`results/part1/stageB/test_evaluations/<id>/`; histogram caches stay outside Git.
The generated notebook `results/part1/stageB/report.ipynb` displays persisted
metrics, LaTeX tables, PR curve, confusion matrix and automatically selected
TP/FN/FP/hard-negative images. A prior notebook is archived before replacement.
No test score influences training or hyperparameter selection.
