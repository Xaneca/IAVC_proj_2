# Project entry points

From the Git root, enter `code/`. Linux and Windows use the same Python commands.

- Part 1 A-G: `python run_part_1.py --dry-run`, then `python run_part_1.py`.
- Part 2 H/I: `python run_part_2.py --smoke`; see `our_src/part2/README.md`.
- Part 4 A-F: `python run_deepmodels.py --help`; see `our_src/part4/README.md`.

Part 1 sources/results use `our_src/part1/stagea` through `stageg` and
`results/part1/stageA` through `stageG`. Part 2 uses `our_src/part2/stageh`,
`stagei`, and shared helpers; outputs `results/part2/stageH`, `stageI`.
Historical outputs remain in their original locations.

Frozen Stage B test evaluation (no training or C selection):

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
