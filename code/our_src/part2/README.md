# Part 2: Stages H and I

Run from `code/` on Linux or Windows with the project's SIFT environment:

```bash
python run_part_2.py --smoke --sift-cache results/part2/sift_cache
python run_part_2.py --preflight
python run_part_2.py --sift-cache results/part2/sift_cache
python -m unittest our_src.part2.shared.test_shared our_src.part1.test_run
```

H lives in `our_src/part2/stageh/` (BoVW/VLAD); I in `our_src/part2/stagei/` (Fisher).
Both delegate to `shared/`, preserving one extraction/training/evaluation pipeline.
Outputs are `results/part2/stageH/` and `results/part2/stageI/`; smoke outputs have a
`smoke/` suffix. Historical `results/part1/stageH/` outputs remain unchanged.

Run H before I. I verifies H's artifact checksums, code/dependency identity, image
content, splits, seeds and protocol settings before merging the comparison.
K=[64,128], linear SVM C=[0.1,1,10]. Train-only vocabulary/GMM fitting; validation
selects C, and test is only evaluated afterwards. All five categories use the
same splits. The shared encoder cache separates full/smoke runs and validates
training provenance. A conflicting output is rejected, not overwritten; use
`--output-dir` and `--baseline-dir` on the individual stage entry points for a new run.

`comparison.csv`, `comparison.json`, per-category metrics,
PR curves and rankings report AP/mAP, dimensions, extraction/model-fit/encoding/
classification timings and feature storage. Inspect timing/cache fields before
comparing costs. Smoke accuracy is only a technical check, not a scientific result.
Full runs are resource-gated by `preflight.json`; insufficient RAM/disk prevents fitting.

Fisher exclusively uses the supplied `src.encoders.FisherVectorEncoder` and
`skimage.feature.fisher_vector`; no Fisher algorithm is reimplemented and the
provided sources are unchanged. The supplied API retains weight statistics:
K(2D+1), not the assignment's 2KD. It offers no flag to omit those components.
For D=128: Fisher K64=16448, Fisher K128=32896; VLAD K128=16384.
Power (signed square-root) and L2 normalization are applied by the integration.
Fisher adds soft GMM assignments and second-order deviations to VLAD's first-order
residual information, plus weight statistics in this API. Costs include GMM fitting,
posterior evaluation, longer encoding and larger feature/classification storage.
See `stagei/STAGE_I.md` for methodological details.
