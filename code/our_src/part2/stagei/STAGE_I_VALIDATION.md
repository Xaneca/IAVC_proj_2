> Historical validation before the Part 2 directory integration. Paths/results below describe that earlier run. For current execution use `../README.md` and current `results/part2/` manifests.

# Stage I validation - 2026-11-06

## Implemented and protected

Stage I Fisher remains E3/E6 of the single Stage H pipeline. The supplied
src.encoders.FisherVectorEncoder and skimage.feature.fisher_vector are used
unchanged. All seven professor Python sources match the pre-change SHA256 hashes.
No custom gradients/responsibilities/aggregation or component removal exists.
K=[64,128], C=[0.1,1,10], seed=42, original historical splits, five categories,
400-descriptor training cap and existing GMM/SVM numerical settings are unchanged.
The API check confirms no direct 2KD option; full K(2D+1) output is retained.

Modified: models.py (cache/provenance integration), stage_h.py (safe outputs,
completion checks, reports and resource preflight), test_stage_h.py, README.md.
Added: provenance.py (wrapper around provided Dense SIFT caching and identities),
reporting.py (tables/mAP/timings/storage), STAGE_I.md and this report.
Configuration, encoders.py and all professor source are unchanged.

## Executed tests

From code/, using the existing Python 3.11 Part 1 environment:

```powershell
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
./.venv-stage-a/Scripts/python.exe -B -m unittest our_src.part1.stageh.test_stage_h -v
```

Result: 7 tests passed in 0.861 seconds. These verify supplied-encoder/API use,
finite 2KD+K dimensions, signed power + L2 and zero descriptors, immutable learned
GMM parameters during later encoding, deterministic actual library GMM fits on
synthetic training descriptors, exclusion of validation/test descriptors,
compatible GMM reuse, Dense SIFT cache reuse/content invalidation/corruption
rejection using real pixels, and consolidated metric/storage reporting.
Synthetic descriptors in unit checks are not scientific performance evidence.

## Real-image smoke

```powershell
./.venv-stage-a/Scripts/python.exe -B -m our_src.part1.stageh.stage_h --smoke-per-class 2 --output-dir results/part1/stageH/stage_i_20261106_cold
```

Completed with 33 original images: 11 train, 11 val, 11 test; 30 independently
selected target/background SVMs and 90 C trials. Fisher E3/E6 are included without
reducing their K values. Run ID: b28c2742c183ab5f5468.

Results: results/part1/stageH/stage_i_20261106_cold/smoke/.
Read STAGE_I_REPORT.md, comparison.csv/json, metrics.csv, validation_trials.csv,
manifest.json, run.json, models/, encoders/, curves/ and rankings/ there.

The source hashes in the manifest match the current implementation. All 90
rankings were checked against reported AP, and binary image/label memberships
match across all six encoders. Attempts to reuse the legacy historical output
and a changed-seed run were rejected without changing their run manifests. All 401
completion artifact checksums were verified. Model arrays have the expected
finite dimensions; mAP recomputed from per-category AP matches the consolidated
table; C matches validation-only selection. Stored GMM training IDs belong to
training only, covariances are positive, priors sum to one and convergence is true.
Repeating the same command reports reuse of the completed run without extraction
or fits. No persisted result from the old smoke was overwritten.

SIFT extraction for the 33-image encoding pass: 10.432378 seconds, shared across
six representations. Encoder-preparation extraction is separately identified.
All 33 accesses in this cold run were actual extractions; no cache hits.
The table includes AP per category, val/test mAP, fit/encoding/inference timing and
float32 feature bytes. Feature matrices are temporary and removed after scoring.
A separate successful cached smoke is retained at
results/part1/stageH/stage_i_20261106_final/smoke/ (run 4eef05c7f665a36414bd).
That run correctly reports cache-read time separately and has 33 cache hits.
The optional cache created for this task (177272565 bytes) was removed after
validation to recover disk space; all model/metric/provenance artifacts remain.
One initial development smoke was interrupted to refine timing instrumentation;
it is marked interrupted and is not counted as a passed run.

## Full benchmark: PENDING, not started

```powershell
./.venv-stage-a/Scripts/python.exe -B -m our_src.part1.stageh.stage_h --preflight --output-dir results/part1/stageH/stage_i_20261106_cold
```

The final preflight returned feasible=false. It accounts for the full 1978
training images and at most 791200 descriptors. Available RAM: 2215268352 bytes;
estimated RAM with EM workspaces/headroom: 4719591424 bytes.
Free disk: 2552983552 bytes; temporary feature matrices require
1466231808 bytes, before headroom. These are resource
estimates, not a failed full-data fit. No full-data extraction/fit was started,
and no smaller training sample or modified K/C grid was silently substituted.

On a machine with adequate resources, the existing entry point runs the full
benchmark and retains the same experimental design:

```bash
python -m our_src.part1.stageh.stage_h --output-dir results/part1/stageH/stage_i_full
```

A full new run is still required before making scientific conclusions about
Fisher vs VLAD accuracy or runtime. The one-seed design also limits stability
claims. Smoke test-set metrics are technical evidence, not final benchmark AP.
No commit or push was performed for this revision.
