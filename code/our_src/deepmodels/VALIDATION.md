> Historical implementation/validation record. For current clone instructions, use [README.md](README.md). Local smoke logs and the old development harness are archival evidence, not clone prerequisites.

# Validation report - controlled protocol v2, 2026-11-05

## IMPLEMENTED

See AUDIT, PROTOCOL and CHANGE_REPORT. The portable package is the source for
this revision. Professor code, original Parts 1/2 and historical results are
preserved. No full 4946-image benchmark or final test evaluation was requested
or executed.

## TESTED

Python used:
`python`.
Working directory:
`<former-development-workspace>`.

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& 'python' -m unittest deepmodels_setup.test_integration test_pipeline stage_c_setup.test_evaluation -v
```

**18 tests passed in 229.467 seconds.** Full log:
`audit_evidence/tests.log`.

Coverage: independent class C selection; deliberately different winning Cs;
validation-only selection with test features absent; real professor/classical SVM
integration; correct multilabel metrics; exact-N positives, deterministic/nested
subsets and fixed actual background negatives; historical splits and mutation
rejection; real pixel flipping and C-independent cache reuse; rectangular DINO
pooling order/grid checks; L2 zero safety; signed occlusion; CLIP average-cosine
numeric check and individual-template selection; complete A-B-C-D-E-F and
checksum-validated resume with no repeated feature extraction; classical runner
and AP/ranking regression. The complete orchestration test uses explicit backbone
doubles and actual images/SVMs; it is not evidence of pretrained performance.

A subsequent metadata-only correction changed the selected CLIP prompt provenance
from inherited prior-knowledge wording to validation selection. Re-ran:

```powershell
& 'python' -m unittest deepmodels_setup.test_integration.IntegrationTests.test_clip_mean_similarity_and_persistence_contract -v
```

**1 test passed in 19.847 seconds.** Log:
`audit_evidence/clip_metadata_regression.log`.

## Verified historical real smoke

The earlier 2026-11-05 33-image smoke used official ResNet/DINO/CLIP checkpoints.
Verified its six-stage complete state and **1460 stage-artifact checksums** in run
`<repository>/code/deepmodels/smoke/2c60e4a5fd407b9d8509`.
Evidence: `audit_evidence/previous_smoke_verified.json` and the original
`deepmodels/smoke_validation/REAL_SMOKE_REPORT.md`.
This supports the old pipeline only; it is not substituted for new feature or
protocol validation. Contradictory old missing-dependency notes are historical.

## Current real pretrained smoke

Command:

```powershell
& 'python' ./validate_part4_smoke.py
```

Current execution status is appended after completion below. This harness uses
33 original images (11 train, 11 val, 11 test), two CPU threads, official pretrained
checkpoints, N=[1,all], five seeds, the full three/five-C grids, both C augmentations
and a 2x2 occlusion grid. Test embeddings may be cached but no test scores are
computed. It checks raw/L2 dimensions and finite values, frozen weights and
preprocessing, exact-N memberships, grids, both explanation modes and baseline
scores, patch-CLS maps, feature checksums and six-stage resume.

One development test invocation was interrupted while source changes were still
being made. A development real smoke was also stopped after A/B to correct the
selected-prompt metadata wording; its caches were retained and reused. Neither
interrupted invocation is counted as a passed run. Logs ending in
`development_interrupted.log` are retained for provenance.

## PENDING

- Full A-F benchmark with N=5/10/20 and complete original training pools.
- Final test evaluation after freezing all choices; no benchmark AP claimed.
- Compatible full-data classical artifacts with required provenance.
- Human inspection of object/context evidence, difficult visual conditions and
  possible under/overfitting; smoke results cannot establish scientific trends.
- Installation and final preservation checks are recorded below after completion.

## Current execution result: PASSED

Official pretrained ResNet-18, DINOv2 ViT-S/14 and OpenAI CLIP ViT-B/32 passed
all six stages and the harness checks. Run: `b150e10207b2755c6bee` under
`deepmodels_output/audit_v2/smoke/`. Technical smoke only, 33 images, no final test
scores. Harness elapsed time: 368.811 seconds (includes model loading,
verification and resume; A/B embeddings from the interrupted development run
were reused). Evidence: `deepmodels_output/audit_v2/VALIDATED_SMOKE.json` and
`audit_evidence/real_smoke_v2.log`.

Real dimensions verified: ResNet 128/256/512, DINO 384/384/1536, CLIP 512.
Verified 100 few-shot category/seed/N/augmentation conditions and 50 D conditions.
D reused all 150 compatible C trials; only 100 extra fits for C=100/1000 were
needed. Shared cache has 400 unique singleton trials. Exact subset IDs and first
three trial records match between C and D. Evidence:
`audit_evidence/trial_reuse_verified.json`.

Six real cases produced 18 panels (two occlusion modes plus patch-CLS each),
with signed finite arrays and baseline-score checks. Inspected a common-category
panel and the learning curve for legibility; no scientific localization conclusion
is claimed. The full run can select up to 15 unique examples if available.

Full-data dry-run (read/hash/validate only):

```powershell
& 'python' ./run_deepmodels.py --project <repository>/code --output ./deepmodels_output/controlled_v2 --dry-run
```

Passed: **4946 images**; historical split SHA256
`b1872fd2e39d96d76bdaa4a3feb9ce26f4efb698dc62a181fa4d558ab4ec0399`.
No full-data feature extraction or SVM fit was started. Log:
`audit_evidence/full_dry_run.log`.

All **65 protected source/split files** have unchanged SHA256 hashes; evidence
`audit_evidence/protected_verified.json`. Original deep-model package was backed
up to `audit_evidence/original_deepmodels/` before installed synchronization.

## Installed package

Synchronized all 15 top-level source/config/documentation files into
`<repository>/code/our_src/deepmodels/`, after preserving the previous
13-file package under `audit_evidence/original_deepmodels/`. SHA256 equality of
portable and installed files was checked. Only that package was written.

Installed import/CLI dry-run passed from the original code directory:

```powershell
& 'python' -B -m our_src.deepmodels.run --project <repository>/code --output <former-development-workspace>/deepmodels_output/audit_v2 --smoke 2 --dry-run
```

Result: 33 images, A-B-C-D-E-F, no final test evaluation, and the same
`b150e10207b2755c6bee` experiment identity as the validated portable run.
Rechecked all 65 protected files after installation: unchanged.
The file-by-file inventory is `audit_evidence/change_inventory.json`.
