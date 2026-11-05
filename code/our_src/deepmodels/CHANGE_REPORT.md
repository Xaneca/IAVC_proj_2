Current repository integration: common.py resolves remote stagea modules with
legacy fallbacks; run.py derives its project path and checks results/part1 before
legacy result folders; the repository launcher imports our_src directly; tests
derive their location; README and the separate requirements file document clone
execution. No model computation or experimental configuration was changed.

> Historical implementation/validation record. For current clone instructions, use [README.md](README.md). Local smoke logs and the old development harness are archival evidence, not clone prerequisites.

# Change report

| File | Purpose |
|---|---|
| features.py | DINO mean/spatial pooling, input-driven grid and patch-CLS visualization; official bicubic preprocessing in our adapter; stronger raw/L2/ID/split cache validation; per-descriptor storage and shared-cost metadata |
| classification.py | Independent per-category validation C selection; reusable checksummed singleton SVM trials; controlled binary model/metric/prediction persistence; per-class F1 |
| fewshot.py | Exact-N binary nested subsets, fixed original background negatives, recorded IDs/counts and augmentation provenance, C/D shared trials, seed aggregation and learning/log-C plots; old union sampler retained as multilabel_subsets only |
| clip_experiments.py | Individual-template validation search and selected-template final evaluation; retain average-cosine ensemble; correct full-dataset retrieval labels and AP score description |
| visual_analysis.py | Confidence/diversity-driven case selection, image-ID/truth alignment checks, two category modes, persisted metadata and signed maps, actual-score consistency checks, crop-aligned optional patch visualization |
| run.py | Controlled A/B representation selection; six representations; C-to-D dependency and trial reuse; E4 cached CLIP SVM; F actual controlled model scores; richer protocol-separated comparisons and explicit missing cells |
| config.json | Three-value controlled grid and separate extended grid; explicit augmentations and optional visualization controls |
| test_integration.py | Focused exact-N/nesting, independent C, trial reuse, patch ordering/grid, numeric prompt aggregation, L2 and signed-occlusion checks; updated full runner/resume integration |
| README.md | Current status, dependency order, exact commands and artifact guide |
| AUDIT.md | Initial findings, reused code, scientific corrections, authoritative tree and historical evidence |
| VALIDATION.md | Actual commands/results and honest tested/executed/pending status |
| PROTOCOL.md (new) | Exact design, leakage boundaries, cache identities, interpretation and reporting schema |
| CHANGE_REPORT.md (new) | This file-level inventory |
| ../validate_part4_smoke.py (new) | Real pretrained 33-image execution, artifact checks and six-stage resume; no full benchmark |
| ../audit_evidence/ (new) | Initial/protected hashes, prior documents, rendered assignment pages, current validation logs |

Unmodified package files: __init__.py and common.py. Launcher, professor source,
original Parts 1/2, splits, old smoke harness and historical result folders remain
unchanged. No full-data benchmark, final test evaluation, commit or push is part
of this revision. See VALIDATION for installation status and measured evidence.
