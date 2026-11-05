> Historical implementation/validation record. For current clone instructions, use [README.md](README.md). Local smoke logs and the old development harness are archival evidence, not clone prerequisites.

# Implementation audit - 2026-11-05

Before changes, inspected assignment Part 4 A-F (pages 9-11), professor
`src/deep_models.py`, `src/utils.py`, `src/visualization.py`, notebook
`2_deep_models.ipynb`, installed `our_src/part1`, portable Parts 1/2 utilities,
classical H encoding/training/results, and all existing deep-model documentation.
Assignment pages were extracted and rendered under `audit_evidence/`.

## Working tree and preserved evidence

Installed runtime/data: `<repository>/code`.
Portable source: `<former-development-workspace>`.
All 13 initial package files matched byte for byte. This workspace's launcher
selects the portable package, so it is the authoritative source for this revision.
Initial package hashes and protected professor/Part 1/split hashes are retained in
`audit_evidence/`. Previous README/AUDIT/VALIDATION are archived there too.
No professor or classical implementation is edited. No historical result is
replaced. Full benchmark execution is outside this audit.

## Findings and changes

| Stage | Existing | Problem | Implemented | Validation / limitations |
|---|---|---|---|---|
| A | Frozen ResNet-18; hooks; GAP; raw/L2 cache; SVM | Primary and supplementary searches chose a common C; controlled grid had five values | Independent C per category; controlled three-value grid; primary binary metrics/model/qualitative-score persistence; richer reporting | Synthetic integration plus real pretrained smoke recorded in VALIDATION; full AP pending |
| B | DINOv2 final CLS | No patch aggregation; professor resize bilinear differs from official bicubic evaluation | CLS 384, mean 384, TL/TR/BL/BR spatial 1536; actual input/patch-size grid checks; adapter-only preprocessing correction | Rectangular-grid ordering test, dimensions/norm/finite checks; real smoke; no intermediate Transformer layers |
| C | Nested multilabel anchor unions, training pixel flips | Co-occurrence could exceed N positives; negatives changed with N; representation selected using supplementary AP | Independent binary exact-N prefixes; all fixed training background negatives; validation-controlled representation selection; IDs/counts/seeds/augmentation recorded | Exact-N, deterministic/nested/train-only tests; real smoke uses N=1/all; N=5/10/20 full curves pending |
| D | Replot of full-data common-C searches | No connection to few-shot regimes; repeated fits not reused | Same C representation/subsets/seeds, no augmentation, all five C values; content-addressed singleton trial/model cache shared with C; AP/time/count/config records and log-C plots | Incremental reuse test verifies only new C values fit; full regularization interpretation pending |
| E | Official OpenAI CLIP; first-template and average-similarity ensemble | No individual-template selection or supervised CLIP baseline; retrieval figures inherited an incorrect background-only title | E1/E2 retained; E3 complete validation search and globally selected template; E4 cached image features plus controlled SVM; F1; explicit full-dataset retrieval labels | Tokenization boundary tests, numeric average-cosine check, no-test-features selection; real-model smoke; no claim ensembles always help |
| F | First-ID eligible cases and predicted-category signed occlusion | Arbitrary ID order; different categories across model maps; no patch visualization or complete case metadata | Within-model confidence ranks and category diversity; unique cases capped at 15; common recorded true category and predicted modes; baseline-score consistency checks; exact-set vs membership distinction; crop-aligned patch-CLS maps | Signed-map tests and real-model score/map checks; small smoke cannot guarantee every failure group; visual interpretation remains human work |

## Reused components and scientific protocol

Historical split validation: existing protocol.split_digest and immutable reference
membership. Dataset IDs/multilabel annotations: existing part1.data.manifest.
SVM: existing part1.svm.train_svm_grid -> professor src.utils.train_svm, which uses
LinearSVC(dual=True, loss='hinge'). No extra SVM normalization is applied.
AP/ranking: existing stagec.evaluation.evaluate and the same sklearn
non-interpolated AP primitive. CLIP retrieval plot labels are corrected locally;
the original binary evaluator remains untouched.

Primary protocol is target vs actual background, separately trained per category.
Supplementary full-dataset multilabel OvR remains under the representation root.
CLIP full-dataset classwise text retrieval is a third explicitly named protocol.
None are merged into an unlabeled ranking.

Frozen ResNet hooks layer2.1/layer3.1/layer4.1 are the second/final BasicBlocks of
the three selected stages, giving 128/256/512 channels. GAP averages H and W
independently. The professor's ImageNet CAM is not reused for task SVMs.
DINO final normalized CLS/patch tokens are aggregated before image-level L2.
Official transforms are associated with actual loaded weight hashes.

## Classical evidence

Classical H source fits each target/background model independently over
C=[0.1,1,10], using validation AP and no additional SVM normalization. Its Fisher
vector implementation includes mixture-weight, mean and variance components:
K*(2*128+1) = 16448/32896 for K64/K128. BoVW is 64/128 and VLAD 8192/16384.
Only `stageH/smoke` is currently complete; no full H manifest is present.
Consequently classical full-data cells remain null. The importer additionally
requires image-content provenance and compatible full-data configuration; it
never fabricates unavailable historical results.

## Historical smoke evidence

The prior 33-image real smoke is supported by
`<repository>/code/deepmodels/smoke_validation/REAL_SMOKE_REPORT.md`, its
JSON report and the old run `2c60e4a5fd407b9d8509`. It used official pretrained
weights and passed the old A-B-D-C-E-F workflow. It does not validate new patch
pooling, exact-N binary few-shot, independent C or E3/E4. The old contradictory
missing-dependency notes in documentation have been superseded, not treated as
current evidence. Current validation is separately documented.
