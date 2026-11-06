# H/I integration validation — 2026-11-06

18 tests passed: Stage B existing/new contracts, shared H/I integration and Part 1 orchestration.
All provided src Python files retain their recorded SHA256 checksums.
H/I smoke completed with 33 real images, seed42, unchanged K64/128 and C0.1/1/10.
20 H models +10 I models; 90 candidate C fits. This is technical validation only.

| Encoder | K | Dimension | Val mAP | Test mAP | Encode s/image | Bytes/image |
|---|---:|---:|---:|---:|---:|---:|
| bovw | 64 | 64 | 0.820833 | 0.711111 | 0.063168 | 256 |
| vlad | 64 | 8192 | 0.655000 | 0.677778 | 0.088302 | 32768 |
| fv | 64 | 16448 | 0.877500 | 0.727778 | 0.724175 | 65792 |
| bovw | 128 | 128 | 0.770833 | 0.711111 | 0.090264 | 512 |
| vlad | 128 | 16384 | 0.746667 | 0.761111 | 0.134180 | 65536 |
| fv | 128 | 32896 | 0.877500 | 0.727778 | 1.200834 | 131584 |

Artifacts verified by SHA256: H267, I139; manifests record image/source/config/dependency identities.
Outputs: results/part2/stageH/smoke and stageI/smoke. Per-category AP, fitting/extraction/encoding/classification time and storage are in comparison.csv.

Full benchmark was NOT run: both resource preflights rejected it.
- H: available RAM 685690880 B, estimated required 2694119424 B; disk free 639021056 B, feature matrices 490010112 B (additional headroom required).
- I: available RAM 831975424 B, estimated required 4719591424 B; disk free 638173184 B, feature matrices 976221696 B (additional headroom required).

No numerical configurations were reduced to fit this PC. Historical results were preserved.
For reproduction use ../README.md. Full-data recognition conclusions remain pending sufficient resources.
