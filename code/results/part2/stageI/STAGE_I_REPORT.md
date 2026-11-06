# Part 2 — I shared comparison

FULL DATASET RUN

| Encoder | K | Dimension | Val mAP | Test mAP | Encoding s/image | Feature bytes/image |
|---|---:|---:|---:|---:|---:|---:|
| bovw | 64 | 64 | 0.589813 | 0.518569 | 0.022694 | 256 |
| vlad | 64 | 8192 | 0.765285 | 0.737840 | 0.032430 | 32768 |
| fv | 64 | 16448 | 0.812780 | 0.752705 | 0.197119 | 65792 |
| bovw | 128 | 128 | 0.606242 | 0.534266 | 0.030639 | 512 |
| vlad | 128 | 16384 | 0.785491 | 0.760049 | 0.047420 | 65536 |
| fv | 128 | 32896 | 0.809110 | 0.761580 | 0.344436 | 131584 |

Per-category validation/test AP, selected C and all timing/storage fields are in comparison.csv.

VLAD K128 has 16384 components; Fisher K64 has 16448: +64 (+0.390625%).
This is the supplied library variant, not exactly equal-dimensional 2KD Fisher.

Compared with hard-assigned first-order VLAD residuals, Fisher uses soft GMM assignments and encodes first/second-order deviations; this implementation also retains mixture-weight statistics. Additional costs include diagonal-GMM estimation, posterior computation for each descriptor/component, and storing/classifying the larger vector. Similar final dimensions do not imply equal encoding costs.

SIFT extraction is shared across methods. Encoder-fit times and hardware provenance must be inspected before comparing timings. A smoke run is not scientific evidence of full-data accuracy.
