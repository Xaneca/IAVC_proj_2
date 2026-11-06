# Part 3: large visual vocabularies + PCA vs small vocabularies

Does a richer representation (large vocabulary, then PCA) beat a small vocabulary used directly?
The three representations have very different dimensions, so every row records the representation
and its exact dimension:

| representation | encoding dimension (D = 128 SIFT dims) |
|---|---|
| BoVW   | K |
| VLAD   | K * 128 |
| Fisher | 2 * K * 128 (diagonal GMM, mean + variance gradients) |

## How to run

From `code/`, with the project environment activated (`conda activate IAVC_2`):

```powershell
python -m unittest our_src.part3.test_part3                                    # sanity tests
python -m our_src.part3.run --config our_src/part3/config.yaml --smoke         # smoke test
python -m our_src.part3.run --config our_src/part3/config.yaml                 # full run
python -m our_src.part3.run --config our_src/part3/config.yaml --seeds 0 1     # some seeds only
python -m our_src.part3.run --config our_src/part3/config.yaml --force         # recompute existing rows
python -m our_src.part3.run --config our_src/part3/config.yaml --clean-cache   # delete cache/part3
```

Or use the `code/run_part_3.ps1` wrapper, which can be called from any directory (shown here from
the repository root, with the environment activated):

```powershell
.\code\run_part_3.ps1                # full run
.\code\run_part_3.ps1 -Smoke         # smoke test
.\code\run_part_3.ps1 -Seeds 0,1     # some seeds only
.\code\run_part_3.ps1 -Force         # recompute existing rows
.\code\run_part_3.ps1 -CleanCache    # delete cache/part3
```

- **Resumable:** rows already in `metrics.csv` are skipped unless `--force` is passed. Codebooks
  and encodings are cached per (representation, K, seed), so an interrupted run does not refit or
  re-encode what was already finished.
- **`--smoke`** uses about 24 train and 16 test images, one seed, tiny K values and 2-fold CV.
  It writes to `results/part3/smoke/`. One smoke PCA setting (VLAD K=4 -> 64) is deliberately
  invalid, to check that skipping works.

## Pipeline, for each (representation, K, seed)

1. **Dense SIFT:** `src.features.computeFeatures`, extracted once per image and cached in
   `cache/part3/sift/` (compressed uint8, about 17 GB for the 4946 images). Extraction times go
   into `_timings.json`.
2. **Codebook:** a seeded subsample of `codebook.n_descriptors` descriptors from TRAIN images,
   shared by all codebooks of that seed.
   - BoVW and VLAD use MiniBatchKMeans (or KMeans).
   - Fisher uses a diagonal GaussianMixture.

   A codebook depends only on (representation, K, seed). The no-PCA row and all PCA rows built on
   it share it, and each of those rows reports its `time_codebook`.
3. **Encoding:** the unmodified `src.encoders.BoVWEncoder`, `VLADEncoder` and
   `FisherVectorEncoder` produce one global vector per image (no spatial tiling).
   - VLAD from `src` is already L2-normalized; with sqrt -> L2 afterwards this gives the same
     result as normalizing the raw vector.
   - skimage's Fisher vector prepends K mixture-weight gradients. These are dropped by default
     (`fisher_include_weight_grads: false`).
   - All encodings of a seed are computed in one parallel pass over the images.
4. **`normalization.post_encoding`:** default `[sqrt, l2]` (signed power with alpha 0.5, then L2).
5. **PCA** (PCA rows only): fitted on TRAIN encodings only, `svd_solver=full` by default (a
   `random_state` is passed if randomized is used), whitening configurable. Before any computation
   a PCA row is skipped with a warning unless `pca_dim < original_dim` and
   `pca_dim <= n_train - 1`.
6. **`normalization.post_pca`:** default `[l2]`.
7. **SVM** for each class in `target_classes`. For each class:
   - Positives are images containing the class. Negatives are all other images
     (`negatives: all`), or only background images (`negatives: background`, the Part 1 protocol).
   - `C` (and `gamma` for RBF) is chosen by `StratifiedKFold(cv_folds, shuffle=True,
     random_state=seed)` on TRAIN only, scored with `cv_scoring` (default `average_precision`).
   - The model is then refit on all of TRAIN.
   - Linear kernel = `LinearSVC`, RBF = `SVC`.
8. **Evaluation** on TEST.

## Output files (`results/part3/`)

| file | content |
|---|---|
| `metrics.csv` | one row per (experiment, seed) |
| `metrics_summary.csv` | mean and sample std (ddof=1) over seeds, grouped by (representation, pca, k, pca_dim); `n_seeds` |
| `skipped_experiments.csv` | PCA settings that violate the constraints, with the reason |
| `pca_variance/` | cumulative explained-variance `.csv` + `.png` (BoVW K=512 and K=1024, first seed); `summary.csv` with d80/d90/d95 and the variance kept at each PCA dim used |
| `run.log` | log, including warnings (skipped experiments, convergence warnings, reduced CV folds) |
| `run.json` | resolved config, library versions, state |

## `metrics.csv` columns

| column | meaning |
|---|---|
| `representation`, `pca`, `k`, `pca_dim`, `seed` | experiment key; `pca_dim = 0` when `pca` is false (same convention as `C0` in `src`) |
| `original_dim` | encoding dimension before PCA (K, K*128 or 2*K*128) |
| `final_dim` | dimension fed to the SVM |
| `compaction_pct` | `(1 - final_dim / original_dim) * 100`; 0 without PCA |
| `n_train`, `n_test` | encoded TRAIN / TEST images |
| `test_accuracy`, `macro_f1` | mean over target classes of accuracy and macro F1 (F1 of positive and negative class averaged) at the SVM threshold 0 |
| `map` | mean over target classes of the AP; with a single target class, `map == ap_<class>` |
| `ap_<c>`, `accuracy_<c>`, `macro_f1_<c>` | per-class test metrics; AP uses `decision_function` scores (`average_precision_score`) |
| `n_train_<c>`, `n_train_pos_<c>`, `n_test_<c>`, `n_test_pos_<c>` | images and positives of each binary problem |
| `cv_score_<c>`, `best_params_<c>` | best mean CV score and the chosen hyperparameters |
| `time_sift_extraction` | total SIFT extraction time over all images (shared by every row, excluded from inference) |
| `time_sift_extraction_per_image` | mean per image |
| `time_descriptor_sampling` | drawing the codebook descriptor sample (shared by all codebooks of the seed) |
| `time_codebook` | fit time of the k-means / GMM this row depends on (shared with the rows that use the same codebook) |
| `time_encoding`, `time_encoding_per_image` | encoding time summed over TRAIN+TEST, and the mean per image |
| `time_pca` | PCA fit time (0 without PCA) |
| `time_svm_cv`, `time_svm_fit`, `time_svm_train` | CV search, final fit, and their sum (summed over target classes) |
| `time_inference_per_image` | mean over test images, one image at a time: encoding of that image + normalization + PCA transform + SVM predict (all target classes). SIFT excluded |
| `feature_storage_bytes_per_image` | `final_dim * 4` (float32) |
| `feature_storage_mb_total` | bytes per image * (n_train + n_test) / 1e6 |
| other | `experiment`, `negatives`, `svm_kernel`, `post_encoding`, `post_pca`, `codebook_n_descriptors`, `svm_convergence_warnings`, `n_inference_images` |

All times are wall-clock seconds measured with `time.perf_counter`.

## Seeds

`seeds: [0, 1, 2]`. The seed controls:
- the descriptor subsample for the codebook,
- k-means / GMM initialisation,
- the CV folds and LinearSVC,
- PCA `random_state` (only used by the randomized solver),
- the optional per-image descriptor cap.

## Limitations

- **The train/test split does not depend on the seed.** It is the project's split (the
  `data/splits_internal` lists validated against Part 1's reference). TRAIN = `train + val` by
  default (`data.train_splits`), TEST = `test` (the supplied `val`). The std over seeds therefore
  measures codebook / CV variability only, not variation across data splits.
- **Results are not directly comparable to Part 1**, which used motorbike vs background only.
  With `negatives: all` (the default) every other image is a negative. Set
  `negatives: background` to follow the Part 1 protocol.
- **Encoding times include some CPU contention.** They are measured per image and per encoder
  inside parallel workers, so they are slightly higher than single-process times. All
  representations are measured under the same conditions. Loading the cached descriptors is not
  counted as encoding time.
- **Inference time reuses the encoding time.** Each image's encoding time comes from the encoding
  pass, where images are already encoded one at a time. Normalization, PCA and predict are then
  timed one image at a time.
- **Two different PCAs.** The PCA here compresses the final image vector. It is not the
  descriptor-level PCA of `src.train` (`C80`).
- **Cache size.** About 17 GB of SIFT plus about 1 GB of encodings in `cache/part3/`
  (gitignored). Remove it with `--clean-cache`.
