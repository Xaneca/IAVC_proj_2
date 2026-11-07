# Part 5: image retrieval

Given a query (an image or a text), each method scores the 220 corpus images and returns a ranking of the
whole corpus. There is no classifier and no classification decision.

- **Stage A** (`stageA/`) builds the PASCAL VOC 2007 corpus and the queries.
- **Stages B and C** (`stageB_C/`) implement the retrieval methods and evaluate them.

## Dataset and ground truth

| File | Content |
|---|---|
| `stageA/corpus.csv` | 220 images: 60 `horse` (relevant); hard negatives `cow`, `dog`, `sheep`, `cat` (25 each); easy negatives `aeroplane`, `bus`, `bicycle` (20 each) |
| `stageA/queries.csv` | 10 horse query images, none of them in the corpus |
| `stageA/queries_text.csv` | 10 text queries for CLIP |

Relevance is category-level: for every query, the relevant images are the corpus rows with
`is_relevant == True`. The ground truth is read from the CSV. Image paths are resolved relative to the CSV's
directory, and Windows separators are accepted on any OS.

## Folder structure

```
our_src/part5/
  config.yaml          all experiments, parameters, paths, K lists, seeds and figure options
  run.py               entry point
  stageA/              dataset construction (not changed by Stages B and C)
  stageB_C/
    config.py          config loading and experiment expansion
    data.py            CSV loading and ground truth
    sift.py            dense SIFT wrapper and descriptor cache
    encoders.py        k-means, BoVW and VLAD (through src.encoders), PCA
    deep.py            ResNet-50, DINOv2 and CLIP
    normalization.py   L1, L2, signed square root, Hellinger, VLAD normalization
    retrieval.py       scoring and ranking
    metrics.py         AP, P@K, R@K, precision-recall curves
    timing.py          timer and storage accounting
    pipeline.py        experiment -> vectors, costs and metrics
    results.py         CSV outputs
    visualization.py   figures
    test_part5.py      sanity checks
```

Generated files go to `results/part5/`. Caches go to `cache/part5/`, which is gitignored and safe to delete.

## How to run

**Activate the environment first** (`conda activate IAVC_2`; it needs `cyvlfeat`, so Python 3.11, plus `torch` and
`clip`). Calling the environment's `python.exe` directly, without activating it, leaves `Library\bin` out of PATH.
The process then dies with exit code 127 and no Python traceback at the first BLAS call (Windows error
`0xc06d007f`).

From `code/`:

```powershell
python -m our_src.part5.run                        # all experiments
python -m our_src.part5.run --only bovw vlad       # only these groups
python -m our_src.part5.run --smoke                # small subset, written to results/part5/smoke
python -m our_src.part5.run --force                # recompute features instead of reusing the cache
python -m our_src.part5.run --figures-only         # redraw figures from the saved CSVs
python -m unittest our_src.part5.stageB_C.test_part5
```

`.\run_part_5.ps1 [-Only bovw,vlad] [-Smoke] [-Force] [-FiguresOnly]` does the same from PowerShell.

Partial runs replace only their own rows in the CSVs. The first run downloads the ResNet-50 (98 MB) and DINOv2
ViT-B/14 (330 MB) weights to the torch cache.

## Configuration and experiments

Each entry under `experiments:` in `config.yaml` is a group, and its key is the name `--only` accepts. A list under
`k` or `n_components` expands into one experiment per value, so adding an experiment only requires editing the
config. Cache files are keyed by a hash of the parameters that affect them, so changing a parameter recomputes only
what depends on it.

| Group | Experiments | Query type |
|---|---|---|
| `random` | fixed-seed random ranking | image |
| `bovw` | dense SIFT BoVW, k = 32, 64, 128, 256, 512, 1024 | image |
| `vlad` | dense SIFT VLAD, k = 16, 32, 64, 128 | image |
| `vlad_pca` | VLAD k = 32 + PCA, 8, 16, 32, 64, 128, 219 components | image |
| `cnn` | ResNet-50 (`IMAGENET1K_V2`), global-average-pooled, classifier removed, 2048-d | image |
| `dinov2` | DINOv2 ViT-B/14, CLS token, 768-d | image |
| `clip_text` | CLIP ViT-B/32 text encoder for queries, image encoder for the corpus, 512-d | text |
| `clip_image` | CLIP ViT-B/32 image encoder for queries and corpus, 512-d | image |

All methods share the same corpus, queries and ground truth. Vocabularies and PCA are fitted on the corpus only.

- **Dense SIFT** uses `standarizeImage` and `vl_phow` from `src/features.py`, with the defaults of
  `computeFeatures`: step 4, bin sizes 4/6/8/10, images taller than 480 px resized. It yields about 40 000 descriptors
  per image. They are computed once and reused by every BoVW and VLAD variant. Low-contrast descriptors are set to
  zero by `vl_phow` and are kept.
- **k-means** is seeded `MiniBatchKMeans`, fitted on up to 200 000 descriptors sampled evenly across corpus images
  (the same sample for every k). BoVW and VLAD with the same k share a vocabulary.
- **Encoding** uses `BoVWEncoder` and `VLADEncoder` from `src/encoders.py`, with hard assignment and no spatial tiling.
- **ResNet-50 and DINOv2** resize the whole image to 224 x 224 (bicubic, no center crop) with ImageNet mean and std.
  This changes the aspect ratio but keeps the whole object in view.
- **CLIP** uses its own `preprocess`, which does center-crop. It loads through `src.deep_models.getModel`.

## Normalizations

These are the only ones applied. There is no RootSIFT, tf-idf, VLAD intra-normalization or PCA whitening.

| Method | Normalization | Similarity |
|---|---|---|
| Dense SIFT | none beyond what SIFT already does | n/a |
| BoVW | L1, then element-wise square root (Hellinger) | cosine (dot product: the vector has unit L2 norm) |
| VLAD | signed square root, then global L2 | cosine |
| VLAD + PCA | VLAD normalization, PCA (fitted on corpus), L2 again | cosine |
| CNN / DINOv2 / CLIP | L2 on the embedding (image and text) | cosine |

`src`'s `VLADEncoder` already L2-normalizes its output. This only rescales the vector, so the final result is the
same as normalizing the raw residuals (`test_part5.py` checks this).

The ranking sorts the corpus by similarity, descending. Ties keep corpus order.

## Metrics

Per query:

- **AP**: non-interpolated, over the full ranking, normalized by the total number of relevant images.
- **P@K**: relevant images in the top K, divided by K (K = 5, 10, 20).
- **R@K**: relevant images in the top K, divided by all relevant images (K = 5, 10, 20, 60).

Per experiment: **mAP**, the **sample standard deviation of AP** across queries (`ddof=1`), and mean P@K and R@K.

Costs:

- **Extraction**: total and per corpus image. The shared dense SIFT time (`sift_*`) is separate from the encoding time
  (`encoding_*`). Encoding includes normalization and, for VLAD + PCA, the projection.
- **Fitting**: `vocabulary_fit_s` and `pca_fit_s`, reported separately.
- **Storage**: `corpus_bytes` = n_images x dim x 4 (float32). `vocabulary_bytes` and `pca_bytes` (components and mean)
  are auxiliary. `model_weights_bytes` is the model's state dict as loaded.
- **Query time**, average per query: `query_extraction_s` (a) is one query encoded alone, including its dense SIFT;
  `query_ranking_s` (b) is similarity plus sorting, averaged over `timing.ranking_repeats` runs; `query_total_s` is
  (a) + (b).

Times use `time.perf_counter()`, with `torch.cuda.synchronize()` around GPU timers. They are stored with the caches,
so a run reusing the cache reports the times of the run that computed the features.

## Outputs (`results/part5/`)

| File | Content |
|---|---|
| `experiments_summary.csv` | one row per experiment: parameters, metrics, dimensions, times, storage, device |
| `per_query_results.csv` | one row per (experiment, query): AP, P@K, R@K; text queries include the text and type |
| `rankings/<experiment>.csv` | full ranking per query: rank, corpus id, score, relevance, label |
| `run.log`, `run.json` | log, and the configuration and library versions of the last run |
| `figures/retrieval/<experiment>.png` | best and worst query (by AP) with their top-10 images: green ✓ relevant, red ✗ irrelevant, VOC label below |
| `figures/map_vs_k.png` | mAP vs vocabulary size for BoVW and VLAD, side by side |
| `figures/map_vs_pca.png` | mAP vs PCA components for VLAD k = 32, with the no-PCA result as a reference line |
| `figures/text_prompts_<experiment>.png` | AP per text prompt |
| `figures/ap_heatmap.png` | AP per query and method (image queries): sensitivity to query choice |
| `figures/pr_curves.png` | mean interpolated precision-recall curve of the best variant of each group |

Each prompt in `queries_text.csv` has its own `type`, so the prompt chart labels each bar with the type and colors
it by the families in `figures.text_type_families`.

## Notes and limitations

- **Random mAP sits above the share of relevant images.** 60/220 = 0.273, but the expected AP of a random ranking
  here is about 0.289, because AP favours the first relevant hits. Compare methods with the measured `random` row.
- **PCA is capped at n_corpus - 1 = 219 components**: 220 centered vectors have rank at most 219. Larger requests are
  clamped automatically.
- **PCA with 219 components is not the same as no PCA.** It preserves distances between corpus vectors but centers
  them, which changes cosines, and it drops the part of each query outside the corpus's span.
- **CLIP text mAP is not directly comparable with the image-query methods**, because it averages over different
  queries, some of them deliberately vague. `clip_image` exists for the fair comparison.
- **Relevance is at category level and the queries are all horses**, so the results measure horse retrieval only. Many
  images contain other objects (47 of the 60 relevant images, and 43 of the 160 others, also contain a person).
- **With 10 queries, AP standard deviations are large**; treat small mAP differences as noise.
- **Timings come from one machine and one run.** CNN, DINOv2 and CLIP ran on the GPU, while dense SIFT and encoding
  ran on one CPU core, so their times are not a like-for-like efficiency comparison. On GPU, CLIP loads in half
  precision, so its weights take half the memory they would on CPU.
- **Model loading is not counted in query time**, because a retrieval system keeps the model loaded.
