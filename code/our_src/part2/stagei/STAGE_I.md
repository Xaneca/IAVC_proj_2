# Stage I — Fisher Vector

Stage I of Part 2 is implemented as E3 (K64) and E6 (K128) in the existing Stage H
comparison engine. No parallel training/encoding pipeline is introduced. Run it
through `our_src.part2.stagei.stage_i` after Stage H; see `../README.md`.

## Mandatory supplied implementation

The only Fisher encoder is `src.encoders.FisherVectorEncoder`, which delegates to
`skimage.feature.fisher_vector`. Its saved learned means, diagonal covariances and
mixture weights are injected into the supplied wrapper, including the precision
needed by that wrapper. No gradients, responsibilities, Fisher aggregation or
alternative Fisher algorithm have been implemented in our integration code.
`src/encoders.py` and the remaining professor sources are unchanged.

Verified API in installed scikit-image 0.26.0:
`fisher_vector(descriptors, gmm, *, improved=False, alpha=0.5)`.
The API offers no switch for excluding the mixture-weight block or producing
2KD directly. The official documentation explicitly describes the output as
2KD+K. Source: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.fisher_vector

We keep the complete supplied output, without dropping/slicing any components.
Its dimension is K(2D+1): 16448/32896 at D=128, K=64/128. The assignment's 2KD
would be 16384/32768. This documented library variant is the explicit deviation;
if strictly 2KD is demanded, instructor approval or an explicitly justified
library-supported alternative is needed. The algorithm is not silently replaced.

The supplied encoder returns its raw library vector (`improved=False`). Our
existing adapter applies signed square-root followed by safe global L2 exactly
once. Empty images produce a fixed zero vector. There is no PCA or extra SVM
normalization. Descriptors and classifiers are the same as for BoVW/VLAD.

## Controlled experiment

K=[64,128], C=[0.1,1,10], seed=42. Five independent target/background classifiers:
motorbike, aeroplane, horse, car, person. Historical train/val/test memberships
are checked unchanged; co-occurring labels are retained. The GMM sample uses
training images only, deterministic ordering and at most 400 SIFT/image, shared
with the K-means training protocol. Diagonal GMM covariance uses max_iter=200,
tol=.001, reg_covar=1e-6, n_init=1 and explicit random_state=42. Failed convergence
raises an error; validation/test encoding does not fit or update it.

Each category/encoder chooses C only by validation AP; first winning C resolves
ties. All model/C fits finish before test scoring. AP is sklearn non-interpolated
AP on continuous SVM margins; target mAP is the unweighted mean over five classes.
Classification threshold is zero. Classwise retrieval ranks the same binary
membership by the corresponding SVM score, with PR/AP and image IDs persisted.
BoVW/VLAD/Fisher share their splits, categories, descriptors and SVM settings.

## Reproducibility and outputs

The runner saves source SHA256, image SHA256, versions, input IDs/labels/splits,
configuration and hardware/thread settings in manifest.json. GMM identities also
include sampling, library versions, supplied encoder hash and convergence data.
Existing unknown/different-provenance results are rejected; use a new output
folder. Matching complete runs are reused only after all artifact checksums pass.

Optional `--sift-cache` wraps the existing `src.features.extractFeatures` cache
API; keys include extractor source, versions and image content. Checksums reject
corruption. It is not another SIFT implementation. Cache reads and extraction
are timed separately; cache-miss extraction includes the provided cache write.
Encoder-preparation SIFT costs are also identified separately. Timings from cache
reuse cannot be presented as uncached extraction performance.

comparison.csv/json contain six rows: encoder, K, dimension, category APs, val/test
mAP, selected C per category, SIFT access/extraction/cache-read times, K-means/GMM
fit time, encoding/image, selected/search SVM fit time, classification inference
and float32 feature storage/image/all images. Storage excludes .npy headers;
feature matrices are temporary and removed after evaluation. Raw SIFT is cached
only if explicitly requested. Repeated shared timings must not be summed across
rows. Reused historical K-means timings require hardware/run caveats.

The generated STAGE_I_REPORT.md explains Fisher separately and compares
VLAD K128 (16384 floats, 65536 bytes/image) with Fisher K64 (16448 floats,
65792 bytes/image). Fisher has 64 more components and 256 more bytes (+0.390625%).
Their dimensions are close, not identical.

## Commands (from code/, intended Part 1 environment active)

The recorded smoke used OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2 and
MKL_NUM_THREADS=2. Set these in the shell before reproducing that run (Linux:
`export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`; PowerShell
assignments are shown in STAGE_I_VALIDATION.md). Changed thread/hardware settings
require a new output directory because provenance is intentionally checked.

```bash
python -m unittest our_src.part2.shared.test_shared -v
python run_part_2.py --smoke --sift-cache results/part2/sift_cache
python run_part_2.py --preflight
# Full benchmark only if resources permit:
python run_part_2.py
```

Use a new --output-dir for a changed source/configuration. No historical output
folder is overwritten. The preflight accounts conservatively for the float64 GMM
sample, EM workspaces, input chunks, temporary feature matrices and headroom.
Dense SIFT caches may be much larger than the feature matrices; leave the full
benchmark cache disabled when storage is constrained. No K/C/sample-cap reduction
is silently made to get a benchmark to fit.

## Assignment question

Compared with hard-assigned first-order VLAD residuals, Fisher encodes soft-assigned
first- and second-order deviations from a learned probabilistic descriptor model.
The supplied variant also retains mixture-weight statistics. Its additional costs
are GMM estimation, descriptor/component posterior computation, higher-order
aggregation, and usually larger vectors to store and classify. Similar final
VLAD128/Fisher64 dimensions do not imply equal extraction/encoding costs or
recognition quality. Those comparisons require measured compatible results.

## Validation and limitations

See STAGE_I_VALIDATION.md for executed tests, smoke evidence, measured results and
resource-preflight outcome. Unit checks use both actual library calls and clearly
identified synthetic training descriptors; scientific performance is supported
only by real-image experiments. A smoke run is technical validation, not a full
benchmark. One fixed seed limits stability conclusions; no performance trend is
asserted without measured full-data evidence.
