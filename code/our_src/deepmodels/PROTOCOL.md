> Historical implementation/validation record. For current clone instructions, use [README.md](README.md). Local smoke logs and the old development harness are archival evidence, not clone prerequisites.

# Experimental protocol and reporting schema

## Fixed design

- Original historical train/val/test membership; five targets aeroplane, car,
  horse, motorbike, person. Background negatives contain no target labels.
- Co-occurring labels are retained. An image with another target but without the
  current target is excluded from that target/background classifier.
- A: ImageNet ResNet-18 layer2/layer3/layer4 GAP, 128/256/512, never concatenated.
- B: DINOv2 ViT-S/14 final CLS 384, mean patches 384, spatial 2x2 1536.
  Spatial order is TL, TR, BL, BR, row-major; odd dimensions split at floor(n/2),
  with the extra row/column in bottom/right. Grid comes from actual tensor H/W
  and model patch size; incompatible counts fail explicitly.
- E4: OpenAI CLIP ViT-B/32 image embeddings, 512.
- RGB input and official evaluation transforms; no stochastic preprocessing.
- All weights frozen/eval, inference mode. Per-image L2 with epsilon 1e-12,
  after final aggregation; zero remains zero. Raw float32 features retained.
- No StandardScaler, PCA, validation/test-fitted transformations or second SVM
  normalization. LinearSVC with original hinge/dual settings, max_iter=20000,
  tol=0.0001; no retraining on train+val.
- A/B/C/E4 grid [0.1,1,10]. D grid [0.1,1,10,100,1000]. C selected independently
  per category by validation AP. Sorted ascending C gives lowest-C tie breaks.
- AP is sklearn non-interpolated average precision from continuous margins;
  target mAP is the unweighted mean of five per-category APs.

## Few-shot and regularization

Select one A/B representation using controlled validation mAP after independent
C selection. Competing scores are persisted. Lexical ascending representation
name breaks ties. Keep it fixed across C/D.
N=[1,5,10,20,all], seeds=[0,1,2,3,4]. Each category/seed receives a deterministic
permutation of its training positives; N uses prefixes. Exactly N original
positive images are used; impossible N raises rather than silently clipping.
All original training background negatives are reused across N and seeds.
Training IDs and actual counts are recorded, including secondary labels.

Conditions are none and horizontal_flip. Flip the original training image before
official preprocessing and extraction. Train on originals plus flipped images;
never average or flip vectors. Val/test are never augmented. Training AP for the
augmented condition includes those duplicate labeled vectors explicitly.

D reuses the exact subsets and representation, with no augmentation as the
principal analysis. Compatible 0.1/1/10 trials come from the shared singleton
cache; only 100/1000 require new fits. Each trial key hashes train/val feature
bytes, labels/IDs/split, category, seed, C, solver settings and trainer/provider
source. Test data are excluded from the key. The cache persists weights, bias,
trial AP/times/counts, and model checksums. C/D output folders remain separate
so D does not invalidate C's artifact manifest.

For each N/augmentation: first average the five category APs within each seed,
then report mean and sample std across the five seed mAPs. Per-category mean/std
is also persisted. Learning curve uses N labels (all is categorical), not an
invented common numeric all-count; original all-positive counts differ by class.
Regularization plots use logarithmic C and measured training/validation AP with
seed uncertainty. No expected bias/variance trend is asserted before measurement.

## CLIP

Descriptions retain airplane, car, horse, motorbike, person and the existing
background scene without target objects. Templates are exactly:
1. a photo of a {}
2. an image showing a {}
3. a picture of a {}

E1: fixed first template. E2: all templates, each text vector independently L2;
score is arithmetic mean of image/text cosine similarities. Mean text is not
renormalized. E3: each template separately on validation, select one globally
using five-target retrieval mAP; first template wins ties. Complete search saved.
Only E1/E2/E3's selected template receive test evaluation in final mode.
E4: cached image L2 features and the same controlled three-C SVM protocol.

Zero-shot argmax predicts one of six categories. Report exact target-label-set
accuracy and top-1 label membership separately, plus per-class precision, recall,
F1 and multilabel 2x2 confusion matrices. Retrieval AP ranks the full split, all
non-target images negative. Background is excluded from target mAP. There is no
conventional multiclass matrix that silently discards secondary labels.

## Visual analysis

Choose the best ResNet and DINO variant using controlled validation mAP. Explain
the actual independently trained target/background SVMs; CLIP uses E3. Case
selection uses within-model confidence ranks, category diversity and deterministic
ID tie breaks. Preserve reasons, labels, predictions, scores and both top-1
membership and exact multilabel correctness. Group availability depends on data;
no artificial examples are generated to fill empty groups.

Original and mean-RGB occluded images use the identical preprocessing path.
R=s_c(original)-s_c(occluded). The unoccluded scorer must reproduce its persisted
score within floating-point tolerance. Save signed arrays, category, image ID,
mode, baseline and score type. Ground-truth mode uses the lexically first
annotated target consistently across all models (recorded for multilabel images).
Predicted mode explains each model's own category. SVM background uses the
negative maximum target margin; this derived score is explicitly identified.
Each signed color scale has its own units; cosine drops and margin drops cannot
be interpreted as comparable magnitudes.

Optional DINO patch/CLS cosine maps are drawn over the actual preprocessed crop.
They describe representation similarity and are not explanations of the SVM.
No object mask or box is assumed. Human inspection is required to discuss
object/context evidence, small/occluded objects and texture confusions.

## Provenance, separation and reporting

Per-image feature cache identities exclude SVM C and include checkpoint content,
preprocess, adapter source, batch/device configuration, augmentation and image
content/ID/labels/split. Original and flipped caches are distinct. Raw/L2 vectors
and stored IDs/splits are verified, as are manifest checksums. No image is skipped.

Development writes train/val scores only. Final mode is a distinct run identity;
selection functions continue to use only train/val. No test metric selects C,
representation or prompt. Test labels may appear in dataset manifests to preserve
annotations, but never enter selection criteria. F uses validation examples in
development and test examples only in final mode.

Outputs and exact commands are in README. comparison_table.json/.csv includes
13 expected representations, dimension, protocol, validation/test mAP, AP per
category, selected C, shared extraction/cache storage and status. Missing
classical/full/test results remain null. comparison.json/.csv retains separate
protocol records. Search files retain all trials, not only winning aggregate AP.
Feature metadata retains component timings and hardware/runtime. Float32 storage
is 4*d per descriptor or 8*d for raw+L2; compressed multi-output cache storage and
forward time are shared, never summed across ResNet/DINO representations.
