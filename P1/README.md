# G03 Brain MRI Triage: Data Preparation (P1)

This part of the repo turns the two raw datasets into one clean, leakage-checked set of
train/dev/test splits. It also provides a shared PyTorch loader that every later stage
(segmentation, classification, Grad-CAM, triage) uses, so all stages see the same images
with the same preprocessing.

Owner: Anna Schellin (P1). Checkpoint: "Data pipeline ready", Nov 4.

## Quick start (P2, P3, P4)

```python
from brain_mri_data import make_loader, class_weights, LABELS

train_loader = make_loader('train', batch_size=32)   # shuffled
dev_loader   = make_loader('dev', batch_size=32)
# On Colab, point at the project folder on Drive:
# make_loader('train', root='/content/drive/MyDrive/integration_project')

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights())   # optional, see Class balance
```

Each batch is a dict:

| Key | Shape / type | Notes |
|---|---|---|
| `image` | float32 `[B, 3, 224, 224]` | ImageNet-normalized, ready for ResNet-50 |
| `mask` | float32 `[B, 1, 224, 224]` | 0/1 tumor mask aligned with `image`; all zeros for no-tumor |
| `has_mask` | bool `[B]` | False for no-tumor images |
| `label` | int64 `[B]` | index into `LABELS = ['glioma', 'meningioma', 'pituitary', 'notumor']` |
| `path` | list of str | original file path, for tracing results back to images |

**Preprocessing** (in `brain_mri_data.py`, identical for every split): grayscale → pad to square
with black → resize to 224×224 → copy to 3 channels → ImageNet mean/std. Masks get the same pad
and resize (nearest neighbour). Options: `keep_aspect=False` stretches instead of padding;
`hflip=True` randomly flips image and mask together. `denormalize(x)` turns a model input back
into a displayable image (useful for Grad-CAM overlays).

**The test split is for final reporting only.** Model selection and threshold tuning use train
and dev (see "Fair test" in the proposal).

If you read the CSVs yourself, use
`pd.read_csv('splits.csv', dtype={'patient_id': str}, keep_default_na=False)`; otherwise pandas
turns patient IDs into numbers and empty cells into NaN.

## Data setup

The image folders are too large for git. Download them and place them in the project folder
(on Colab, upload the whole folder to Drive) with these exact names, since all CSV paths are
relative to the project folder:

```
integration_project/
├── braintumormri_figshare/          # https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
│   ├── brainTumorDataPublic_1-766/      (.mat files, unzipped)
│   ├── brainTumorDataPublic_767-1532/
│   ├── brainTumorDataPublic_1533-2298/
│   ├── brainTumorDataPublic_2299-3064/
│   └── cvind.mat
└── braintumormri_kaggle/            # https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
    ├── Training/notumor/ ...
    └── Testing/notumor/ ...
```

Only Kaggle's `notumor` folders are used. Kaggle's tumor classes overlap with figshare.

Requirements: `h5py numpy pandas pillow scipy scikit-learn torch torchvision`.

## Pipeline

Run from the project folder, in this order. Every script is deterministic; splits use seed 539.

| # | Script | Does | Writes |
|---|---|---|---|
| 0 | `convert_mat.py` | Converts figshare `.mat` (MATLAB v7.3 / HDF5) to JPEG images and PNG masks | `brainTumorDataPublic_*_jpeg/images/<label>/`, `.../masks/<label>/` |
| 1 | `build_manifest.py` | One row per image: label, source, patient ID, size, color mode | `manifest.csv` |
| 2 | `clean_data.py` | Converts to grayscale, drops exact duplicates, groups near-duplicates | `manifest_clean.csv`, `cleaning_log.csv`, `cleaned/`, `near_duplicates.png` |
| 3 | `verify_masks.py` | Checks every image–mask pair against the hand-drawn tumor outline | `mask_check.csv`, `mask_overlays.png` |
| 4 | `make_splits.py` | Grouped, stratified 70/15/15 train/dev/test split with leakage assertions | **`splits.csv`** |
| 5 | `class_distribution.py` | Class distribution table | `class_distribution.csv`, `class_distribution.md` |
| 6 | `image_stats_baseline.py` | Shortcut check: simple models on non-tumor features (train → dev) | `baseline_results.csv`, `baseline_results.md`, `image_stats_features.csv` |
| 7 | `brain_mri_data.py` | Shared preprocessing + loader; `python brain_mri_data.py` runs a self-test | `preprocess_check.png` |

Steps 0–2 take a few minutes; the rest take about a minute each.

**`splits.csv` columns:** `path` (original file), `clean_path` (grayscale file to load),
`mask_path` (empty for no-tumor), `label`, `source` (figshare/kaggle), `patient_id` (figshare
only), `orig_split` (Kaggle's original folder, **not used**), `orig_w`, `orig_h`, `mode`,
`pixel_md5`, `phash`, `had_true_color`, `near_dup_group`, `mask_ok`, `mask_area_frac`,
`n_components`, `group_id`, `split`.

## What was done

**Cleaning (step 2).** No files were unreadable. 1,771 Kaggle images (RGB, RGBA, palette) were
converted to grayscale. 143 exact duplicates (identical grayscale pixels) were dropped, all from
Kaggle no-tumor. Near-duplicates (the same slice re-saved with different contrast or sharpening)
were found with a perceptual hash and confirmed by 64×64 pixel correlation (≥ 0.90 for Kaggle
pairs; checked by eye). They were grouped (1,417 images in 248 groups) rather than dropped. A
perceptual hash on its own is not enough here: it also matches neighbouring slices of the same
patient.

**Mask verification (step 3).** All 3,064 figshare pairs pass: same size as the image, strictly
binary, not empty, plausible area, inside the head, and matching the `tumorBorder` outline stored
in the `.mat` (median agreement 1.00). This also confirms the conversion kept the correct
orientation.

**Splits (step 4).** The split unit is a group, never a single image:
- figshare: one group per **patient** (233 patients, each with one label)
- Kaggle no-tumor: one group per **near-duplicate group** (single images are their own group)

Groups are assigned at random within each label to hit 70/15/15. The script stops with an error
if any patient, near-duplicate group or identical image appears in two splits.

## Results

### Class distribution

| Label | Source | Raw | After cleaning | Train | Dev | Test | Groups train / dev / test |
|---|---|---:|---:|---:|---:|---:|---|
| glioma | figshare | 1426 | 1426 | 997 (30.1%) | 225 (30.8%) | 204 (29.9%) | 61 / 14 / 14 patients |
| meningioma | figshare | 708 | 708 | 498 (15.1%) | 113 (15.5%) | 97 (14.2%) | 61 / 11 / 10 patients |
| pituitary | figshare | 930 | 930 | 647 (19.6%) | 148 (20.2%) | 135 (19.8%) | 42 / 11 / 9 patients |
| notumor | kaggle | 1800 | 1657 | 1166 (35.2%) | 245 (33.5%) | 246 (36.1%) | 353 / 66 / 69 unique slices |
| **total** | | 4864 | 4721 | 3308 | 731 | 682 | |

Percentages are each class's share of that split. The train imbalance is 2.3× (no-tumor vs.
meningioma). `class_weights()` gives meningioma 1.48, pituitary 1.14, glioma 0.74, no-tumor 0.63.

### Image-stats baseline (dev split)

Simple classifiers trained on features that contain **no tumor information**. Best model shown
(random forest in every row); full grid in `baseline_results.csv`.

| Task | Features | Dev accuracy | Dev macro-F1 |
|---|---|---:|---:|
| Tumor vs. no-tumor | majority class | 66.5% | 0.399 |
| | file metadata (size, aspect, color mode, file size) | **100%** | 1.000 |
| | brightness stats of the 224×224 grayscale input | **97.9%** | 0.977 |
| | brightness stats after histogram equalization | 92.6% | 0.916 |
| 4-class | majority class | 33.5% | 0.126 |
| | brightness stats of the 224×224 input | **86.5%** | **0.843** |
| | metadata + brightness | 89.5% | 0.879 |
| 3-class, tumors only | majority class | 46.3% | 0.211 |
| | brightness stats of the 224×224 input | 83.1% | 0.826 |

A depth-2 decision tree on metadata gets 99.5% on tumor vs. no-tumor with two rules: "color
file → no tumor; otherwise taller than 571 px → no tumor".

## Findings the team should know

1. **Figshare has patient IDs** (`cjdata.PID`, 233 patients). Tumor classes are split at the
   patient level. The proposal's "no patient IDs" limitation now applies only to the Kaggle
   no-tumor class, which is split at the slice level (with near-duplicates kept together).
   Different slices from the same no-tumor patient could still land in different splits.

2. **Kaggle's own Training/Testing split is not used.** Kaggle's v2 notes say duplicates and
   train/test overlap were removed, and that holds for byte-identical files (0 across folders).
   It does not hold for re-saved copies: 9 pixel-identical pairs and 179 near-duplicate groups
   span both folders (e.g. `Tr-no_1096.jpg` and `Te-no_77.jpg` have identical pixels). The
   no-tumor class has about 490 unique slices behind its 1,657 images.

3. **The source shortcut is real and survives preprocessing.** File metadata alone separates
   tumor from no-tumor perfectly, and brightness statistics of the preprocessed input still
   reach 97.9%. High no-tumor accuracy is therefore not evidence that a model looks at tumors.
   **For P3:** the bar to beat is 86.5% accuracy / 0.843 macro-F1 on dev, not the 33.5%
   majority-class baseline. Report macro-F1.

4. **The no-tumor images appear to be a different MRI sequence.** In sample batches, Kaggle
   no-tumor slices look like axial T2/FLAIR, while all figshare tumors are T1 contrast-enhanced,
   in mixed views. This is observed in samples, not measured. If it holds, it is likely the main
   driver of finding 3, and no preprocessing removes it.

5. **For P2, cropping no-tumor images.** There is no tumor mask for no-tumor images. However
   they are cropped (empty U-Net mask, full slice, random patch), that difference is visible to
   the classifier and can become a label shortcut. Decide this before training the crop
   classifier.

6. **For P2, mask shape.** 35 masks have 2–3 separate regions, so crop to the bounding box of
   all regions. Pituitary tumors are about half the area of the others (median 0.7% of the slice
   vs. 1.5–1.8%), so fixed-padding crops include more surrounding tissue for pituitary.

7. **Small evaluation sets.** Dev and test have about 10–14 patients per tumor class, so per-class
   F1 will be noisy. Report it with that caveat.

8. **Brightness separates tumor types too** (83% among figshare only). Source can't explain
   this; the class-by-class mix of views or genuine tumor brightness might. Not investigated
   further.

## Notes

- `scipy.io.loadmat` cannot read the figshare files (MATLAB v7.3); `convert_mat.py` uses `h5py`
  and transposes, because MATLAB stores arrays column-major.
- On Windows, scikit-learn's `n_jobs=-1` stalled the random forest in step 6, so it runs
  single-threaded.
- Thresholds in steps 2 and 3 were chosen after visual inspection; the reasoning is in each
  script's docstring.
