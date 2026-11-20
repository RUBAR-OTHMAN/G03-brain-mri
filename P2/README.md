# G03 Brain MRI Triage: Segmentation & Tumor Cropping (P2)

Owner: Anika Lindell (P2)  
Checkpoints: 
- **Nov 8:** Preliminary U-Net segmentation on sample images
- **Nov 10:** Held-out quantitative evaluation & Dice benchmark
- **Nov 15:** Tumor cropping module for downstream ResNet-50 classifier (P3) and Grad-CAM pipeline (P4)

---

## 1. Overview & Architectural Decisions

P2 provides the segmentation and standardized cropping pipeline bridging raw full-slice MRI images (from P1) to the deep learning classifier (P3) and explainability/triage pipeline (P4).

### Key Decisions:
1. **Autonomous U-Net vs. Interactive MedSAM (Nov 10 Decision):**
   - **Autonomous Pipeline (U-Net):** We trained our own 4-stage U-Net (`train_unet.py`, `best_unet.pth`) with parameter updates across 5 epochs. Model selection is based strictly on **Tumor-Only Mean Dice** (ignoring the free 1.0 points from empty no-tumor predictions). It takes raw, unannotated MRI slices and generates binary masks completely autonomously with zero human intervention.
   - **Interactive Benchmark (MedSAM):** Evaluated pretrained MedSAM (`p2_medsam.ipynb`, PR #5). Because MedSAM requires bounding box prompts (which were derived from ground-truth annotations in evaluation, yielding 0.922 Dice), it serves as our "radiologist-in-the-loop / oracle upper-bound" benchmark.
2. **Tumor Cropping & Decoy Box Strategy (Nov 15 Milestone):**
   - Tumors are cropped with proportional padding (`pad_frac=0.20`), square bounding boxes (preserving anatomical aspect ratio), and resized to standardized 224×224 tensors.
   - **No-Tumor Shortcut Elimination:** Rather than keeping the full uncropped slice for no-tumor images (which allowed trivial classification via black border area / zoom alone), `crop_tumor.py` defaults to `notumor_mode='decoy'`. It crops a 48–80 px box inside the brain parenchyma matching typical train tumor dimensions.
3. **Non-Oracle Evaluation & Zero Label Leakage:**
   - `CroppedBrainMRIDataset` supports `masks='gt'` (for classifier training) and `masks='pred'` (for non-oracle dev/test evaluation).
   - When `masks='pred'`, the crop decision depends strictly on the predicted mask from `predict_masks.py`—if empty, it automatically triggers a decoy crop without inspecting the true `has_mask` label.

---

## 2. Quantitative Evaluation Results (Full Dev Split, N=731)

Evaluated on the complete held-out `dev` set (486 tumor slices, 245 no-tumor slices) across 5 training epochs:

| Metric | Epoch 1 Baseline | **Epoch 5 (Current Best)** |
|---|---|---|
| **Glioma Mean Dice** | 0.265 | **0.562** (+112%) |
| **Meningioma Mean Dice** | 0.465 | **0.828** (+78%) |
| **Pituitary Mean Dice** | 0.263 | **0.767** (+192%) |
| **Tumor-Only Mean Dice** | 0.331 | **0.719** (+117%) |
| **No-Tumor Specificity (Dice)** | 0.661 | **0.992** |
| **Combined Dev Dice** | 0.428 | **0.789** |

*All epoch-by-epoch loss and Dice metrics are recorded in `P2/train_unet_history.csv`.*

---

## 3. Quickstart: How to Use for P3 and P4

### A. Load Pre-Computed Predicted Masks
Pre-computed binary masks for the entire `dev` (731 slices) and `test` (682 slices) splits are saved as dual-keyed `.npz` files (supporting both raw `path` and `clean_path` lookups):
```python
import numpy as np

# Load masks dictionary:
dev_masks = np.load('P2/dev_predicted_masks.npz')
mask_224 = dev_masks['cleaned/braintumormri_figshare/1000.png']  # [224, 224] uint8 (0 or 1)
```

### B. Use in PyTorch DataLoader
Drop-in replacement matching P1's `make_loader` API:
```python
from crop_tumor import make_cropped_loader

# 1. Training loader (uses clean ground-truth crops):
train_loader = make_cropped_loader('train', batch_size=32)

# 2. Non-oracle evaluation loader (uses U-Net predicted masks, zero label leak):
dev_loader = make_cropped_loader('dev', batch_size=32, masks='pred')
```

### C. Re-generate Masks for Any Split
```bash
python P2/predict_masks.py --splits dev test --checkpoint best_unet.pth --output-dir P2/
```

---

## 4. File Manifest

| File | Description |
|---|---|
| `P2/crop_tumor.py` | Aspect-preserving tumor cropping module with decoy crop & `masks='pred'` support |
| `P2/predict_masks.py` | Script to generate predicted binary mask archives (.npz) for any split |
| `P2/train_unet.py` | PyTorch training pipeline selecting checkpoints on Tumor-Only Dice |
| `P2/train_unet_history.csv` | Full 5-epoch training history showing loss and per-class validation Dice |
| `P2/dev_predicted_masks.npz` | Pre-computed predicted masks for all 731 dev images (dual-keyed, 0.39 MB) |
| `P2/test_predicted_masks.npz` | Pre-computed predicted masks for all 682 test images (dual-keyed, 0.36 MB) |
| `P2/p2_unet.ipynb` | U-Net training and evaluation notebook |
| `P2/p2_crop_tumor.ipynb` | Tumor cropping verification and visual inspection notebook |
| `P2/p2_medsam.ipynb` | MedSAM interactive benchmark notebook (0.922 Dice) |
| `P2/nov8_preliminary_5masks.png` | Nov 8 milestone: 5-sample preliminary mask visual report |
| `P2/nov10_20image_mask_report.png` | Nov 10 milestone: 20-image evaluation report sampled across 16 patients (Epoch 5 predictions) |
| `P2/nov15_cropped_preview.png` | Nov 15 milestone: 8-sample tumor crop verification preview |

### Model Weights
Trained weights (`best_unet.pth`, 93 MB) are located in the project root and mirrored on the team Google Drive under `models/best_unet.pth`.
