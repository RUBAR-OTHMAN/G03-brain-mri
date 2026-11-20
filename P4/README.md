# P4: Trust check, triage rule, and pipeline integration

**Owner:** Rubar Abdalla (@RUBAR-OTHMAN)

## Files
- `trust_triage.py`: Grad-CAM, heatmap–mask IoU, the triage rule, the full pipeline, summaries and plots.
- `p4_trust_triage.ipynb`: runs the pipeline on dev, shows 3 end-to-end examples, confident mistakes, the U-Net vs true mask comparison, and the final test run.

## Pipeline
slice → mask (true or P2 U-Net) → crop (`P2/crop_tumor.py`) → ResNet-50 (P3 `best.pt`) → class + confidence → Grad-CAM on the crop (layer4) → heatmap pasted back to the full slice → IoU with mask → tier.

## Triage rule
| Tier | Condition |
|---|---|
| High Priority | glioma or meningioma, confidence > 0.80, IoU > 0.50 |
| Needs Review | confidence < 0.60, or IoU < 0.30 |
| Routine | everything else |

Added: when the prediction is `notumor`, IoU is undefined, so only the confidence test is used.
IoU uses the heatmap thresholded at 0.5 (of its max).

## What it needs
- P1 data root (`integration_project`, with `brain_mri_data.py` and `splits.csv`)
- `P2/crop_tumor.py` and `P2/train_unet.py` (branch `p2-tumor-cropping` until merged)
- P3 crop classifier checkpoint `best.pt` (not in git)
- P2 U-Net checkpoint `best_unet.pth` (not in git), only for `MASK_SOURCE = 'unet'`

## Status
- Code is tested end to end on sample dev images with untrained weights. Real numbers come once the checkpoints are in.
- `MASK_SOURCE = 'gt'` uses true masks as a stand-in. The final run should use `'unet'`.

## Limitation
Crops are centred on the mask, so a heatmap near the crop centre already overlaps the mask. The IoU checks "did the model look at the tumor inside the crop", not "did it find the tumor in the whole slice".
