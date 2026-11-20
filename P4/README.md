# P4: Trust check, triage rule, and pipeline integration

**Owner:** Rubar Abdalla (@RUBAR-OTHMAN)

## Files
- `trust_triage.py`: Grad-CAM, heatmap–mask IoU, triage rule, full pipeline, threshold tuning, summaries and plots.
- `p4_trust_triage.ipynb`: runs the full pipeline on dev (outputs saved), tunes thresholds, shows 3 end-to-end examples, confident mistakes, and the U-Net vs. true mask comparison. Runs the test set once when `RUN_TEST = True`.
- `results/`: dev results (see below).

## Pipeline
slice → U-Net mask (P2) → crop (`P2/crop_tumor.py`) → ResNet-50 (P3) → class + confidence → Grad-CAM on the crop (layer4) → heatmap pasted back to the full slice → IoU with mask → tier.

## Triage rule
| Tier | Condition |
|---|---|
| High Priority | glioma or meningioma, confidence > 0.80, IoU > 0.20 |
| Needs Review | confidence < 0.60, or IoU < 0.02 |
| Routine | everything else |

- When the prediction is `notumor`, IoU is undefined, so only the confidence test is used.
- IoU uses the heatmap thresholded at 0.5 of its max.
- **IoU thresholds were tuned on dev only.** The proposal used 0.50 / 0.30, but IoU is low even for correct predictions (median 0.14), so those values sent 56% of cases to review and only 1.5% of urgent cases to High Priority. We chose LO = 0.02 (error catch is flat from 0.02 to 0.08, so this gives it with the fewest reviews) and HI = 0.20 (lowest value with no wrong prediction in High Priority). Full grid: `results/dev_threshold_grid_unet.csv`.

## Dev results (U-Net masks, crop model, n = 731)
| | Proposal thresholds (0.30 / 0.50) | Tuned (0.02 / 0.20) |
|---|---|---|
| Accuracy | 0.952 | 0.952 |
| Sent to review | 56% | 15% |
| Errors caught by Needs Review | 74% | 51% |
| Glioma/meningioma reaching High Priority | 1.5% | 38% |
| Wrong predictions in High Priority | 0 | 0 |

- With true masks instead of U-Net masks, accuracy is 0.978 (`results/dev_crop_gt.csv`). Full-slice model: `results/dev_full_gt.csv`.
- Mean IoU is 0.155 with U-Net masks and 0.146 against true masks. Only 13 of 474 tumor predictions go to review because of a bad mask, so low IoU comes from Grad-CAM, not mask quality.

## Limitations
- Grad-CAM on ResNet-50 layer4 is a coarse 7×7 grid, and the tumor fills only about 24% of the crop, so IoU stays low even when the model is right.
- Crops are centred on the mask, so IoU checks "did the model look at the tumor inside the crop", not "did it find the tumor in the whole slice".
- Most missed errors are `notumor` predictions, where IoU cannot help; confidence is the only check there.
- 140 correctly classified glioma/meningioma cases land in Routine (IoU between 0.02 and 0.20). A clinical system would likely send these to review.

## How to run
Put `p3_crop_best.pt` (Mihir) and `best_unet.pth` (Anika) in `MODELS_DIR` (model files are not in git), set `DATA_ROOT` and `REPO_ROOT` in the first cell, and run all cells. About 4 minutes on CPU for dev.
