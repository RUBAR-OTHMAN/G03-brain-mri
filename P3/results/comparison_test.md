# Full slice vs. crop (test, n = 682)

ResNet-50 with the same training setup and seed; only the input differs. Checkpoints: best dev macro-F1 epoch of each train_classifier.py run. Test was run once with settings fixed on dev.

Inputs:

- **full slice**: the full-slice model on the whole slice.
- **crop, true masks (oracle)**: the crop model on crops around the ground-truth mask. A real input has no ground truth, so this is an upper bound.
- **crop, U-Net masks (pipeline)**: the crop model on crops around P2's U-Net mask; an empty mask gets P2's decoy box. These are the crops P4's pipeline uses, with no ground truth anywhere.
- **crop, U-Net masks, P3 decoy**: as the pipeline row, but an empty mask gets the size-matched decoy the crop model trained on.

Patients per class: glioma 14, meningioma 10, pituitary 9, notumor 69 (no-tumor counts near-duplicate groups). 95% CIs: patient bootstrap, 2000 resamples within each class, paired across inputs.

| input                        |   acc |   macro-F1 | macro-F1 95% CI   |   change vs. full | change 95% CI    |   errors |   missed tumors |   false alarms |
|:-----------------------------|------:|-----------:|:------------------|------------------:|:-----------------|---------:|----------------:|---------------:|
| full slice                   | 0.972 |      0.966 | 0.938 to 0.989    |            +0.000 |                  |       19 |               0 |              3 |
| crop, true masks (oracle)    | 0.963 |      0.956 | 0.909 to 0.988    |            -0.011 | -0.049 to 0.024  |       25 |               7 |              0 |
| crop, U-Net masks (pipeline) | 0.940 |      0.930 | 0.884 to 0.967    |            -0.036 | -0.071 to -0.004 |       41 |              12 |              2 |
| crop, U-Net masks, P3 decoy  | 0.931 |      0.923 | 0.875 to 0.960    |            -0.043 | -0.077 to -0.013 |       47 |              21 |              1 |

Missed tumors: glioma, meningioma or pituitary predicted as no-tumor. False alarms: no-tumor predicted as a tumor class.

## U-Net masks

- 50 of 436 tumor images (11.5%) get an empty U-Net mask and so a decoy crop: glioma 36, meningioma 1, pituitary 13. The pipeline classifies 36 of them correctly and calls 7 no-tumor.
- 5 of 246 no-tumor images get a non-empty U-Net mask.
- On the 386 tumor images with a U-Net mask, accuracy is 0.935.

## Per-class F1

| class      |   n |   F1 full |   F1 crop_gt |   F1 crop_pred |   F1 crop_pred_p3decoy |
|:-----------|----:|----------:|-------------:|---------------:|-----------------------:|
| glioma     | 204 |     0.968 |        0.964 |          0.947 |                  0.936 |
| meningioma |  97 |     0.943 |        0.928 |          0.918 |                  0.913 |
| pituitary  | 135 |     0.960 |        0.945 |          0.882 |                  0.885 |
| notumor    | 246 |     0.994 |        0.986 |          0.972 |                  0.957 |

## Per-class recall

| class      |   full |   crop_gt |   crop_pred |   crop_pred_p3decoy |
|:-----------|-------:|----------:|------------:|--------------------:|
| glioma     |  0.971 |     0.971 |       0.966 |               0.936 |
| meningioma |  0.938 |     0.866 |       0.866 |               0.866 |
| pituitary  |  0.970 |     0.956 |       0.859 |               0.852 |
| notumor    |  0.988 |     1.000 |       0.992 |               0.996 |

## Per-class precision

| class      |   full |   crop_gt |   crop_pred |   crop_pred_p3decoy |
|:-----------|-------:|----------:|------------:|--------------------:|
| glioma     |  0.966 |     0.957 |       0.929 |               0.936 |
| meningioma |  0.948 |     1.000 |       0.977 |               0.966 |
| pituitary  |  0.949 |     0.935 |       0.906 |               0.920 |
| notumor    |  1.000 |     0.972 |       0.953 |               0.921 |

## Confusion, full slice (rows true, columns predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      198 |            5 |           1 |         0 |
| meningioma    |        2 |           91 |           4 |         0 |
| pituitary     |        4 |            0 |         131 |         0 |
| notumor       |        1 |            0 |           2 |       243 |

## Confusion, crop, true masks (oracle) (rows true, columns predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      198 |            0 |           1 |         5 |
| meningioma    |        5 |           84 |           8 |         0 |
| pituitary     |        4 |            0 |         129 |         2 |
| notumor       |        0 |            0 |           0 |       246 |

## Confusion, crop, U-Net masks (pipeline) (rows true, columns predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      197 |            0 |           2 |         5 |
| meningioma    |        4 |           84 |           9 |         0 |
| pituitary     |       10 |            2 |         116 |         7 |
| notumor       |        1 |            0 |           1 |       244 |

## Confusion, crop, U-Net masks, P3 decoy (rows true, columns predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      191 |            1 |           2 |        10 |
| meningioma    |        4 |           84 |           8 |         1 |
| pituitary     |        8 |            2 |         115 |        10 |
| notumor       |        1 |            0 |           0 |       245 |
