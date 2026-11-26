# Full slice vs. crop (dev, n = 731)

ResNet-50 with the same training setup and seed; only the input differs. Checkpoints: best dev macro-F1 epoch of each train_classifier.py run. Test is not used.

Inputs:

- **full slice**: the full-slice model on the whole slice.
- **crop, true masks (oracle)**: the crop model on crops around the ground-truth mask. A real input has no ground truth, so this is an upper bound.
- **crop, U-Net masks (pipeline)**: the crop model on crops around P2's U-Net mask; an empty mask gets P2's decoy box. These are the crops P4's pipeline uses, with no ground truth anywhere.
- **crop, U-Net masks, P3 decoy**: as the pipeline row, but an empty mask gets the size-matched decoy the crop model trained on.

Patients per class: glioma 14, meningioma 11, pituitary 11, notumor 66 (no-tumor counts near-duplicate groups). 95% CIs: patient bootstrap, 2000 resamples within each class, paired across inputs.

| input                        |   acc |   macro-F1 | macro-F1 95% CI   |   change vs. full | change 95% CI   |   errors |   missed tumors |   false alarms |
|:-----------------------------|------:|-----------:|:------------------|------------------:|:----------------|---------:|----------------:|---------------:|
| full slice                   | 0.971 |      0.966 | 0.937 to 0.988    |            +0.000 |                 |       21 |               0 |              3 |
| crop, true masks (oracle)    | 0.979 |      0.978 | 0.959 to 0.993    |            +0.012 | -0.017 to 0.044 |       15 |               4 |              0 |
| crop, U-Net masks (pipeline) | 0.952 |      0.947 | 0.915 to 0.969    |            -0.019 | -0.048 to 0.009 |       35 |              13 |              1 |
| crop, U-Net masks, P3 decoy  | 0.943 |      0.940 | 0.909 to 0.961    |            -0.027 | -0.058 to 0.003 |       42 |              19 |              0 |

Missed tumors: glioma, meningioma or pituitary predicted as no-tumor. False alarms: no-tumor predicted as a tumor class.

## U-Net masks

- 52 of 486 tumor images (10.7%) get an empty U-Net mask and so a decoy crop: glioma 39, meningioma 8, pituitary 5. The pipeline classifies 39 of them correctly and calls 10 no-tumor.
- 2 of 245 no-tumor images get a non-empty U-Net mask.
- On the 434 tumor images with a U-Net mask, accuracy is 0.952.

## Per-class F1

| class      |   n |   F1 full |   F1 crop_gt |   F1 crop_pred |   F1 crop_pred_p3decoy |
|:-----------|----:|----------:|-------------:|---------------:|-----------------------:|
| glioma     | 225 |     0.961 |        0.975 |          0.950 |                  0.927 |
| meningioma | 113 |     0.924 |        0.982 |          0.927 |                  0.927 |
| pituitary  | 148 |     0.987 |        0.963 |          0.940 |                  0.943 |
| notumor    | 245 |     0.994 |        0.992 |          0.972 |                  0.963 |

## Per-class recall

| class      |   full |   crop_gt |   crop_pred |   crop_pred_p3decoy |
|:-----------|-------:|----------:|------------:|--------------------:|
| glioma     |  0.942 |     0.956 |       0.929 |               0.898 |
| meningioma |  0.965 |     0.982 |       0.894 |               0.894 |
| pituitary  |  0.993 |     0.980 |       0.959 |               0.953 |
| notumor    |  0.988 |     1.000 |       0.996 |               1.000 |

## Per-class precision

| class      |   full |   crop_gt |   crop_pred |   crop_pred_p3decoy |
|:-----------|-------:|----------:|------------:|--------------------:|
| glioma     |  0.981 |     0.995 |       0.972 |               0.957 |
| meningioma |  0.886 |     0.982 |       0.962 |               0.962 |
| pituitary  |  0.980 |     0.948 |       0.922 |               0.934 |
| notumor    |  1.000 |     0.984 |       0.949 |               0.928 |

## Confusion, full slice (rows true, columns predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      212 |           13 |           0 |         0 |
| meningioma    |        4 |          109 |           0 |         0 |
| pituitary     |        0 |            1 |         147 |         0 |
| notumor       |        0 |            0 |           3 |       242 |

## Confusion, crop, true masks (oracle) (rows true, columns predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      215 |            1 |           6 |         3 |
| meningioma    |        0 |          111 |           2 |         0 |
| pituitary     |        1 |            1 |         145 |         1 |
| notumor       |        0 |            0 |           0 |       245 |

## Confusion, crop, U-Net masks (pipeline) (rows true, columns predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      209 |            3 |           8 |         5 |
| meningioma    |        4 |          101 |           4 |         4 |
| pituitary     |        2 |            0 |         142 |         4 |
| notumor       |        0 |            1 |           0 |       244 |

## Confusion, crop, U-Net masks, P3 decoy (rows true, columns predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      202 |            3 |           7 |        13 |
| meningioma    |        5 |          101 |           3 |         4 |
| pituitary     |        4 |            1 |         141 |         2 |
| notumor       |        0 |            0 |           0 |       245 |
