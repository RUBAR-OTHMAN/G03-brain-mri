# Full slice vs. oracle crop (dev, n = 731)

ResNet-50, same training setup and seed; only the input differs. Each run is scored at its best dev macro-F1 epoch. Crops use ground-truth masks, so the crop numbers are likely optimistic. Test is not used.

| input       | best epoch   |   dev acc |   dev macro-F1 |   dev errors |
|:------------|:-------------|----------:|---------------:|-------------:|
| full slice  | 12 / 12      |     0.971 |          0.966 |           21 |
| oracle crop | 7 / 12       |     0.979 |          0.978 |           15 |

## Per class

| class      |   n |   F1 full |   F1 crop |   F1 change |   recall full |   recall crop |   precision full |   precision crop |
|:-----------|----:|----------:|----------:|------------:|--------------:|--------------:|-----------------:|-----------------:|
| glioma     | 225 |     0.961 |     0.975 |      +0.014 |         0.942 |         0.956 |            0.981 |            0.995 |
| meningioma | 113 |     0.924 |     0.982 |      +0.059 |         0.965 |         0.982 |            0.886 |            0.982 |
| pituitary  | 148 |     0.987 |     0.963 |      -0.023 |         0.993 |         0.980 |            0.980 |            0.948 |
| notumor    | 245 |     0.994 |     0.992 |      -0.002 |         0.988 |         1.000 |            1.000 |            0.984 |

## Confusion, full slice (rows true, cols predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      212 |           13 |           0 |         0 |
| meningioma    |        4 |          109 |           0 |         0 |
| pituitary     |        0 |            1 |         147 |         0 |
| notumor       |        0 |            0 |           3 |       242 |

## Confusion, oracle crop (rows true, cols predicted)

| true \ pred   |   glioma |   meningioma |   pituitary |   notumor |
|:--------------|---------:|-------------:|------------:|----------:|
| glioma        |      215 |            1 |           6 |         3 |
| meningioma    |        0 |          111 |           2 |         0 |
| pituitary     |        1 |            1 |         145 |         1 |
| notumor       |        0 |            0 |           0 |       245 |
