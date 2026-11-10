# Class distribution

| Label | Source | Raw | After cleaning | Train | Dev | Test | Groups train / dev / test |
|---|---|---:|---:|---:|---:|---:|---|
| glioma | figshare | 1426 | 1426 | 997 (30.1%) | 225 (30.8%) | 204 (29.9%) | 61 / 14 / 14 patients |
| meningioma | figshare | 708 | 708 | 498 (15.1%) | 113 (15.5%) | 97 (14.2%) | 61 / 11 / 10 patients |
| pituitary | figshare | 930 | 930 | 647 (19.6%) | 148 (20.2%) | 135 (19.8%) | 42 / 11 / 9 patients |
| notumor | kaggle | 1800 | 1657 | 1166 (35.2%) | 245 (33.5%) | 246 (36.1%) | 353 / 66 / 69 unique slices |
| **total** |  | 4864 | 4721 | 3308 (100.0%) | 731 (100.0%) | 682 (100.0%) | 517 / 102 / 102 |

Percentages are each class's share of that split.
Raw to cleaned: exact duplicates (identical grayscale pixels) removed; no unreadable files or failed masks.
Splits are grouped (figshare by patient ID, Kaggle no-tumor by near-duplicate group) and stratified by label, seed 539.
Majority-class baseline (always predict notumor): train accuracy 35.2%, dev accuracy 33.5%, test accuracy 36.1%.
Imbalance: largest/smallest class in train = 2.34x.
