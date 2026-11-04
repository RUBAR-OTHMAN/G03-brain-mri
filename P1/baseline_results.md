# Image-stats baseline (dev split)

Simple classifiers on features with no tumor information. Fit on train, scored on dev; test untouched. Best model per row shown; full grid in baseline_results.csv.

| Task | Features | Best model | Dev accuracy | Dev macro-F1 |
|---|---|---|---:|---:|
| 3-class_tumors_only | none | majority | 46.3% | 0.211 |
| 3-class_tumors_only | intensity | random_forest | 83.1% | 0.826 |
| 3-class_tumors_only | intensity_eq | random_forest | 80.0% | 0.791 |
| 4-class | none | majority | 33.5% | 0.126 |
| 4-class | metadata | random_forest | 69.6% | 0.636 |
| 4-class | intensity | random_forest | 86.5% | 0.843 |
| 4-class | all | random_forest | 89.5% | 0.879 |
| 4-class | intensity_eq | random_forest | 81.3% | 0.790 |
| tumor_vs_notumor | none | majority | 66.5% | 0.399 |
| tumor_vs_notumor | metadata | random_forest | 100.0% | 1.000 |
| tumor_vs_notumor | intensity | random_forest | 97.9% | 0.977 |
| tumor_vs_notumor | all | random_forest | 100.0% | 1.000 |
| tumor_vs_notumor | intensity_eq | random_forest | 92.6% | 0.916 |

Depth-2 tree on metadata, tumor vs no-tumor (dev accuracy 99.5%):

```
|--- was_color_file <= 0.50
|   |--- orig_h <= 571.00
|   |   |--- class: tumor
|   |--- orig_h >  571.00
|   |   |--- class: notumor
|--- was_color_file >  0.50
|   |--- class: notumor
```
