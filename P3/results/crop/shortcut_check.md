Random forest on 11 brightness statistics of the model input (no tumor shape). Fit on train, scored on dev.

| input                                    |   dev_acc_4class |   dev_macro_f1_4class |   dev_acc_tumor_vs_notumor |
|:-----------------------------------------|-----------------:|----------------------:|---------------------------:|
| full slice (no crop)                     |            0.865 |                 0.845 |                      0.970 |
| crop, no-tumor = full slice (P2 default) |            0.837 |                 0.803 |                      0.986 |
| crop, no-tumor = size-matched decoy (P3) |            0.721 |                 0.690 |                      0.866 |
