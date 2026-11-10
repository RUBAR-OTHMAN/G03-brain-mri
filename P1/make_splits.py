"""P1 step 4: stratified, leakage-free train/dev/test splits.

Splitting unit is a group, never a single image:
- figshare: one group per patient (all slices of a patient stay together)
- kaggle no-tumor: one group per near_dup_group (all re-saved copies of a slice stay together);
  images without near-duplicates are their own group
Kaggle's own Training/Testing folders are ignored: they share ~2,000 near-duplicate pairs.

Each group has a single label, so groups are split within each label (stratified), aiming for
TRAIN/DEV/TEST fractions of that label's images.

Output: splits.csv (manifest_clean.csv + group_id, split, mask columns from mask_check.csv)
Run after clean_data.py and verify_masks.py.
"""
import os

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
manifest_csv = os.path.join(project_root, 'manifest_clean.csv')
mask_check_csv = os.path.join(project_root, 'mask_check.csv')
output_csv = os.path.join(project_root, 'splits.csv')

SPLIT_FRACS = {'train': 0.70, 'dev': 0.15, 'test': 0.15}
SEED = 539

df = pd.read_csv(manifest_csv, dtype={'patient_id': str}, keep_default_na=False)
masks = pd.read_csv(mask_check_csv, keep_default_na=False)[['path', 'mask_ok', 'mask_area_frac', 'n_components']]
df = df.merge(masks, on='path', how='left')

# Drop figshare pairs that failed the mask checks (none at the time of writing)
bad_masks = (df.source == 'figshare') & (df.mask_ok != True)
if bad_masks.any():
    print(f"Dropping {int(bad_masks.sum())} figshare images with failed mask checks")
df = df[~bad_masks].copy()

df['group_id'] = np.where(
    df.source == 'figshare', 'patient_' + df.patient_id,
    np.where(df.near_dup_group != '', 'kaggle_' + df.near_dup_group, 'kaggle_single_' + df.index.astype(str)))

assert (df.groupby('group_id').label.nunique() == 1).all(), "a group has more than one label"

# Greedy assignment per label: shuffle groups, give each to the split furthest below its target
rng = np.random.default_rng(SEED)
group_split = {}
for label, label_df in df.groupby('label'):
    sizes = label_df.groupby('group_id').size()
    # Random order (not largest-first) so big and small groups are spread evenly across splits
    order = rng.permutation(sizes.index.to_numpy())
    total = sizes.sum()
    filled = dict.fromkeys(SPLIT_FRACS, 0)
    for g in order:
        split = max(SPLIT_FRACS, key=lambda s: SPLIT_FRACS[s] * total - filled[s])
        group_split[g] = split
        filled[split] += sizes[g]

df['split'] = df.group_id.map(group_split)

# Leakage checks: nothing identifying may appear in more than one split
for col in ['group_id', 'pixel_md5']:
    assert (df.groupby(col).split.nunique() == 1).all(), f"{col} appears in more than one split"
fig = df[df.source == 'figshare']
assert (fig.groupby('patient_id').split.nunique() == 1).all(), "a patient appears in more than one split"
nd = df[df.near_dup_group != '']
assert (nd.groupby('near_dup_group').split.nunique() == 1).all(), "a near-duplicate group is split"

df.to_csv(output_csv, index=False)

# Summary
print(f"Wrote {len(df)} rows to {output_csv} (seed {SEED})")
print("Leakage checks passed: no patient, near-duplicate group, or identical image spans two splits\n")
counts = pd.crosstab(df.label, df.split)[list(SPLIT_FRACS)]
counts.loc['total'] = counts.sum()
print("Images per label/split:")
print(counts.to_string())
print("\nFraction of each label's images per split (target "
      + '/'.join(f'{v:.2f}' for v in SPLIT_FRACS.values()) + "):")
print(counts.div(counts.sum(axis=1), axis=0).round(3).to_string())
print("\nGroups (patients / Kaggle slices) per label/split:")
print(pd.crosstab(df.label, df.split, values=df.group_id, aggfunc='nunique')[list(SPLIT_FRACS)].to_string())
