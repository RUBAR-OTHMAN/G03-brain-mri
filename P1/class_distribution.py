"""P1 step 5: class distribution table.

Per label: raw image count (manifest.csv), count after cleaning, images per split,
share of each split, and number of independent groups (patients / unique Kaggle slices) per split.

Outputs: class_distribution.csv, class_distribution.md (paste into report / README)
Run after make_splits.py.
"""
import os

import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
manifest_csv = os.path.join(project_root, 'manifest.csv')
splits_csv = os.path.join(project_root, 'splits.csv')
output_csv = os.path.join(project_root, 'class_distribution.csv')
output_md = os.path.join(project_root, 'class_distribution.md')

SPLITS = ['train', 'dev', 'test']
LABELS = ['glioma', 'meningioma', 'pituitary', 'notumor']

raw = pd.read_csv(manifest_csv, keep_default_na=False)
df = pd.read_csv(splits_csv, dtype={'patient_id': str}, keep_default_na=False)

images = pd.crosstab(df.label, df.split).reindex(index=LABELS, columns=SPLITS)
groups = pd.crosstab(df.label, df.split, values=df.group_id, aggfunc='nunique').reindex(index=LABELS, columns=SPLITS)
share = images / images.sum()

table = pd.DataFrame({
    'source': df.groupby('label').source.first().reindex(LABELS),
    'group_unit': ['patient', 'patient', 'patient', 'unique slice'],
    'raw_images': raw.label.value_counts().reindex(LABELS),
    'clean_images': images.sum(axis=1),
})
for s in SPLITS:
    table[f'{s}_images'] = images[s]
for s in SPLITS:
    table[f'{s}_share'] = share[s].round(3)
for s in SPLITS:
    table[f'{s}_groups'] = groups[s].astype(int)

total = table.select_dtypes('number').sum()
total[[f'{s}_share' for s in SPLITS]] = 1.0
table.loc['total'] = total
table.loc['total', ['source', 'group_unit']] = ['', '']
count_cols = [c for c in table.columns if c.endswith(('_images', '_groups'))]
table[count_cols] = table[count_cols].astype(int)
table.index.name = 'label'
table.to_csv(output_csv)

# Markdown version: images (share of split) and groups in one readable table
lines = [
    '| Label | Source | Raw | After cleaning | Train | Dev | Test | Groups train / dev / test |',
    '|---|---|---:|---:|---:|---:|---:|---|',
]
for label, r in table.iterrows():
    cells = [f"{int(r[f'{s}_images'])} ({r[f'{s}_share']:.1%})" for s in SPLITS]
    unit = f" {r.group_unit}s" if r.group_unit else ''
    group_cell = ' / '.join(str(int(r[f'{s}_groups'])) for s in SPLITS) + unit
    name = f'**{label}**' if label == 'total' else label
    lines.append(f"| {name} | {r.source} | {int(r.raw_images)} | {int(r.clean_images)} | "
                 + ' | '.join(cells) + f" | {group_cell} |")

majority = images['train'].idxmax()
notes = [
    '',
    'Percentages are each class\'s share of that split.',
    'Raw to cleaned: exact duplicates (identical grayscale pixels) removed; no unreadable files or failed masks.',
    'Splits are grouped (figshare by patient ID, Kaggle no-tumor by near-duplicate group) and stratified by label, seed 539.',
    f"Majority-class baseline (always predict {majority}): "
    + ', '.join(f"{s} accuracy {share.loc[majority, s]:.1%}" for s in SPLITS) + '.',
    f"Imbalance: largest/smallest class in train = {images['train'].max() / images['train'].min():.2f}x.",
]
with open(output_md, 'w', encoding='utf-8') as f:
    f.write('# Class distribution\n\n' + '\n'.join(lines + notes) + '\n')

print('\n'.join(lines + notes))
print(f"\nSaved {output_csv} and {output_md}")
