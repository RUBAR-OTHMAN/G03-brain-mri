"""P1 step 1: build manifest.csv with one row per image.

Columns: path, mask_path, label, source, patient_id, orig_split, orig_w, orig_h, mode
Paths are relative to this project folder so the CSV works on any machine / Colab.
Run after converting the figshare .mat files to JPEG (import scipy.py).
"""
import glob
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image

project_root = os.path.dirname(os.path.abspath(__file__))
figshare_folder = os.path.join(project_root, 'braintumormri_figshare')
kaggle_folder = os.path.join(project_root, 'braintumormri_kaggle')
output_csv = os.path.join(project_root, 'manifest.csv')

FIGSHARE_LABELS = {1: 'meningioma', 2: 'glioma', 3: 'pituitary'}


def image_info(path):
    # PIL only reads the header here, so this is fast
    with Image.open(path) as im:
        return im.size[0], im.size[1], im.mode


def rel(path):
    return os.path.relpath(path, project_root).replace('\\', '/')


rows = []
missing = []

# Figshare: label + patient ID come from the .mat, pixels from the converted JPEG/PNG
for mat_path in sorted(glob.glob(os.path.join(figshare_folder, 'brainTumorDataPublic_*', '*.mat'))):
    with h5py.File(mat_path, 'r') as f:
        cjdata = f['cjdata']
        label_id = int(np.array(cjdata['label'])[0, 0])
        # PID is stored as an array of character codes
        pid = ''.join(chr(c) for c in np.array(cjdata['PID']).flatten())

    batch_dir = os.path.dirname(mat_path) + '_jpeg'
    stem = os.path.splitext(os.path.basename(mat_path))[0]
    img_path = os.path.join(batch_dir, 'images', str(label_id), stem + '.jpg')
    mask_path = os.path.join(batch_dir, 'masks', str(label_id), stem + '_mask.png')

    if not (os.path.exists(img_path) and os.path.exists(mask_path)):
        missing.append(mat_path)
        continue

    w, h, mode = image_info(img_path)
    rows.append({
        'path': rel(img_path),
        'mask_path': rel(mask_path),
        'label': FIGSHARE_LABELS[label_id],
        'source': 'figshare',
        'patient_id': pid,
        'orig_split': '',          # figshare has no predefined split
        'orig_w': w,
        'orig_h': h,
        'mode': mode,
    })

# Kaggle: only the no-tumor class (its tumor classes overlap with figshare)
for orig_split in ['Training', 'Testing']:
    for img_path in sorted(glob.glob(os.path.join(kaggle_folder, orig_split, 'notumor', '*'))):
        w, h, mode = image_info(img_path)
        rows.append({
            'path': rel(img_path),
            'mask_path': '',           # no tumor, no mask
            'label': 'notumor',
            'source': 'kaggle',
            'patient_id': '',          # Kaggle has no patient IDs
            'orig_split': orig_split,
            'orig_w': w,
            'orig_h': h,
            'mode': mode,
        })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print(f"Wrote {len(df)} rows to {output_csv}")
if missing:
    print(f"WARNING: {len(missing)} .mat files had no converted JPEG/mask, e.g. {missing[:3]}")
print("\nImages per label/source:")
print(df.groupby(['source', 'label']).size().to_string())
print("\nFigshare patients per label:")
print(df[df.source == 'figshare'].groupby('label').patient_id.nunique().to_string())
print("\nImage sizes per source (top 5):")
for source, group in df.groupby('source'):
    sizes = (group.orig_w.astype(str) + 'x' + group.orig_h.astype(str)).value_counts()
    print(f"  {source}: {len(sizes)} distinct sizes; {sizes.head(5).to_dict()}")
print("\nColor modes per source:")
print(df.groupby(['source', 'mode']).size().to_string())
