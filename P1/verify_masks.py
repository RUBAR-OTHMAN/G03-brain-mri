"""P1 step 3: verify every figshare image-mask pair.

Per pair checks:
- mask file readable, same size as the image, strictly binary (0/255)
- mask not empty; area fraction not an outlier
- mask lies inside the head (filled foreground of the slice), not on the black background.
  Darkness alone is not used: gliomas often have dark necrotic/cystic cores
- mask agrees with the manually drawn tumorBorder stored in the .mat file
  (border points that fall on/near the mask; a couple of stray points are allowed, since
  small outlines have only ~8-15 points and thin/concave edges rasterize loosely)

Outputs: mask_check.csv (one row per figshare image, with an `issues` column),
         mask_overlays.png (20 overlays, stratified by label, plus any flagged pairs)
Run after clean_data.py.
"""
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

project_root = os.path.dirname(os.path.abspath(__file__))
manifest_csv = os.path.join(project_root, 'manifest_clean.csv')
output_csv = os.path.join(project_root, 'mask_check.csv')
overlay_png = os.path.join(project_root, 'mask_overlays.png')

MIN_AREA_FRAC = 0.0005      # tumor smaller than 0.05% of the slice is suspicious (smallest real one ~0.1%)
MAX_AREA_FRAC = 0.25        # tumor larger than 25% of the slice is suspicious
MAX_OUTSIDE_HEAD_FRAC = 0.05  # at most 5% of mask pixels may fall outside the head
BACKGROUND_LEVEL = 10       # pixel value (0-255) below which we call it background
MAX_BORDER_MISSES = 2       # tumorBorder points allowed further than BORDER_TOL px from the mask
                            # (the 7 pairs with exactly 2 misses were inspected: points on thin/concave edges)
BORDER_TOL = 3
N_OVERLAYS = 20
SEED = 0


def mat_path_for(img_rel_path):
    # .../brainTumorDataPublic_1-766_jpeg/images/1/12.jpg -> .../brainTumorDataPublic_1-766/12.mat
    parts = img_rel_path.split('/')
    batch = parts[-4][:-len('_jpeg')]
    stem = os.path.splitext(parts[-1])[0]
    return os.path.join(project_root, *parts[:-4], batch, stem + '.mat')


def head_region(img):
    # Largest bright blob with interior holes (dark ventricles, necrotic cores) filled in
    labeled, n = ndimage.label(ndimage.binary_closing(img >= BACKGROUND_LEVEL, iterations=3))
    if n == 0:
        return np.zeros_like(img, dtype=bool)
    largest = np.argmax(ndimage.sum(np.ones_like(img), labeled, range(1, n + 1))) + 1
    return ndimage.binary_fill_holes(labeled == largest)


def border_agreement(border, mask, swap):
    # border is [x1, y1, x2, y2, ...]; swap=False treats x as column, swap=True treats x as row
    if border.size < 2:
        return np.nan
    pts = np.round(border.reshape(-1, 2)).astype(int)
    cols, rows = (pts[:, 1], pts[:, 0]) if swap else (pts[:, 0], pts[:, 1])
    near = ndimage.binary_dilation(mask, iterations=BORDER_TOL)
    inside = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    hits = np.zeros(len(pts), dtype=bool)
    hits[inside] = near[rows[inside], cols[inside]]
    return float(hits.mean())


df = pd.read_csv(manifest_csv, dtype={'patient_id': str}, keep_default_na=False)
fig = df[df.source == 'figshare'].reset_index(drop=True)

records = []
for row in fig.itertuples():
    rec = {'path': row.path, 'mask_path': row.mask_path, 'label': row.label, 'patient_id': row.patient_id}
    issues = []
    try:
        img = np.asarray(Image.open(os.path.join(project_root, row.clean_path)).convert('L'))
        mask_raw = np.asarray(Image.open(os.path.join(project_root, row.mask_path)))
    except Exception as e:
        rec['issues'] = f'unreadable: {e}'
        records.append(rec)
        continue

    rec['img_h'], rec['img_w'] = img.shape
    rec['mask_h'], rec['mask_w'] = mask_raw.shape[:2]
    if mask_raw.shape[:2] != img.shape:
        issues.append('size_mismatch')

    values = np.unique(mask_raw)
    rec['mask_values'] = ' '.join(map(str, values))
    if not set(values.tolist()) <= {0, 255}:
        issues.append('not_binary')
    mask = mask_raw > 127

    area = int(mask.sum())
    rec['mask_area_frac'] = area / mask.size
    rec['n_components'] = int(ndimage.label(mask)[1])
    if area == 0:
        issues.append('empty_mask')
    elif rec['mask_area_frac'] < MIN_AREA_FRAC:
        issues.append('tiny_mask')
    elif rec['mask_area_frac'] > MAX_AREA_FRAC:
        issues.append('huge_mask')

    if area and mask.shape == img.shape:
        rec['mask_background_frac'] = float((img[mask] < BACKGROUND_LEVEL).mean())
        # Contrast-enhanced tumors are usually brighter than the rest of the slice
        rec['mask_vs_slice_intensity'] = float(img[mask].mean() / max(img[img >= BACKGROUND_LEVEL].mean(), 1))
        rec['mask_outside_head_frac'] = float((~head_region(img))[mask].mean())
        if rec['mask_outside_head_frac'] > MAX_OUTSIDE_HEAD_FRAC:
            issues.append('mask_outside_head')

    with h5py.File(mat_path_for(row.path), 'r') as f:
        border = np.array(f['cjdata']['tumorBorder']).flatten()
    rec['n_border_points'] = border.size // 2
    if area:
        rec['border_agreement'] = border_agreement(border, mask, swap=False)
        rec['border_agreement_swapped'] = border_agreement(border, mask, swap=True)

    rec['issues'] = ';'.join(issues)
    records.append(rec)

check = pd.DataFrame(records)

# Decide the coordinate convention from the data, then flag disagreements
use_swapped = check.border_agreement_swapped.median() > check.border_agreement.median()
agree_col = 'border_agreement_swapped' if use_swapped else 'border_agreement'
check['border_misses'] = ((1 - check[agree_col]) * check.n_border_points).round()
low = check.border_misses > MAX_BORDER_MISSES
check.loc[low, 'issues'] = check.loc[low, 'issues'].map(lambda s: (s + ';' if s else '') + 'border_disagrees')
check['mask_ok'] = check.issues == ''
check.to_csv(output_csv, index=False)

# Overlay grid: stratified random sample plus any flagged pairs, red = mask outline
rng = np.random.default_rng(SEED)
per_label = N_OVERLAYS // check.label.nunique()
picks = [rng.choice(g.index.to_numpy(), per_label, replace=False) for _, g in check[check.mask_ok].groupby('label')]
idx = list(np.concatenate(picks))
idx = idx + [i for i in check.index[~check.mask_ok] if i not in idx][:8]
tile, cols = 192, 5
sheet = Image.new('RGB', (tile * cols, tile * ((len(idx) + cols - 1) // cols)), 'white')
for k, i in enumerate(idx):
    r = check.loc[i]
    img = Image.open(os.path.join(project_root, fig.clean_path[i])).convert('L').resize((tile, tile))
    mask = np.asarray(Image.open(os.path.join(project_root, r.mask_path)).convert('L').resize((tile, tile))) > 127
    edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
    rgb = np.stack([np.asarray(img)] * 3, axis=-1)
    rgb[edge] = (255, 0, 0) if r.mask_ok else (255, 200, 0)   # yellow outline = flagged pair
    sheet.paste(Image.fromarray(rgb), ((k % cols) * tile, (k // cols) * tile))
sheet.save(overlay_png)

# Summary
print(f"Checked {len(check)} figshare image-mask pairs -> {output_csv}")
print(f"tumorBorder convention: first coordinate is the {'row' if use_swapped else 'column'} "
      f"(median agreement {check[agree_col].median():.3f} vs "
      f"{check['border_agreement' if use_swapped else 'border_agreement_swapped'].median():.3f} the other way)")
print(f"Pairs passing all checks: {int(check.mask_ok.sum())} / {len(check)}")
issue_counts = check.issues[check.issues != ''].str.split(';').explode().value_counts()
print("Issues:", issue_counts.to_dict() if len(issue_counts) else 'none')
print("\nMask area as % of slice, per label:")
print((check.groupby('label').mask_area_frac.describe(percentiles=[.05, .5, .95]) * 100)
      .drop(columns='count').round(2).to_string())
print("\nConnected components per mask:", check.n_components.value_counts().sort_index().to_dict())
print(f"Mask pixels outside the head: median {check.mask_outside_head_frac.median():.3%}, "
      f"max {check.mask_outside_head_frac.max():.1%}")
print(f"Mask pixels darker than {BACKGROUND_LEVEL}/255 (e.g. necrotic cores): "
      f"{(check.mask_background_frac > 0.10).sum()} masks have >10%")
print("tumorBorder points missing the mask:", check.border_misses.value_counts().sort_index().to_dict())
print(f"Tumor / slice brightness ratio: median {check.mask_vs_slice_intensity.median():.2f}")
print(f"Overlays saved to {overlay_png}")
