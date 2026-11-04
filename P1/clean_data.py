"""P1 step 2: clean the dataset listed in manifest.csv.

- Logs unreadable files and leaves them out
- Converts every image to single-channel grayscale (non-L images saved as PNG under cleaned/)
- Drops exact duplicates (identical grayscale pixels), keeping the first copy
- Flags near-duplicates with a near_dup_group id, so the split step can keep each group inside
  a single split. Candidates come from a perceptual hash, then are confirmed by correlation of
  64x64 thumbnails. pHash alone is too coarse: it matches adjacent slices of the same patient.
    Kaggle <-> Kaggle: corr >= KAGGLE_MIN_CORR (Kaggle re-saves the same slice with different
      contrast/sharpening; visually confirmed down to ~0.90)
    pairs involving figshare: corr >= FIGSHARE_MIN_CORR and different patient/source
      (same-patient adjacent slices reach ~0.98 but are handled by patient-level splitting)

Outputs: manifest_clean.csv, cleaning_log.csv, near_duplicates.png (borderline pairs to inspect)
Run after build_manifest.py.
"""
import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

project_root = os.path.dirname(os.path.abspath(__file__))
manifest_csv = os.path.join(project_root, 'manifest.csv')
cleaned_folder = os.path.join(project_root, 'cleaned')
output_csv = os.path.join(project_root, 'manifest_clean.csv')
log_csv = os.path.join(project_root, 'cleaning_log.csv')
pairs_png = os.path.join(project_root, 'near_duplicates.png')

CANDIDATE_DIST = 10       # max differing pHash bits (out of 64) to compare two images closely
KAGGLE_MIN_CORR = 0.90
FIGSHARE_MIN_CORR = 0.97
THUMB = 64


def to_gray(im):
    # Palette / alpha images go through RGB first so the conversion is well defined
    if im.mode in ('P', 'RGBA', 'LA'):
        im = im.convert('RGB')
    return im.convert('L')


def is_true_color(im):
    # RGB files whose channels differ noticeably (not just grayscale saved as RGB)
    if im.mode != 'RGB':
        return False
    arr = np.asarray(im, dtype=np.int16)
    return int(np.abs(arr[..., 0] - arr[..., 1]).max()) > 10 or int(np.abs(arr[..., 1] - arr[..., 2]).max()) > 10


def phash(gray):
    # Standard pHash: 32x32 DCT, keep low-frequency 8x8, threshold at the median
    small = np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = dctn(small, norm='ortho')[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int(np.packbits(bits).view('>u8')[0])


def thumbnail(gray):
    # Zero-mean, unit-variance 64x64 vector; dot product / THUMB**2 = Pearson correlation
    small = np.asarray(gray.resize((THUMB, THUMB), Image.LANCZOS), dtype=np.float32).ravel()
    return (small - small.mean()) / (small.std() + 1e-6)


df = pd.read_csv(manifest_csv, dtype={'patient_id': str}, keep_default_na=False)
log = []
clean_paths, pixel_md5s, phashes, true_color, thumbs = [], [], [], [], []

for row in df.itertuples():
    try:
        with Image.open(os.path.join(project_root, row.path)) as im:
            im.load()
            color = is_true_color(im)
            gray = to_gray(im)
    except Exception as e:
        log.append({'path': row.path, 'action': 'dropped_unreadable', 'detail': str(e)})
        clean_paths.append(None); pixel_md5s.append(None); phashes.append(None); true_color.append(None)
        thumbs.append(None)
        continue

    if row.mode == 'L':
        clean_path = row.path
    else:
        stem = os.path.splitext(os.path.basename(row.path))[0]
        out_dir = os.path.join(cleaned_folder, f'{row.source}_{row.label}', row.orig_split or 'all')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, stem + '.png')
        gray.save(out_path)
        clean_path = os.path.relpath(out_path, project_root).replace('\\', '/')
        log.append({'path': row.path, 'action': 'converted_to_gray',
                    'detail': f'{row.mode} -> L' + (' (had real color)' if color else '')})

    pixels = np.asarray(gray)
    clean_paths.append(clean_path)
    pixel_md5s.append(hashlib.md5(pixels.tobytes() + str(pixels.shape).encode()).hexdigest())
    phashes.append(phash(gray))
    true_color.append(color)
    thumbs.append(thumbnail(gray))

df['clean_path'] = clean_paths
df['pixel_md5'] = pixel_md5s
df['phash'] = phashes
df['had_true_color'] = true_color
df['thumb'] = thumbs
df = df[df.clean_path.notna()].copy()

# Exact duplicates: keep the first copy (manifest order puts Training before Testing)
dup_mask = df.duplicated('pixel_md5', keep='first')
first_copy = df.drop_duplicates('pixel_md5').set_index('pixel_md5').path
for row in df[dup_mask].itertuples():
    log.append({'path': row.path, 'action': 'dropped_exact_duplicate',
                'detail': f'same pixels as {first_copy[row.pixel_md5]}'})
df = df[~dup_mask].reset_index(drop=True)

# Near duplicates: pHash candidates, confirmed by thumbnail correlation
hashes = df.phash.to_numpy(dtype=np.uint64)
thumbs = np.stack(df.thumb.to_numpy())
source = df.source.to_numpy()
pid = df.patient_id.to_numpy()
pairs = []   # (i, j, corr)
for i in range(len(hashes) - 1):
    dist = np.bitwise_count(hashes[i] ^ hashes[i + 1:])
    for j in np.nonzero(dist <= CANDIDATE_DIST)[0] + i + 1:
        corr = float(thumbs[i] @ thumbs[j]) / THUMB ** 2
        if source[i] == 'kaggle' and source[j] == 'kaggle':
            is_dup = corr >= KAGGLE_MIN_CORR
        else:
            is_dup = corr >= FIGSHARE_MIN_CORR and (source[i] != source[j] or pid[i] != pid[j])
        if is_dup:
            pairs.append((i, j, corr))

# Union-find to merge pairs into groups
parent = list(range(len(df)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
for i, j, _ in pairs:
    parent[find(i)] = find(j)

roots = [find(i) for i in range(len(df))]
group_sizes = pd.Series(roots).value_counts()
df['near_dup_group'] = [f'nd{r}' if group_sizes[r] > 1 else '' for r in roots]
df['phash'] = df.phash.map(lambda h: f'{h:016x}')
df.drop(columns='thumb').to_csv(output_csv, index=False)
pd.DataFrame(log).to_csv(log_csv, index=False)

# Contact sheet of the least similar confirmed pairs (most likely false positives) for a visual check
pairs.sort(key=lambda p: p[2])
shown = pairs[:12]
if shown:
    tile = 160
    sheet = Image.new('L', (tile * 2, tile * len(shown)), 255)
    for k, (i, j, _) in enumerate(shown):
        for c, idx in enumerate((i, j)):
            with Image.open(os.path.join(project_root, df.clean_path[idx])) as im:
                sheet.paste(im.convert('L').resize((tile, tile)), (c * tile, k * tile))
    sheet.save(pairs_png)

# Summary
counts = pd.DataFrame(log).action.value_counts() if log else pd.Series(dtype=int)
print(f"Kept {len(df)} images -> {output_csv}")
print(counts.to_string())
print(f"Real-color RGB files (converted anyway): {int(df.had_true_color.sum())}")
print(f"\nNear-duplicate pairs: {len(pairs)}, in {(group_sizes > 1).sum()} groups "
      f"covering {(df.near_dup_group != '').sum()} images (largest group: {group_sizes.max()})")
if pairs:
    kinds = []
    for i, j, _ in pairs:
        a, b = df.iloc[i], df.iloc[j]
        if a.source != b.source:
            kinds.append('figshare <-> kaggle')
        elif a.source == 'kaggle':
            kinds.append('kaggle Training <-> Testing' if a.orig_split != b.orig_split else f'kaggle within {a.orig_split}')
        else:
            kinds.append('figshare same patient' if a.patient_id == b.patient_id else 'figshare different patients')
    print("Pair types:", pd.Series(kinds).value_counts().to_dict())
    print(f"Least similar confirmed pairs saved to {pairs_png}")
print("\nKept per source/label:")
print(df.groupby(['source', 'label']).size().to_string())
