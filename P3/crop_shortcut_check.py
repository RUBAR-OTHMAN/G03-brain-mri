"""P3: does the crop input leak the label through brightness or zoom alone?

Same idea as P1's image_stats_baseline.py: a random forest on brightness statistics of exactly
what the classifier sees (no tumor shape information), fit on train, scored on dev. Run for
three inputs: the full slice, P2's crops with no-tumor kept as full slice, and P3's crops with
size-matched decoy boxes for no-tumor. Test is never touched.

    python crop_shortcut_check.py --data-root ../../integration_project

Writes results/crop/shortcut_check.md and results/crop/crop_preview.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

p = argparse.ArgumentParser()
p.add_argument('--data-root', required=True)
p.add_argument('--out', default='results/crop')
p.add_argument('--workers', type=int, default=4)
args = p.parse_args()

data_root = os.path.abspath(args.data_root)
sys.path.append(data_root)
from brain_mri_data import LABELS, denormalize, make_loader  # noqa: E402
from crop_data import make_crop_loader, train_tumor_box_dims  # noqa: E402

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
NOTUMOR = LABELS.index('notumor')


def brightness_features(images):
    """[B, 3, H, W] normalized -> [B, 11] brightness stats on the 0..255 grayscale image."""
    g = torch.stack([denormalize(x) for x in images]).flatten(1).numpy() * 255
    pct = np.percentile(g, PERCENTILES, axis=1).T
    return np.column_stack([g.mean(1), g.std(1), (g < 10).mean(1), (g > 200).mean(1), pct])


def features(loader):
    xs, ys = [], []
    for batch in loader:
        xs.append(brightness_features(batch['image']))
        ys.append(batch['label'].numpy())
    return np.concatenate(xs), np.concatenate(ys)


def score(train, dev):
    rf = RandomForestClassifier(n_estimators=300, random_state=539, n_jobs=-1).fit(*train)
    pred, y = rf.predict(dev[0]), dev[1]
    return {'dev_acc_4class': accuracy_score(y, pred),
            'dev_macro_f1_4class': f1_score(y, pred, average='macro'),
            'dev_acc_tumor_vs_notumor': accuracy_score(y == NOTUMOR, pred == NOTUMOR)}


def save_preview(loader, path, per_class=3):
    """Full slice with crop box (cyan) next to the crop the classifier sees, per_class rows each."""
    picked = {i: [] for i in range(len(LABELS))}
    for batch in loader:
        for k, label in enumerate(batch['label'].tolist()):
            if len(picked[label]) < per_class:
                picked[label].append((batch['full_image'][k], batch['image'][k], batch['crop_box'][k]))
        if all(len(v) == per_class for v in picked.values()):
            break
    fig, axes = plt.subplots(per_class, 2 * len(LABELS), figsize=(4 * len(LABELS), 2.1 * per_class))
    for label, items in picked.items():
        for r, (full, crop, (y1, y2, x1, x2)) in enumerate(items):
            ax_full, ax_crop = axes[r, 2 * label], axes[r, 2 * label + 1]
            ax_full.imshow(denormalize(full), cmap='gray')
            ax_full.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='cyan', lw=1))
            ax_crop.imshow(denormalize(crop), cmap='gray')
            for ax in (ax_full, ax_crop):
                ax.axis('off')
        axes[0, 2 * label].set_title(f'{LABELS[label]}: slice + box', fontsize=9)
        axes[0, 2 * label + 1].set_title('crop (model input)', fontsize=9)
    fig.suptitle('Dev crops. Tumor boxes come from ground-truth masks; no-tumor boxes are size-matched decoys')
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    torch.manual_seed(539)
    os.makedirs(args.out, exist_ok=True)
    box_dims = train_tumor_box_dims(data_root)
    print(f'train tumor boxes: {len(box_dims)}, median h x w {np.median(box_dims, 0)}')

    inputs = {
        'full slice (no crop)': lambda split: make_loader(split, 64, root=data_root, shuffle=False,
                                                          num_workers=args.workers),
        'crop, no-tumor = full slice (P2 default)': lambda split: make_crop_loader(
            split, data_root, box_dims, 64, args.workers, notumor='full_slice'),
        'crop, no-tumor = size-matched decoy (P3)': lambda split: make_crop_loader(
            split, data_root, box_dims, 64, args.workers, notumor='decoy'),
    }
    rows = []
    for name, loader in inputs.items():
        rows.append({'input': name, **score(features(loader('train')), features(loader('dev')))})
        print(rows[-1])

    table = pd.DataFrame(rows)
    with open(os.path.join(args.out, 'shortcut_check.md'), 'w') as f:
        f.write('Random forest on 11 brightness statistics of the model input (no tumor shape). '
                'Fit on train, scored on dev.\n\n')
        f.write(table.to_markdown(index=False, floatfmt='.3f') + '\n')
    print(table.to_string(index=False))

    save_preview(make_crop_loader('dev', data_root, box_dims, 64, args.workers),
                 os.path.join(args.out, 'crop_preview.png'))


if __name__ == '__main__':
    main()
