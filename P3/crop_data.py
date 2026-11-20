"""P3 crop inputs: every class goes through P2's tumor crop (P2/crop_tumor.py).

Tumor images are cropped around their ground-truth mask, so these are oracle crops: an upper
bound until P2's segmenter produces masks without ground truth.

No-tumor images have no mask. Returning their full slice (P2's default) would make scale alone
separate tumor from no-tumor, so each one gets a decoy box instead: the height and width of a
random train tumor's mask box, centred on a random point inside the head. The decoy goes through
the same crop_tumor_image call, so every class gets the same padding, minimum size and zoom.
Dev and test decoys are fixed per image; train decoys are redrawn every epoch.

Import after the data root (with brain_mri_data.py) is on sys.path.
"""
import os
import sys

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from brain_mri_data import IMG_SIZE, BrainMRIDataset, denormalize, load_splits, preprocess_mask
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'P2'))
from crop_tumor import crop_tumor_image  # noqa: E402

HEAD_THRESHOLD = 0.1   # denormalized intensity above which a pixel counts as head, not background


def mask_box_dims(mask):
    """(height, width) of the box around all positive pixels of a [1, H, W] mask."""
    ys, xs = np.nonzero(mask[0].numpy() > 0.5)
    return ys.max() - ys.min() + 1, xs.max() - xs.min() + 1


def train_tumor_box_dims(root):
    """Box (height, width) of every train tumor mask, in the 224x224 model space."""
    df = load_splits(root)
    rows = df[(df.split == 'train') & (df.mask_path != '')]
    dims = []
    for mask_path in rows.mask_path:
        with Image.open(os.path.join(root, mask_path)) as m:
            dims.append(mask_box_dims(preprocess_mask(m)))
    return np.array(dims)


def decoy_mask(image, box_hw, rng):
    """Rectangle mask of size box_hw centred on a random point inside the head."""
    head = ndimage.binary_fill_holes(denormalize(image).numpy() > HEAD_THRESHOLD)
    if not head.any():
        raise ValueError('no head pixels above HEAD_THRESHOLD; check the image preprocessing')
    h, w = box_hw
    inner = ndimage.binary_erosion(head, iterations=int(max(h, w)) // 2)
    centres = np.argwhere(inner if inner.any() else head)   # small heads: centre anywhere in the head
    cy, cx = centres[rng.integers(len(centres))]
    y1, x1 = max(0, cy - h // 2), max(0, cx - w // 2)
    mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)
    mask[0, y1:min(IMG_SIZE, y1 + h), x1:min(IMG_SIZE, x1 + w)] = 1
    return mask


class CropDataset(Dataset):
    """BrainMRIDataset items with 'image' replaced by a crop and 'crop_mask' set to the box source.

    notumor='decoy' (default) uses a size-matched decoy box; 'full_slice' keeps P2's behaviour
    (used only by the shortcut check to show why it is not used for training).
    """

    def __init__(self, split, root, box_dims, hflip=False, notumor='decoy', seed=539):
        if notumor not in ('decoy', 'full_slice'):
            raise ValueError(f'notumor must be decoy or full_slice, not {notumor!r}')
        self.base = BrainMRIDataset(split, root=root, hflip=hflip)
        self.box_dims = box_dims
        self.redraw = split == 'train'
        self.notumor = notumor
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        if item['has_mask']:
            mask = item['mask']
        elif self.notumor == 'decoy':
            # torch's per-worker seed makes train draws reproducible; dev/test are fixed per index
            seed = torch.randint(2**31, (1,)).item() if self.redraw else (self.seed, i)
            rng = np.random.default_rng(seed)
            mask = decoy_mask(item['image'], self.box_dims[rng.integers(len(self.box_dims))], rng)
        else:
            mask = None
        crop, box = crop_tumor_image(item['image'], mask)
        return {**item, 'image': crop, 'full_image': item['image'],
                'crop_box': torch.tensor(box if box is not None else (-1, -1, -1, -1))}


def make_crop_loader(split, root, box_dims, batch_size=32, num_workers=0, hflip=False, notumor='decoy'):
    dataset = CropDataset(split, root, box_dims, hflip=hflip, notumor=notumor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train', num_workers=num_workers)
