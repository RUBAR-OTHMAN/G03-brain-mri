"""P3 crop inputs: every class goes through P2's tumor crop (P2/crop_tumor.py).

masks='gt' crops around the ground-truth mask. The classifier trains on these; on dev or test
they are oracle crops, an upper bound. masks='pred' crops around P2's U-Net mask
(P2/<split>_predicted_masks.npz) and never reads the label or the ground-truth mask, so it is
what the real pipeline sees.

An image without a mask (no-tumor, or an empty predicted mask) needs a box too. Returning its
full slice would make scale alone separate tumor from no-tumor, so notumor='decoy' gives it the
height and width of a random train tumor's mask box, centred on a random point inside the head.
The decoy goes through the same crop_tumor_image call, so every image gets the same padding,
minimum size and zoom. Dev and test decoys are fixed per image; train decoys are redrawn every
epoch. notumor='p2_decoy' uses P2's own decoy (a 48-80 px square, seeded by index, as P4's
pipeline does) and notumor='full_slice' keeps the full slice.

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


def load_predicted_masks(path, image_paths):
    """{image path: [1, H, W] float mask} from a P2 predict_masks.py archive, for these images."""
    with np.load(path) as archive:
        missing = [p for p in image_paths if p not in archive.files]
        if missing:
            raise KeyError(f'{path} has no mask for {len(missing)} of {len(image_paths)} images, '
                           f'e.g. {missing[0]}. Rerun P2/predict_masks.py for this split.')
        return {p: torch.from_numpy(archive[p]).float()[None] for p in image_paths}


class CropDataset(Dataset):
    """BrainMRIDataset items with 'image' replaced by a crop.

    Adds 'crop_source': 'mask' (cropped around the mask), 'decoy' or 'full_slice'.
    masks='pred' needs pred_masks_path, a P2 predict_masks.py archive for this split.
    """

    def __init__(self, split, root, box_dims, hflip=False, notumor='decoy', masks='gt',
                 pred_masks_path=None, seed=539):
        if notumor not in ('decoy', 'p2_decoy', 'full_slice'):
            raise ValueError(f'notumor must be decoy, p2_decoy or full_slice, not {notumor!r}')
        if masks not in ('gt', 'pred'):
            raise ValueError(f'masks must be gt or pred, not {masks!r}')
        if (masks == 'pred') != (pred_masks_path is not None):
            raise ValueError("pass pred_masks_path exactly when masks='pred'")
        if hflip and masks == 'pred':
            raise ValueError('predicted masks are not flipped; use hflip only with masks=gt')
        self.base = BrainMRIDataset(split, root=root, hflip=hflip)
        self.pred_masks = load_predicted_masks(pred_masks_path, list(self.base.df.path)) if masks == 'pred' else None
        self.box_dims = box_dims
        self.redraw = split == 'train'
        self.notumor = notumor
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        if self.pred_masks is None:
            mask = item['mask'] if item['has_mask'] else None
        else:
            mask = self.pred_masks[item['path']]   # the label and ground-truth mask are not read
            mask = mask if (mask > 0.5).any() else None   # same threshold as P2's crop box

        p2_kwargs = {}
        if mask is not None:
            source = 'mask'
        elif self.notumor == 'decoy':
            source = 'decoy'
            # torch's per-worker seed makes train draws reproducible; dev/test are fixed per index
            seed = torch.randint(2**31, (1,)).item() if self.redraw else (self.seed, i)
            rng = np.random.default_rng(seed)
            mask = decoy_mask(item['image'], self.box_dims[rng.integers(len(self.box_dims))], rng)
        elif self.notumor == 'p2_decoy':
            source = 'decoy'
            p2_kwargs = {'notumor_mode': 'decoy', 'decoy_seed': i}
        else:
            source = 'full_slice'
            p2_kwargs = {'notumor_mode': 'full_slice'}
        crop, box = crop_tumor_image(item['image'], mask, **p2_kwargs)
        return {**item, 'image': crop, 'full_image': item['image'], 'crop_source': source,
                'crop_box': torch.tensor(box if box is not None else (-1, -1, -1, -1))}


def make_crop_loader(split, root, box_dims, batch_size=32, num_workers=0, hflip=False, notumor='decoy',
                     masks='gt', pred_masks_path=None):
    dataset = CropDataset(split, root, box_dims, hflip=hflip, notumor=notumor, masks=masks,
                          pred_masks_path=pred_masks_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train', num_workers=num_workers)
