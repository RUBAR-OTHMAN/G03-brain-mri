"""P2 tumor cropping.

Takes full-slice 224x224 images and segmentation masks (ground truth or predicted)
and generates standardized 224x224 cropped tumor patches for P3's classifier.

Key design specifications addressed:
1. Multi-region masks: 35 masks have 2-3 disconnected regions. The bounding box
   is computed across ALL positive pixels in the slice.
2. Pituitary size protection: Pituitary tumors are median 0.7% of slice area.
   Proportional padding (pad_frac=0.20) and a minimum bounding box size (min_size=48 px)
   ensure small lesions retain adequate brain tissue context and avoid pixelation.
3. No-tumor handling: Kaggle no-tumor slices have no mask. Under 'decoy' mode (default),
   a random bounding box matching train tumor box dimensions is sampled inside the
   brain parenchyma so the classifier cannot exploit full-slice zoom shortcuts.
   'full_slice' and 'center_crop' remain available for baseline ablation.
4. Aspect ratio preservation: The bounding box is made square before cropping,
   preventing anisotropic anatomical distortion when resizing back to 224x224.
5. P3 Drop-in compatibility: Provides `CroppedBrainMRIDataset` and `make_cropped_loader`,
   matching P1's `brain_mri_data.make_loader` API exactly.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import brain_mri_data # in same folder as data
from brain_mri_data import (
    IMG_SIZE,
    LABELS,
    LABEL_TO_IDX,
    PROJECT_ROOT,
    BrainMRIDataset,
    class_weights,
    denormalize,
    load_splits,
)


def compute_tumor_bounding_box(mask, pad_frac=0.20, min_size=48):
    """Compute an aspect-ratio-preserving square bounding box covering all tumor regions.

    Args:
        mask: Tensor [1, H, W] or [H, W], or numpy array (binary 0/1 or boolean).
        pad_frac: Proportional padding to expand around the tumor (default 0.20 = 20%).
        min_size: Minimum width/height in pixels for the crop (default 48 px).

    Returns:
        tuple (y1, y2, x1, x2) or None if mask is empty.
    """
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 3:
            mask = mask[0]
        pos = (mask > 0.5).nonzero(as_tuple=False)
        if len(pos) == 0:
            return None
        y_min = pos[:, 0].min().item()
        y_max = pos[:, 0].max().item()
        x_min = pos[:, 1].min().item()
        x_max = pos[:, 1].max().item()
        h_img, w_img = mask.shape
    else:
        mask = np.squeeze(mask)
        pos = np.argwhere(mask > 0.5)
        if len(pos) == 0:
            return None
        y_min = pos[:, 0].min()
        y_max = pos[:, 0].max()
        x_min = pos[:, 1].min()
        x_max = pos[:, 1].max()
        h_img, w_img = mask.shape

    raw_h = y_max - y_min + 1
    raw_w = x_max - x_min + 1

    # Ensure square crop to preserve anatomical aspect ratio
    side = max(raw_h, raw_w)
    side = int(round(side * (1.0 + 2.0 * pad_frac)))
    side = max(side, min_size)
    side = min(side, min(h_img, w_img))  # cannot exceed slice dimensions

    # Center crop around the multi-region centroid
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2

    y1 = cy - side // 2
    x1 = cx - side // 2
    y2 = y1 + side
    x2 = x1 + side

    # Shift box if it spills over image boundaries
    if y1 < 0:
        y2 += -y1
        y1 = 0
    elif y2 > h_img:
        y1 -= (y2 - h_img)
        y2 = h_img

    if x1 < 0:
        x2 += -x1
        x1 = 0
    elif x2 > w_img:
        x1 -= (x2 - w_img)
        x2 = w_img

    # Final clamping
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(y1 + 1, min(y2, h_img))
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))

    return y1, y2, x1, x2


def generate_decoy_bounding_box(image, side_range=(48, 80), seed=None):
    """Generate a realistic square crop box located inside the brain parenchyma for no-tumor images.

    Prevents the classifier from exploiting zoom level / field-of-view differences
    between tumor crops and full no-tumor slices.

    Args:
        image: Tensor [3, H, W] (ImageNet-normalized or denormalized).
        side_range: tuple (min_side, max_side) matching typical tumor bounding box sizes.
        seed: optional int seed for reproducibility on dev/test splits.

    Returns:
        tuple (y1, y2, x1, x2)
    """
    gray = denormalize(image) if image.dim() == 3 and image.shape[0] == 3 else image.squeeze()
    h, w = gray.shape[-2:]

    # Brain parenchyma mask: non-background pixels (intensity > 0.08)
    brain_pixels = (gray > 0.08).nonzero(as_tuple=False)
    if len(brain_pixels) > 0:
        y_min = brain_pixels[:, 0].min().item()
        y_max = brain_pixels[:, 0].max().item()
        x_min = brain_pixels[:, 1].min().item()
        x_max = brain_pixels[:, 1].max().item()
    else:
        # Fallback to center 60% if background detection is ambiguous
        y_min, y_max = int(h * 0.2), int(h * 0.8)
        x_min, x_max = int(w * 0.2), int(w * 0.8)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    # Sample box side length matching typical tumor box sizes
    min_side, max_side = side_range
    side = int(rng.randint(min_side, max_side + 1))

    # Constrain side length to fit within detected brain bounds
    avail_h = max(1, y_max - y_min + 1)
    avail_w = max(1, x_max - x_min + 1)
    side = min(side, min(avail_h, avail_w))
    side = max(side, 32)

    # Sample top-left corner within brain area
    max_y = max(y_min, y_max - side)
    max_x = max(x_min, x_max - side)
    y1 = int(rng.randint(y_min, max_y + 1))
    x1 = int(rng.randint(x_min, max_x + 1))
    y2 = y1 + side
    x2 = x1 + side

    return y1, y2, x1, x2


def crop_tumor_image(image, mask=None, pad_frac=0.20, min_size=48, notumor_mode='decoy', target_size=IMG_SIZE, decoy_seed=None):
    """Crop tumor patch from a single 3-channel image tensor [3, H, W] and resize to target_size.

    Args:
        image: Tensor [3, H, W], typically ImageNet-normalized.
        mask: Tensor [1, H, W] or [H, W] (ground truth or predicted). If None or empty,
              notumor_mode is applied.
        pad_frac: Proportional padding fraction around tumor.
        min_size: Minimum crop size in pixels before resize.
        notumor_mode: Strategy for slices without mask ('decoy', 'full_slice', or 'center_crop').
        target_size: Output spatial resolution (default 224).
        decoy_seed: Optional seed for reproducible decoy box generation.

    Returns:
        cropped_image: Tensor [3, target_size, target_size]
        box: tuple (y1, y2, x1, x2) or None
    """
    box = None
    if mask is not None:
        box = compute_tumor_bounding_box(mask, pad_frac=pad_frac, min_size=min_size)

    if box is None:
        # No tumor present in mask (e.g. notumor class)
        if notumor_mode == 'decoy':
            box = generate_decoy_bounding_box(image, side_range=(48, 80), seed=decoy_seed)
            y1, y2, x1, x2 = box
            crop = image[:, y1:y2, x1:x2]
        elif notumor_mode == 'full_slice':
            if image.shape[-2:] == (target_size, target_size):
                return image.clone(), None
            crop = image
        elif notumor_mode == 'center_crop':
            # 60% center crop of brain parenchyma
            h, w = image.shape[-2:]
            side = int(min(h, w) * 0.60)
            y1 = (h - side) // 2
            x1 = (w - side) // 2
            crop = image[:, y1:y1 + side, x1:x1 + side]
            box = (y1, y1 + side, x1, x1 + side)
        else:
            raise ValueError(f"Unknown notumor_mode: {notumor_mode}")
    else:
        y1, y2, x1, x2 = box
        crop = image[:, y1:y2, x1:x2]

    # Resize back to standard target_size (224x224) using bilinear interpolation
    crop_4d = crop.unsqueeze(0)  # [1, 3, H_c, W_c]
    resized = F.interpolate(crop_4d, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return resized.squeeze(0), box


class CroppedBrainMRIDataset(Dataset):
    """PyTorch Dataset returning cropped tumor patches aligned with P1's BrainMRIDataset.

    Drop-in replacement for brain_mri_data.BrainMRIDataset.
    """

    def __init__(
        self,
        split,
        root=PROJECT_ROOT,
        keep_aspect=True,
        hflip=False,
        pad_frac=0.20,
        min_size=48,
        notumor_mode='decoy',
        target_size=IMG_SIZE,
    ):
        self.split = split
        self.base_ds = BrainMRIDataset(split, root=root, keep_aspect=keep_aspect, hflip=hflip)
        self.pad_frac = pad_frac
        self.min_size = min_size
        self.notumor_mode = notumor_mode
        self.target_size = target_size
        self.root = root

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        item = self.base_ds[i]
        full_image = item['image']
        mask = item['mask']

        # For dev/test, use index i as seed for deterministic decoy crops
        decoy_seed = i if self.split != 'train' else None

        cropped_img, box = crop_tumor_image(
            full_image,
            mask=mask if item['has_mask'] else None,
            pad_frac=self.pad_frac,
            min_size=self.min_size,
            notumor_mode=self.notumor_mode,
            target_size=self.target_size,
            decoy_seed=decoy_seed,
        )

        return {
            'image': cropped_img,               # [3, 224, 224] cropped & normalized
            'full_image': full_image,           # [3, 224, 224] original full slice
            'mask': mask,                       # [1, 224, 224] original mask
            'has_mask': item['has_mask'],
            'label': item['label'],
            'path': item['path'],
            'crop_box': torch.tensor(box if box is not None else [-1, -1, -1, -1], dtype=torch.int32),
        }


def make_cropped_loader(
    split,
    batch_size=32,
    root=PROJECT_ROOT,
    shuffle=None,
    num_workers=0,
    pad_frac=0.20,
    min_size=48,
    notumor_mode='decoy',
    **dataset_kwargs,
):
    """Factory creating a DataLoader of cropped tumor images for P3's classifier.

    Usage for P3:
        from crop_tumor import make_cropped_loader
        train_loader = make_cropped_loader('train', batch_size=32)
    """
    dataset = CroppedBrainMRIDataset(
        split,
        root=root,
        pad_frac=pad_frac,
        min_size=min_size,
        notumor_mode=notumor_mode,
        **dataset_kwargs,
    )
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy import ndimage

    print("Running crop_tumor verification on dev split...")
    loader = make_cropped_loader('dev', batch_size=16, shuffle=True)
    batch = next(iter(loader))

    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch full_image shape: {batch['full_image'].shape}")
    print(f"Batch labels: {[LABELS[l] for l in batch['label'][:8]]}")

    # Generate visual comparison report for verification
    n_display = min(8, len(batch['image']))
    fig, axes = plt.subplots(n_display, 3, figsize=(10, 3 * n_display))
    if n_display == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_display):
        label_name = LABELS[batch['label'][i]]
        has_mask = batch['has_mask'][i].item()
        box = batch['crop_box'][i].numpy()

        # Original slice with mask boundary
        orig_gray = denormalize(batch['full_image'][i]).numpy()
        orig_rgb = np.stack([orig_gray] * 3, axis=-1)
        if has_mask:
            m = batch['mask'][i, 0].numpy() > 0.5
            edge = m & ~ndimage.binary_erosion(m, iterations=2)
            orig_rgb[edge] = [1.0, 0.0, 0.0]

            # Draw bounding box in cyan
            if (box >= 0).all():
                y1, y2, x1, x2 = box
                orig_rgb[max(0, y1):min(224, y2), max(0, x1)] = [0.0, 1.0, 1.0]
                orig_rgb[max(0, y1):min(224, y2), max(0, min(223, x2))] = [0.0, 1.0, 1.0]
                orig_rgb[max(0, y1), max(0, x1):min(224, x2)] = [0.0, 1.0, 1.0]
                orig_rgb[max(0, min(223, y2)), max(0, x1):min(224, x2)] = [0.0, 1.0, 1.0]

        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(f"Full Slice ({label_name})", fontsize=10)
        axes[i, 0].axis('off')

        # Mask display
        mask_disp = batch['mask'][i, 0].numpy()
        axes[i, 1].imshow(mask_disp, cmap='gray')
        box_text = f"Box: {tuple(box)}" if (box >= 0).all() else "No Box (Full Slice)"
        axes[i, 1].set_title(box_text, fontsize=9)
        axes[i, 1].axis('off')

        # Cropped & resized result
        cropped_gray = denormalize(batch['image'][i]).numpy()
        axes[i, 2].imshow(cropped_gray, cmap='gray')
        axes[i, 2].set_title(f"P3 Input Crop (224x224)", fontsize=10)
        axes[i, 2].axis('off')

    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "crop_preview_verification.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visual verification to: {out_path}")
