"""P1 step 7: shared preprocessing and PyTorch data loading for the whole team.

Usage (from the integration_project folder, or pass root= to point at it, e.g. on Colab):

    from brain_mri_data import BrainMRIDataset, make_loader, class_weights, LABELS

    train_loader = make_loader('train', batch_size=32)          # shuffled
    dev_loader = make_loader('dev', batch_size=32)
    batch = next(iter(train_loader))
    batch['image']     # float32 [B, 3, 224, 224], ImageNet-normalized (ResNet input)
    batch['mask']      # float32 [B, 1, 224, 224], 0/1, all zeros for no-tumor images
    batch['has_mask']  # bool    [B], False for no-tumor images
    batch['label']     # int64   [B], index into LABELS
    batch['path']      # list of original file paths

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights())  # optional, for the 2.3x imbalance

Preprocessing (same for every split and every teammate):
    grayscale -> pad to square with black (keeps anatomy undistorted) -> resize to 224
    -> copy to 3 channels -> ImageNet mean/std normalization
Masks get the same pad + resize (nearest neighbour), so they stay aligned with the image.

Reads splits.csv (made by make_splits.py). The test split is for final reporting only.
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = ['glioma', 'meningioma', 'pituitary', 'notumor']
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}
IMG_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_splits(root=PROJECT_ROOT):
    # dtype/keep_default_na keep patient IDs as strings and empty cells as ''
    return pd.read_csv(os.path.join(root, 'splits.csv'), dtype={'patient_id': str}, keep_default_na=False)


def pad_to_square(im):
    w, h = im.size
    if w == h:
        return im
    side = max(w, h)
    square = Image.new(im.mode, (side, side), 0)
    square.paste(im, ((side - w) // 2, (side - h) // 2))
    return square


def preprocess_image(im, keep_aspect=True, size=IMG_SIZE):
    """PIL image (any mode/size) -> normalized float tensor [3, size, size]."""
    im = im.convert('L')
    if keep_aspect:
        im = pad_to_square(im)
    im = im.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_mask(mask_im, keep_aspect=True, size=IMG_SIZE):
    """PIL mask (0/255) -> float tensor [1, size, size] of 0/1, aligned with preprocess_image."""
    mask_im = mask_im.convert('L')
    if keep_aspect:
        mask_im = pad_to_square(mask_im)
    mask_im = mask_im.resize((size, size), Image.NEAREST)
    return torch.from_numpy((np.asarray(mask_im) > 127).astype(np.float32)).unsqueeze(0)


def denormalize(x):
    """Undo ImageNet normalization for display: [3, H, W] -> [H, W] in 0..1."""
    return (x * IMAGENET_STD + IMAGENET_MEAN)[0].clamp(0, 1)


class BrainMRIDataset(Dataset):
    """Images, masks and labels for one or more splits ('train', 'dev', 'test')."""

    def __init__(self, split, root=PROJECT_ROOT, keep_aspect=True, hflip=False):
        splits = [split] if isinstance(split, str) else list(split)
        df = load_splits(root)
        self.df = df[df.split.isin(splits)].reset_index(drop=True)
        self.root = root
        self.keep_aspect = keep_aspect
        self.hflip = hflip   # random horizontal flip, applied to image and mask together

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(os.path.join(self.root, row.clean_path)) as im:
            image = preprocess_image(im, self.keep_aspect)
        if row.mask_path:
            with Image.open(os.path.join(self.root, row.mask_path)) as m:
                mask = preprocess_mask(m, self.keep_aspect)
        else:
            mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)
        if self.hflip and torch.rand(1).item() < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        return {
            'image': image,
            'mask': mask,
            'has_mask': bool(row.mask_path),
            'label': LABEL_TO_IDX[row.label],
            'path': row.path,
        }


def make_loader(split, batch_size=32, root=PROJECT_ROOT, shuffle=None, num_workers=0, **dataset_kwargs):
    """DataLoader for a split. Shuffles only 'train' unless told otherwise.
    num_workers=0 is safe everywhere; on Colab 2 is faster."""
    dataset = BrainMRIDataset(split, root=root, **dataset_kwargs)
    if shuffle is None:
        shuffle = split == 'train'
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def class_weights(root=PROJECT_ROOT):
    """Inverse-frequency weights from the train split, in LABELS order, mean 1."""
    counts = load_splits(root).query("split == 'train'").label.value_counts().reindex(LABELS)
    w = 1.0 / counts.to_numpy(dtype=np.float64)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


if __name__ == '__main__':
    # Self-test: shapes, value ranges, mask alignment, and a ResNet-50 forward pass
    import torchvision
    from scipy import ndimage

    torch.manual_seed(0)
    for split in ['train', 'dev', 'test']:
        ds = BrainMRIDataset(split)
        print(f"{split}: {len(ds)} images")

    ds = BrainMRIDataset('dev')
    batch = next(iter(make_loader('dev', batch_size=16, shuffle=True)))
    print("image", tuple(batch['image'].shape), batch['image'].dtype,
          f"range {batch['image'].min():.2f}..{batch['image'].max():.2f}")
    print("mask ", tuple(batch['mask'].shape), "values", batch['mask'].unique().tolist())
    print("label", batch['label'].tolist(), "has_mask", batch['has_mask'].tolist())
    print("class weights", {name: round(w, 3) for name, w in zip(LABELS, class_weights().tolist())})

    # Mask area after preprocessing should match the area measured on the original mask
    fig = ds.df[ds.df.mask_path != '']
    diffs = []
    for i in fig.sample(200, random_state=0).index:
        item = ds[i]
        diffs.append(abs(item['mask'].mean().item() - float(ds.df.mask_area_frac[i])))
    print(f"mask area change after resize: mean {np.mean(diffs):.5f}, max {np.max(diffs):.5f} (fraction of slice)")

    # Every no-tumor image must come back with an empty mask
    nt = ds.df.index[ds.df.label == 'notumor'][:50]
    assert all(ds[i]['mask'].sum() == 0 and not ds[i]['has_mask'] for i in nt)

    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    with torch.no_grad():
        out = model(batch['image'][:4])
    print("ResNet-50 forward:", tuple(out.shape))

    # Visual check: what the model sees, with the mask outline in red
    tiles = []
    for k in range(len(batch['image'])):
        gray = (denormalize(batch['image'][k]).numpy() * 255).astype(np.uint8)
        m = batch['mask'][k, 0].numpy() > 0.5
        edge = m & ~ndimage.binary_erosion(m, iterations=2)
        rgb = np.stack([gray] * 3, axis=-1)
        rgb[edge] = (255, 0, 0)
        tiles.append(rgb)
    rows = [np.concatenate(tiles[r * 8:(r + 1) * 8], axis=1) for r in range(len(tiles) // 8)]
    out_png = os.path.join(PROJECT_ROOT, 'preprocess_check.png')
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_png)
    print(f"Saved {out_png}")
