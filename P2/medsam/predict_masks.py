"""Generate predicted segmentation masks using trained U-Net.

Writes binary masks (0/1 uint8) keyed by relative image path into a compressed
.npz archive for each requested split.

Usage:
    python P2/predict_masks.py --splits dev test
    python P2/predict_masks.py --splits train dev test --output-dir P2/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == 'P2' or os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits.csv')) else os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

from brain_mri_data import make_loader, load_splits
from train_unet import UNet


def predict_and_save_masks(splits=('dev', 'test'), checkpoint_path='best_unet.pth', output_dir='P2', threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading U-Net from {checkpoint_path} on {device}...")

    model = UNet(in_channels=3, out_channels=1, init_features=32)
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join(PROJECT_ROOT, checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        print(f"\nProcessing split '{split}'...")
        loader = make_loader(split, batch_size=32, shuffle=False)
        masks_dict = {}

        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                paths = batch['path']
                preds = (torch.sigmoid(model(images)) > threshold).cpu().numpy().astype(np.uint8)

                for i, p in enumerate(paths):
                    masks_dict[p] = preds[i, 0]  # [224, 224]

        out_file = os.path.join(output_dir, f"{split}_predicted_masks.npz")
        np.savez_compressed(out_file, **masks_dict)
        size_mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f"Saved {len(masks_dict)} masks to {out_file} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate U-Net predicted masks for MRI splits.")
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'], help="Splits to process: train, dev, test")
    parser.add_argument('--checkpoint', default='best_unet.pth', help="Path to best_unet.pth checkpoint")
    parser.add_argument('--output-dir', default='P2', help="Output directory for .npz archives")
    parser.add_argument('--threshold', type=float, default=0.5, help="Sigmoid threshold for binary mask")
    args = parser.parse_args()

    predict_and_save_masks(splits=args.splits, checkpoint_path=args.checkpoint, output_dir=args.output_dir, threshold=args.threshold)
