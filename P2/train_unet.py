"""Train U-Net on Figshare + Kaggle Brain MRI Dataset for Tumor Segmentation.

Usage:
    python train_unet.py --epochs 10 --batch-size 16 --lr 5e-4
"""

import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from brain_mri_data import make_loader, LABELS


# ==============================================================================
# 1. Model Architecture (Standard U-Net with skip connections)
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output raw logits for numerical stability with BCEWithLogitsLoss
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict([
                (f"{name}conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                (f"{name}norm1", nn.BatchNorm2d(features)),
                (f"{name}relu1", nn.ReLU(inplace=True)),
                (f"{name}conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                (f"{name}norm2", nn.BatchNorm2d(features)),
                (f"{name}relu2", nn.ReLU(inplace=True)),
            ])
        )


# ==============================================================================
# 2. Loss & Metrics
# ==============================================================================
class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Soft Dice Loss."""
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def compute_dice_score(pred_mask, true_mask, eps=1e-6):
    """Computes the Dice coefficient between binary predictions and ground truth.
    
    If both prediction and target are empty (as in no-tumor images), Dice = 1.0.
    """
    p = (pred_mask > 0.5).float().view(-1)
    t = (true_mask > 0.5).float().view(-1)
    if t.sum() == 0 and p.sum() == 0:
        return 1.0
    intersection = (p * t).sum()
    return ((2.0 * intersection + eps) / (p.sum() + t.sum() + eps)).item()


# ==============================================================================
# 3. Validation Evaluation
# ==============================================================================
def evaluate(model, loader, device):
    model.eval()
    dice_scores = []
    per_class_dice = {label: [] for label in LABELS}

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)
            labels = batch['label']

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(len(images)):
                d = compute_dice_score(preds[i, 0], targets[i, 0])
                dice_scores.append(d)
                class_name = LABELS[labels[i]]
                per_class_dice[class_name].append(d)

    mean_dice = float(np.mean(dice_scores))
    class_means = {k: float(np.mean(v)) if v else 0.0 for k, v in per_class_dice.items()}
    return mean_dice, class_means


# ==============================================================================
# 4. Training Loop
# ==============================================================================
def train(args):
    device = torch.device(
        'mps' if torch.backends.mps.is_available() and not args.cpu
        else 'cuda' if torch.cuda.is_available() and not args.cpu
        else 'cpu'
    )
    print(f"--> Training on device: {device}")

    # Load data using the shared project data loader
    print("--> Preparing data loaders...")
    train_loader = make_loader('train', batch_size=args.batch_size, shuffle=True, hflip=True, num_workers=args.num_workers)
    dev_loader = make_loader('dev', batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"    Train samples: {len(train_loader.dataset)} | Dev samples: {len(dev_loader.dataset)}")

    # Instantiate model
    model = UNet(in_channels=3, out_channels=1, init_features=args.init_features).to(device)

    # Optionally load Buda's weights for transfer learning (matching encoder/decoder blocks)
    if args.pretrained_buda:
        print("--> Transfer learning: loading initial weights from Buda U-Net...")
        try:
            buda = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                  in_channels=3, out_channels=1, init_features=32, pretrained=True)
            # Copy matching state dict keys
            buda_dict = buda.state_dict()
            model_dict = model.state_dict()
            matched = {k: v for k, v in buda_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            model.load_state_dict(model_dict)
            print(f"    Successfully transferred {len(matched)} / {len(model_dict)} weight tensors.")
        except Exception as e:
            print(f"    Could not load Buda weights ({e}); training from scratch.")

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_dev_dice = -1.0
    history = []

    print(f"\n{'='*70}")
    print(f"Starting training for {args.epochs} epochs (Batch size: {args.batch_size}, LR: {args.lr})")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        n_batches = len(train_loader)

        for step, batch in enumerate(train_loader, 1):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 50 == 0 or step == n_batches:
                print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step:03d}/{n_batches:03d}] - Loss: {running_loss / step:.4f}", end='\r')

        scheduler.step()
        train_loss = running_loss / n_batches
        elapsed = time.time() - start_time

        # Validate on dev set
        dev_dice, class_dices = evaluate(model, dev_loader, device)

        is_best = dev_dice > best_dev_dice
        if is_best:
            best_dev_dice = dev_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_dice': dev_dice,
                'class_dices': class_dices,
            }, args.save_path)
            saved_flag = " [BEST SAVED]"
        else:
            saved_flag = ""

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'dev_dice': dev_dice,
            **{f"dice_{k}": v for k, v in class_dices.items()}
        })

        print(f"\nEpoch {epoch:02d}/{args.epochs:02d} ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Dev Mean Dice: {dev_dice:.4f}{saved_flag}")
        print(f"         Per-class: Glioma: {class_dices.get('glioma', 0):.3f} | "
              f"Meningioma: {class_dices.get('meningioma', 0):.3f} | "
              f"Pituitary: {class_dices.get('pituitary', 0):.3f} | "
              f"NoTumor: {class_dices.get('notumor', 0):.3f}")

    # Save training history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("train_unet_history.csv", index=False)
    print(f"\n{'='*70}")
    print(f"Training complete! Best Dev Dice: {best_dev_dice:.4f}")
    print(f"Saved checkpoint to: {args.save_path}")
    print(f"Saved history to:    train_unet_history.csv")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train U-Net for Brain MRI Tumor Segmentation")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument('--init-features', type=int, default=32, help="U-Net base features (default: 32)")
    parser.add_argument('--num-workers', type=int, default=2, help="DataLoader worker processes (default: 2)")
    parser.add_argument('--pretrained-buda', action='store_true', default=True, help="Initialize weights from Buda U-Net")
    parser.add_argument('--cpu', action='store_true', help="Force CPU training instead of MPS/CUDA")
    parser.add_argument('--save-path', type=str, default="best_unet.pth", help="Checkpoint save path")
    args = parser.parse_args()

    train(args)
