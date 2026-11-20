"""P4: Grad-CAM trust check and triage rule for the G03 brain MRI pipeline.

Pipeline for one slice:
    image -> mask (ground truth or P2 U-Net) -> crop (P2 crop_tumor_image)
          -> ResNet-50 (P3 checkpoint) -> class + confidence
          -> Grad-CAM on the crop -> pasted back to the full slice
          -> IoU(heatmap, mask) -> triage tier

Triage rule:
    High Priority : class is glioma or meningioma, confidence > 0.80 and IoU > HI_IOU
    Needs Review  : confidence < 0.60 or IoU < LO_IOU
    Routine       : everything else
The proposal used HI_IOU = 0.50 and LO_IOU = 0.30. On dev, Grad-CAM-mask IoU is low even for
correct predictions (median ~0.15), so those values send over half of all cases to review.
Tuned on dev only (see threshold_grid): HI_IOU = 0.20, LO_IOU = 0.02.
  - LO_IOU: error catch stays flat from 0.02 to 0.08, so 0.02 gives it with the fewest reviews.
  - HI_IOU: the lowest value that still puts no wrong prediction in High Priority.
Added rule: when the prediction is 'notumor' there is no tumor region, so IoU is undefined
and only the confidence test is used.

Note: crops are centred on the mask, so a heatmap near the crop centre already overlaps the
mask well. IoU here checks "did the model look at the tumor inside the crop", not "did it
find the tumor in the whole slice". Report this as a limitation.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

HIGH_PRIORITY_CLASSES = ('glioma', 'meningioma')
TIERS = ['High Priority', 'Needs Review', 'Routine']
HI_CONF, LO_CONF = 0.80, 0.60
HI_IOU, LO_IOU = 0.20, 0.02          # tuned on dev; proposal values were 0.50 / 0.30


def setup_paths(data_root, repo_root):
    """Make P1 (data root), P2 and P3 modules importable."""
    for p in (data_root, os.path.join(repo_root, 'P2'), os.path.join(repo_root, 'P3')):
        if p not in sys.path:
            sys.path.insert(0, p)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ----------------------------------------------------------------------------------------
# Triage rule and IoU (pure functions, easy to unit-test)
# ----------------------------------------------------------------------------------------
def triage(pred_label, confidence, iou, hi_conf=HI_CONF, hi_iou=HI_IOU, lo_conf=LO_CONF, lo_iou=LO_IOU):
    """Return one of TIERS. iou may be NaN (no tumor region to compare with)."""
    if pred_label == 'notumor' or iou is None or np.isnan(iou):
        return 'Needs Review' if confidence < lo_conf else 'Routine'
    if pred_label in HIGH_PRIORITY_CLASSES and confidence > hi_conf and iou > hi_iou:
        return 'High Priority'
    if confidence < lo_conf or iou < lo_iou:
        return 'Needs Review'
    return 'Routine'


def heatmap_iou(cam, mask, cam_thr=0.5):
    """IoU between the binarized heatmap (cam >= cam_thr) and the mask (> 0.5).
    cam: [H, W] in [0, 1]; mask: [H, W] or [1, H, W]. NaN if both are empty."""
    cam = torch.as_tensor(cam)
    mask = torch.as_tensor(mask).squeeze()
    a = cam >= cam_thr
    b = mask > 0.5
    union = (a | b).sum().item()
    if union == 0:
        return float('nan')
    return (a & b).sum().item() / union


def paste_heatmap(cam_crop, box, size=224):
    """Map a heatmap computed on the resized crop back to full-slice coordinates.
    box = (y1, y2, x1, x2) in the full slice, or None when the crop is the full slice."""
    if box is None:
        return cam_crop
    y1, y2, x1, x2 = box
    full = torch.zeros(size, size)
    part = F.interpolate(cam_crop[None, None], size=(y2 - y1, x2 - x1),
                         mode='bilinear', align_corners=False)[0, 0]
    full[y1:y2, x1:x2] = part
    return full


# ----------------------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on one conv layer, default ResNet-50 layer4."""

    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._save_acts)
        layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self.acts = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """x: [1, 3, H, W]. Returns (cam [H, W] in [0, 1], probs [K])."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = logits.softmax(1)[0].detach()
        if class_idx is None:
            class_idx = int(probs.argmax())
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)        # [1, C, 1, 1]
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))   # [1, 1, h, w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu(), probs.cpu()


# ----------------------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------------------
def load_classifier(ckpt_path, n_classes=4, device='cpu'):
    """P3 ResNet-50. ckpt_path=None gives an untrained model (only for testing the code)."""
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
    return model.to(device).eval()


def load_unet(ckpt_path, device='cpu'):
    """P2 U-Net (P2/train_unet.py). Checkpoint key: 'model_state_dict'."""
    from train_unet import UNet
    model = UNet(in_channels=3, out_channels=1)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


# ----------------------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------------------
class TriagePipeline:
    """mask_source: 'gt' (ground-truth masks, stand-in) or 'unet' (P2 predicted masks).
    input_mode: 'crop' (proposal pipeline) or 'full' (full-slice baseline)."""

    def __init__(self, classifier_ckpt, mask_source='gt', unet_ckpt=None, input_mode='crop',
                 cam_thr=0.5, device=None):
        from brain_mri_data import LABELS
        self.labels = LABELS
        self.device = device or pick_device()
        self.model = load_classifier(classifier_ckpt, len(LABELS), self.device)
        self.cam = GradCAM(self.model, self.model.layer4)
        self.mask_source = mask_source
        self.unet = load_unet(unet_ckpt, self.device) if mask_source == 'unet' else None
        self.input_mode = input_mode
        self.cam_thr = cam_thr

    @torch.no_grad()
    def predict_mask(self, image):
        logits = self.unet(image[None].to(self.device))
        return (torch.sigmoid(logits)[0].cpu() > 0.5).float()      # [1, H, W]

    def run_one(self, item, idx=0):
        """item: one BrainMRIDataset item. Returns a dict with every intermediate result."""
        from crop_tumor import crop_tumor_image
        image = item['image']
        mask = self.predict_mask(image) if self.mask_source == 'unet' else item['mask']

        if self.input_mode == 'crop':
            crop, box = crop_tumor_image(image, mask, decoy_seed=idx)
        else:
            crop, box = image, None

        cam_crop, probs = self.cam(crop[None].to(self.device))
        cam_full = paste_heatmap(cam_crop, box, size=image.shape[-1])
        k = int(probs.argmax())
        pred, conf = self.labels[k], float(probs[k])
        iou = float('nan') if pred == 'notumor' else heatmap_iou(cam_full, mask, self.cam_thr)
        # Also score against the ground-truth mask when the pipeline used predicted masks
        iou_gt = float('nan') if pred == 'notumor' else heatmap_iou(cam_full, item['mask'], self.cam_thr)
        return {
            'path': item['path'], 'true': self.labels[item['label']], 'pred': pred,
            'confidence': conf, 'iou': iou, 'iou_vs_gt': iou_gt,
            'tier': triage(pred, conf, iou),
            'image': image, 'mask': mask, 'crop': crop, 'box': box,
            'cam_crop': cam_crop, 'cam_full': cam_full,
        }

    def run_split(self, split, root, limit=None):
        """Run on a whole split. Returns a DataFrame (no tensors)."""
        from brain_mri_data import BrainMRIDataset
        ds = BrainMRIDataset(split, root=root)
        n = len(ds) if limit is None else min(limit, len(ds))
        rows = []
        for i in range(n):
            r = self.run_one(ds[i], idx=i)
            rows.append({k: v for k, v in r.items() if not torch.is_tensor(v) and k != 'box'})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Summaries and plots
# ----------------------------------------------------------------------------------------
def summarize(df, labels=('glioma', 'meningioma', 'pituitary', 'notumor')):
    """Tier distribution, tier by true class, IoU by class, and the key triage numbers."""
    tiers = df.tier.value_counts().reindex(TIERS, fill_value=0).to_frame('count')
    tiers['fraction'] = (tiers['count'] / len(df)).round(3)
    by_class = pd.crosstab(df.true, df.tier).reindex(index=list(labels), columns=TIERS, fill_value=0)
    iou = df.groupby('true')[['iou', 'iou_vs_gt']].agg(['mean', 'median']).round(3)
    urgent = df.true.isin(HIGH_PRIORITY_CLASSES)
    correct = df.true == df.pred
    key = {
        'n': len(df),
        'accuracy': round(correct.mean(), 3),
        'urgent cases moved to High Priority': round((df[urgent].tier == 'High Priority').mean(), 3),
        'High Priority cases that are truly urgent': round(df[df.tier == 'High Priority'].true.isin(HIGH_PRIORITY_CLASSES).mean(), 3),
        'errors caught by Needs Review': round((df[~correct].tier == 'Needs Review').mean(), 3) if (~correct).any() else None,
        'confident errors (conf > 0.8)': int(((~correct) & (df.confidence > 0.8)).sum()),
    }
    return tiers, by_class, iou, key


def threshold_grid(df, lo_values=(0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.30),
                   hi_values=(0.10, 0.15, 0.20, 0.30, 0.50)):
    """Tier outcomes for each (LO_IOU, HI_IOU) pair, from a run_split() DataFrame. Use on dev only."""
    ok = df.true == df.pred
    urgent = df.true.isin(HIGH_PRIORITY_CLASSES)
    rows = []
    for lo in lo_values:
        for hi in hi_values:
            if hi <= lo:
                continue
            t = np.array([triage(r.pred, r.confidence, r.iou, hi_iou=hi, lo_iou=lo) for r in df.itertuples()])
            rows.append({
                'lo_iou': lo, 'hi_iou': hi,
                'sent to review': (t == 'Needs Review').mean(),
                'errors caught': (t[~ok.values] == 'Needs Review').mean() if (~ok).any() else np.nan,
                'urgent -> High Priority': (t[urgent.values] == 'High Priority').mean(),
                'wrong in High Priority': int(((t == 'High Priority') & ~ok.values).sum()),
            })
    return pd.DataFrame(rows).round(3)


def retier(df, **thresholds):
    """Recompute tiers on a saved results DataFrame with new thresholds (no model run needed)."""
    out = df.copy()
    out['tier'] = [triage(r.pred, r.confidence, r.iou, **thresholds) for r in out.itertuples()]
    return out


def confusion(df, labels=('glioma', 'meningioma', 'pituitary', 'notumor')):
    return pd.crosstab(df.true, df.pred).reindex(index=list(labels), columns=list(labels), fill_value=0)


def plot_case(r, ax_row=None):
    """One row: slice | mask | crop | heatmap on slice. r is a run_one() result."""
    import matplotlib.pyplot as plt
    from brain_mri_data import denormalize
    if ax_row is None:
        _, ax_row = plt.subplots(1, 4, figsize=(13, 3.4))
    img = denormalize(r['image']).numpy()      # [H, W] grayscale
    crop = denormalize(r['crop']).numpy()
    ax_row[0].imshow(img, cmap='gray'); ax_row[0].set_title(f"input (true: {r['true']})")
    ax_row[1].imshow(img, cmap='gray'); m = r['mask'].squeeze().numpy()
    ax_row[1].imshow(np.ma.masked_where(m < 0.5, m), alpha=0.5, cmap='autumn')
    ax_row[1].set_title('mask')
    ax_row[2].imshow(crop, cmap='gray'); ax_row[2].set_title('crop -> classifier')
    ax_row[3].imshow(img, cmap='gray'); cam = r['cam_full'].numpy()
    ax_row[3].imshow(np.ma.masked_where(cam < 0.05, cam), alpha=0.5, cmap='jet', vmin=0, vmax=1)
    ax_row[3].contour(r['mask'].squeeze().numpy(), levels=[0.5], colors='white', linewidths=1)
    iou = 'n/a' if np.isnan(r['iou']) else f"{r['iou']:.2f}"
    ax_row[3].set_title(f"{r['pred']} {r['confidence']:.2f} | IoU {iou}\n-> {r['tier']}")
    for a in ax_row:
        a.axis('off')
    return ax_row
