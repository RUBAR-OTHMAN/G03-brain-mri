"""P3: fine-tune ResNet-50 on the 4 classes (glioma, meningioma, pituitary, notumor).

Uses P1's shared loader (brain_mri_data.py) so preprocessing and splits match the team.
Only train and dev are touched here. Test is for the final comparison.

    python train_classifier.py --data-root ../../integration_project
    # on colab: --data-root /content/drive/MyDrive/integration_project

data-root is the integration_project folder: it has splits.csv, brain_mri_data.py and the images.

Training (from the proposal): ImageNet ResNet-50, new 4-class fc head, backbone frozen for a
warm-up, then everything unfrozen and fine-tuned. Weighted cross-entropy for the 2.3x imbalance.
The checkpoint kept is the epoch with the best dev macro-F1.
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import classification_report, confusion_matrix, f1_score

p = argparse.ArgumentParser()
p.add_argument('--data-root', required=True)
p.add_argument('--out', default='results/full_slice')
p.add_argument('--warmup-epochs', type=int, default=2)
p.add_argument('--finetune-epochs', type=int, default=10)
p.add_argument('--batch-size', type=int, default=32)
p.add_argument('--head-lr', type=float, default=1e-3)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--workers', type=int, default=4)
p.add_argument('--seed', type=int, default=539)
args = p.parse_args()

data_root = os.path.abspath(args.data_root)
sys.path.append(data_root)
from brain_mri_data import LABELS, class_weights, make_loader  # noqa: E402


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def run_epoch(model, loader, loss_fn, device, opt=None):
    # opt=None means eval
    model.train(opt is not None)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(opt is not None):
        for batch in loader:
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(y)
            preds.append(out.argmax(1).cpu())
            labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        'loss': total_loss / len(labels),
        'acc': (preds == labels).mean(),
        'macro_f1': f1_score(labels, preds, average='macro'),
    }, preds, labels


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    os.makedirs(args.out, exist_ok=True)

    train_loader = make_loader('train', batch_size=args.batch_size, root=data_root,
                               num_workers=args.workers, hflip=True)
    dev_loader = make_loader('dev', batch_size=64, root=data_root, num_workers=args.workers)
    print(f"device {device} | train {len(train_loader.dataset)} | dev {len(dev_loader.dataset)}")

    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(data_root).to(device))

    # warm-up: only the new head trains
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc.')
    opt = torch.optim.Adam(model.fc.parameters(), lr=args.head_lr)

    history, best_f1 = [], -1.0
    n_epochs = args.warmup_epochs + args.finetune_epochs
    for epoch in range(1, n_epochs + 1):
        if epoch == args.warmup_epochs + 1:
            # unfreeze everything and fine-tune end to end
            for param in model.parameters():
                param.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.finetune_epochs)
        phase = 'warmup' if epoch <= args.warmup_epochs else 'finetune'

        t0 = time.time()
        tr, _, _ = run_epoch(model, train_loader, loss_fn, device, opt)
        dv, dev_preds, dev_labels = run_epoch(model, dev_loader, loss_fn, device)
        if phase == 'finetune':
            sched.step()

        history.append({'epoch': epoch, 'phase': phase,
                        **{f'train_{k}': v for k, v in tr.items()},
                        **{f'dev_{k}': v for k, v in dv.items()}})
        print(f"epoch {epoch:2d} {phase:8s} | train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"dev loss {dv['loss']:.3f} acc {dv['acc']:.3f} f1 {dv['macro_f1']:.3f} | "
              f"{time.time() - t0:.0f}s")

        if dv['macro_f1'] > best_f1:
            best_f1 = dv['macro_f1']
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'labels': LABELS, 'args': vars(args)},
                       os.path.join(args.out, 'best.pt'))
            best = (epoch, dev_preds, dev_labels)

    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(args.out, 'history.csv'), index=False)

    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ['loss', 'acc', 'macro_f1']):
        ax.plot(hist.epoch, hist[f'train_{metric}'], 'o-', label='train')
        ax.plot(hist.epoch, hist[f'dev_{metric}'], 'o-', label='dev')
        ax.axvline(args.warmup_epochs + 0.5, color='gray', ls='--', lw=1)
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.legend()
    axes[1].axhline(0.865, color='red', ls=':', lw=1, label='brightness baseline (dev)')
    axes[2].axhline(0.843, color='red', ls=':', lw=1, label='brightness baseline (dev)')
    axes[1].legend()
    axes[2].legend()
    fig.suptitle('ResNet-50 full-slice (dashed line = unfreeze backbone)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'curves.png'), dpi=120)

    # dev report for the best epoch
    best_epoch, preds, labels = best
    report = classification_report(labels, preds, target_names=LABELS, digits=3)
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=[f'true_{l}' for l in LABELS], columns=[f'pred_{l}' for l in LABELS])
    cm_df.to_csv(os.path.join(args.out, 'dev_confusion.csv'))
    per_class = f1_score(labels, preds, average=None)
    pd.DataFrame({'label': LABELS, 'dev_f1': per_class}).to_csv(
        os.path.join(args.out, 'dev_per_class_f1.csv'), index=False)

    with open(os.path.join(args.out, 'dev_report.txt'), 'w') as f:
        f.write(f"best epoch {best_epoch} (by dev macro-F1)\n\n{report}\nconfusion (rows true, cols pred)\n"
                f"{cm_df.to_string()}\n")
    print(f"\nbest epoch {best_epoch}\n{report}\n{cm_df.to_string()}")


if __name__ == '__main__':
    main()
