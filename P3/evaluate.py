"""P3: score the saved classifiers on dev or test, one row per image.

    python evaluate.py --data-root ../../integration_project --split dev
    python evaluate.py --data-root ../../integration_project --split test     # final comparison only
    python compare_inputs.py --data-root ../../integration_project --split test

Settings (checkpoint from train_classifier.py, input):
  full               results/full_slice/best.pt on full slices
  crop_gt            results/crop/best.pt on crops around the ground-truth mask (oracle upper bound)
  crop_pred          results/crop/best.pt on crops around P2's U-Net mask, P2's decoy when it is
                     empty: the real pipeline, no ground truth, same crops as P4's pipeline
  crop_pred_p3decoy  like crop_pred, but empty masks get the P3 decoy the crop model trained on

Writes results/<split>/<setting>.csv with the label, prediction, class probabilities and
crop source of every image. Settings are fixed before test is run; test is not used to choose
anything.
"""
import argparse
import os
import sys

import pandas as pd
import torch
import torchvision

SETTINGS = {
    # name: (checkpoint folder, input, mask source, no-tumor rule)
    'full': ('full_slice', 'full', None, None),
    'crop_gt': ('crop', 'crop', 'gt', 'decoy'),
    'crop_pred': ('crop', 'crop', 'pred', 'p2_decoy'),
    'crop_pred_p3decoy': ('crop', 'crop', 'pred', 'decoy'),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_classifier(path, labels, device):
    ckpt = torch.load(path, map_location='cpu')
    if ckpt['labels'] != labels:
        raise ValueError(f"{path} was trained with labels {ckpt['labels']}, the loader has {labels}")
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), ckpt['epoch']


def make_loader(setting, split, data_root, pred_masks_path, workers):
    from brain_mri_data import make_loader as make_full_loader
    _, input_mode, masks, notumor = SETTINGS[setting]
    if input_mode == 'full':
        return make_full_loader(split, batch_size=64, root=data_root, num_workers=workers)
    from crop_data import make_crop_loader, train_tumor_box_dims
    return make_crop_loader(split, data_root, train_tumor_box_dims(data_root), 64, workers,
                            notumor=notumor, masks=masks,
                            pred_masks_path=pred_masks_path if masks == 'pred' else None)


@torch.no_grad()
def predict(model, loader, labels, device):
    rows = []
    for batch in loader:
        probs = torch.softmax(model(batch['image'].to(device)), 1).cpu()
        sources = batch.get('crop_source', ['full_slice'] * len(probs))
        for path, y, p, source in zip(batch['path'], batch['label'].tolist(), probs, sources):
            rows.append({'path': path, 'label': labels[y], 'pred': labels[int(p.argmax())],
                         'confidence': float(p.max()), 'crop_source': source,
                         **{f'p_{l}': float(v) for l, v in zip(labels, p)}})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', choices=['dev', 'test'], required=True)
    p.add_argument('--settings', nargs='+', choices=list(SETTINGS), default=list(SETTINGS))
    p.add_argument('--pred-masks', help='P2 predicted-mask archive; default ../P2/<split>_predicted_masks.npz')
    p.add_argument('--results', default=os.path.join(HERE, 'results'))
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    data_root = os.path.abspath(args.data_root)
    sys.path.append(data_root)
    from brain_mri_data import LABELS

    pred_masks = args.pred_masks or os.path.join(HERE, '..', 'P2', f'{args.split}_predicted_masks.npz')
    out_dir = os.path.join(args.results, args.split)
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device()

    for setting in args.settings:
        ckpt_path = os.path.join(args.results, SETTINGS[setting][0], 'best.pt')
        model, epoch = load_classifier(ckpt_path, LABELS, device)
        loader = make_loader(setting, args.split, data_root, pred_masks, args.workers)
        df = predict(model, loader, LABELS, device)
        df.to_csv(os.path.join(out_dir, f'{setting}.csv'), index=False)
        print(f"{args.split} {setting:18s} | {ckpt_path} (epoch {epoch}) | n {len(df)} | "
              f"acc {(df.label == df.pred).mean():.3f} | crop sources {df.crop_source.value_counts().to_dict()}")


if __name__ == '__main__':
    main()
