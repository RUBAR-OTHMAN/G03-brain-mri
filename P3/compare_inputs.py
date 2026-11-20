"""P3: full slice vs. crop comparison on dev, from the saved train_classifier.py results.

    python compare_inputs.py

Reads dev_confusion.csv and history.csv from results/full_slice and results/crop and writes
results/comparison.md. Both runs use the same model, training setup and seed; only the input
differs. The crop run uses ground-truth masks (oracle crops), so its numbers are likely
optimistic until P2's predicted masks replace them. Test is not used.
"""
import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--full', default='results/full_slice')
p.add_argument('--crop', default='results/crop')
p.add_argument('--out', default='results/comparison.md')
args = p.parse_args()


def load_run(folder):
    """Best-epoch dev confusion matrix and history row of one train_classifier.py run.

    Raises if the confusion matrix doesn't match the best history row, which means the
    files come from different runs.
    """
    cm = pd.read_csv(os.path.join(folder, 'dev_confusion.csv'), index_col=0)
    labels = [c.removeprefix('pred_') for c in cm.columns]
    if list(cm.index) != [f'true_{l}' for l in labels]:
        raise ValueError(f'{folder}/dev_confusion.csv: rows {list(cm.index)} do not match columns {list(cm.columns)}')
    cm = cm.to_numpy()

    hist = pd.read_csv(os.path.join(folder, 'history.csv'))
    best = hist.loc[hist.dev_macro_f1.idxmax()]   # first max, same rule as the checkpoint in training
    stats = per_class(cm)
    acc, macro_f1 = np.trace(cm) / cm.sum(), stats.f1.mean()
    if not np.isclose([acc, macro_f1], [best.dev_acc, best.dev_macro_f1]).all():
        raise ValueError(f'{folder}: dev_confusion.csv gives acc {acc:.4f} / macro-F1 {macro_f1:.4f} but the best '
                         f'history.csv epoch ({int(best.epoch)}) has {best.dev_acc:.4f} / {best.dev_macro_f1:.4f}. '
                         f'Rerun train_classifier.py for this input.')
    return {'labels': labels, 'cm': cm, 'stats': stats, 'acc': acc, 'macro_f1': macro_f1,
            'best_epoch': int(best.epoch), 'epochs': len(hist)}


def per_class(cm):
    """Precision, recall, F1 and support per class from a confusion matrix (rows true, cols pred)."""
    tp, n_true, n_pred = np.diag(cm), cm.sum(1), cm.sum(0)
    return pd.DataFrame({'precision': tp / n_pred, 'recall': tp / n_true,
                         'f1': 2 * tp / (n_true + n_pred), 'n': n_true})


def confusion_table(run):
    return pd.DataFrame(run['cm'], index=run['labels'], columns=run['labels']).rename_axis('true \\ pred')


def main():
    full, crop = load_run(args.full), load_run(args.crop)
    if full['labels'] != crop['labels'] or (full['stats'].n != crop['stats'].n).any():
        raise ValueError('the two runs were scored on different dev labels or class counts')
    labels, n_dev = full['labels'], int(full['cm'].sum())

    overall = pd.DataFrame([
        {'input': name, 'best epoch': f"{run['best_epoch']} / {run['epochs']}", 'dev acc': run['acc'],
         'dev macro-F1': run['macro_f1'], 'dev errors': int(n_dev - np.trace(run['cm']))}
        for name, run in [('full slice', full), ('oracle crop', crop)]])

    classes = pd.DataFrame({
        'class': labels,
        'n': full['stats'].n,
        'F1 full': full['stats'].f1,
        'F1 crop': crop['stats'].f1,
        'F1 change': crop['stats'].f1 - full['stats'].f1,
        'recall full': full['stats'].recall,
        'recall crop': crop['stats'].recall,
        'precision full': full['stats'].precision,
        'precision crop': crop['stats'].precision,
    })

    sections = [
        f'# Full slice vs. oracle crop (dev, n = {n_dev})\n',
        'ResNet-50, same training setup and seed; only the input differs. Each run is scored at its '
        'best dev macro-F1 epoch. Crops use ground-truth masks, so the crop numbers are likely optimistic. '
        'Test is not used.\n',
        overall.to_markdown(index=False, floatfmt='.3f') + '\n',
        '## Per class\n',
        classes.to_markdown(index=False, floatfmt=('', '', '.3f', '.3f', '+.3f', '.3f', '.3f', '.3f', '.3f')) + '\n',
        '## Confusion, full slice (rows true, cols predicted)\n',
        confusion_table(full).to_markdown() + '\n',
        '## Confusion, oracle crop (rows true, cols predicted)\n',
        confusion_table(crop).to_markdown() + '\n',
    ]
    text = '\n'.join(sections)
    with open(args.out, 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
