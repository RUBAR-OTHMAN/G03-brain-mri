"""P3: full slice vs. crop comparison from evaluate.py's per-image results.

    python compare_inputs.py --data-root ../../integration_project --split dev
    python compare_inputs.py --data-root ../../integration_project --split test

Reads results/<split>/<setting>.csv and writes results/comparison_<split>.md. On dev, the full
and crop_gt rows must reproduce the confusion matrices train_classifier.py saved for the best
epoch; anything else means the checkpoints or the crop code changed.

Confidence intervals come from a patient bootstrap: whole patients (splits.csv group_id) are
resampled within each class, because images of one patient are not independent and test has
only 9-14 patients per tumor class. All settings are scored on the same resamples, so the
differences against the full slice are paired.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {
    'full': 'full slice',
    'crop_gt': 'crop, true masks (oracle)',
    'crop_pred': 'crop, U-Net masks (pipeline)',
    'crop_pred_p3decoy': 'crop, U-Net masks, P3 decoy',
}
TRAINING_CONFUSION = {'full': 'full_slice', 'crop_gt': 'crop'}
N_BOOT = 2000


def confusion(df, labels):
    """[len(labels), len(labels)] counts, rows true, columns predicted."""
    idx = {l: k for k, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(cm, (df.label.map(idx), df.pred.map(idx)), 1)
    return cm


def per_class(cm):
    """Precision, recall and F1 per class from confusion matrices [..., true, pred]."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    n_true, n_pred = cm.sum(-1), cm.sum(-2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return tp / n_pred, tp / n_true, 2 * tp / (n_true + n_pred)


def macro_f1(cm):
    return per_class(cm)[2].mean(-1)


def check_against_training(results_dir, setting, cm, labels):
    path = os.path.join(results_dir, TRAINING_CONFUSION[setting], 'dev_confusion.csv')
    saved = pd.read_csv(path, index_col=0)
    if (list(saved.columns) != [f'pred_{l}' for l in labels] or list(saved.index) != [f'true_{l}' for l in labels]
            or not (saved.to_numpy() == cm).all()):
        raise ValueError(f'dev {setting} does not reproduce {path}:\n{cm}\nvs.\n{saved}\n'
                         f'The checkpoint or the crop code no longer matches the training run.')


def patient_bootstrap(cms_by_group, group_labels, rng):
    """Resampled confusion matrices [N_BOOT, settings, C, C] from per-patient matrices
    cms_by_group [groups, settings, C, C], resampling patients within each class."""
    counts = np.zeros((N_BOOT, len(group_labels)))
    for label in np.unique(group_labels):
        members = np.flatnonzero(group_labels == label)
        draws = rng.choice(members, size=(N_BOOT, len(members)))
        for b in range(N_BOOT):
            counts[b] += np.bincount(draws[b], minlength=len(group_labels))
    return np.einsum('bg,gsij->bsij', counts, cms_by_group)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', choices=['dev', 'test'], required=True)
    p.add_argument('--results', default=os.path.join(HERE, 'results'))
    args = p.parse_args()

    sys.path.append(os.path.abspath(args.data_root))
    from brain_mri_data import LABELS, load_splits

    runs = {s: pd.read_csv(os.path.join(args.results, args.split, f'{s}.csv')) for s in NAMES
            if os.path.exists(os.path.join(args.results, args.split, f'{s}.csv'))}
    if 'full' not in runs:
        raise FileNotFoundError(f'no full-slice results for {args.split}; run evaluate.py first')
    settings = list(runs)

    splits = load_splits(os.path.abspath(args.data_root))
    split_df = splits[splits.split == args.split].set_index('path')
    for s, df in runs.items():
        if sorted(df.path) != sorted(split_df.index) or (df.label.values != split_df.label[df.path].values).any():
            raise ValueError(f'{args.split}/{s}.csv does not cover the {args.split} split with its labels')
        runs[s] = df.assign(group=split_df.group_id[df.path].values).sort_values('path', ignore_index=True)

    cms = {s: confusion(df, LABELS) for s, df in runs.items()}
    if args.split == 'dev':
        for s in TRAINING_CONFUSION:
            if s in cms:
                check_against_training(args.results, s, cms[s], LABELS)

    # patient bootstrap
    groups = runs['full'].group
    group_ids = np.array(sorted(groups.unique()))
    group_labels = runs['full'].groupby('group').label.agg(set)[group_ids]
    if (group_labels.map(len) > 1).any():
        raise ValueError('a patient group has images of more than one class; the stratified bootstrap needs one')
    group_labels = group_labels.map(lambda x: next(iter(x))).to_numpy()
    cms_by_group = np.stack([[confusion(runs[s][groups == g], LABELS) for s in settings] for g in group_ids])
    boot = macro_f1(patient_bootstrap(cms_by_group, group_labels, np.random.default_rng(539)))  # [N_BOOT, settings]
    ci = lambda x: f'{np.percentile(x, 2.5):.3f} to {np.percentile(x, 97.5):.3f}'

    tumor = [l for l in LABELS if l != 'notumor']
    n = len(runs['full'])
    overall = []
    for k, s in enumerate(settings):
        df, cm = runs[s], cms[s]
        f1 = macro_f1(cm)
        row = {'input': NAMES[s], 'acc': np.trace(cm) / n, 'macro-F1': f1, 'macro-F1 95% CI': ci(boot[:, k]),
               'change vs. full': f1 - macro_f1(cms['full']),
               'change 95% CI': '' if s == 'full' else ci(boot[:, k] - boot[:, 0]),
               'errors': int(n - np.trace(cm)),
               'missed tumors': int(((df.label != 'notumor') & (df.pred == 'notumor')).sum()),
               'false alarms': int(((df.label == 'notumor') & (df.pred != 'notumor')).sum())}
        overall.append(row)
    overall = pd.DataFrame(overall)

    classes = pd.DataFrame({'class': LABELS, 'n': cms['full'].sum(1)})
    for s in settings:
        prec, rec, f1 = per_class(cms[s])
        classes[f'F1 {s}'] = f1
    recalls = pd.DataFrame({'class': LABELS, **{s: per_class(cms[s])[1] for s in settings}})
    precisions = pd.DataFrame({'class': LABELS, **{s: per_class(cms[s])[0] for s in settings}})

    n_patients = pd.Series(group_labels).value_counts().reindex(LABELS)
    sections = [
        f'# Full slice vs. crop ({args.split}, n = {n})\n',
        'ResNet-50 with the same training setup and seed; only the input differs. Checkpoints: '
        'best dev macro-F1 epoch of each train_classifier.py run. '
        + ('Test was run once with settings fixed on dev.\n' if args.split == 'test' else 'Test is not used.\n'),
        'Inputs:\n',
        '- **full slice**: the full-slice model on the whole slice.',
        '- **crop, true masks (oracle)**: the crop model on crops around the ground-truth mask. A real '
        'input has no ground truth, so this is an upper bound.',
        '- **crop, U-Net masks (pipeline)**: the crop model on crops around P2\'s U-Net mask; an empty '
        'mask gets P2\'s decoy box. These are the crops P4\'s pipeline uses, with no ground truth anywhere.',
        '- **crop, U-Net masks, P3 decoy**: as the pipeline row, but an empty mask gets the size-matched '
        'decoy the crop model trained on.\n',
        f'Patients per class: {", ".join(f"{l} {c}" for l, c in n_patients.items())} '
        '(no-tumor counts near-duplicate groups). 95% CIs: patient bootstrap, '
        f'{N_BOOT} resamples within each class, paired across inputs.\n',
        overall.to_markdown(index=False, floatfmt=('', '.3f', '.3f', '', '+.3f', '', '', '', '')) + '\n',
        'Missed tumors: glioma, meningioma or pituitary predicted as no-tumor. False alarms: no-tumor '
        'predicted as a tumor class.\n',
    ]
    if 'crop_pred' in runs:
        df = runs['crop_pred']
        no_mask = df[(df.label != 'notumor') & (df.crop_source == 'decoy')]
        fp_mask = df[(df.label == 'notumor') & (df.crop_source == 'mask')]
        n_tumor = int((df.label != 'notumor').sum())
        sections += [
            '## U-Net masks\n',
            f'- {len(no_mask)} of {n_tumor} tumor images ({len(no_mask) / n_tumor:.1%}) get an empty U-Net mask '
            f'and so a decoy crop: {", ".join(f"{l} {c}" for l, c in no_mask.label.value_counts().reindex(tumor, fill_value=0).items())}. '
            f'The pipeline classifies {int((no_mask.label == no_mask.pred).sum())} of them correctly and calls '
            f'{int((no_mask.pred == "notumor").sum())} no-tumor.',
            f'- {len(fp_mask)} of {int((df.label == "notumor").sum())} no-tumor images get a non-empty U-Net mask.',
            f'- On the {n_tumor - len(no_mask)} tumor images with a U-Net mask, accuracy is '
            f'{(df.label == df.pred)[(df.label != "notumor") & (df.crop_source == "mask")].mean():.3f}.\n',
        ]
    sections += [
        '## Per-class F1\n',
        classes.to_markdown(index=False, floatfmt='.3f') + '\n',
        '## Per-class recall\n',
        recalls.to_markdown(index=False, floatfmt='.3f') + '\n',
        '## Per-class precision\n',
        precisions.to_markdown(index=False, floatfmt='.3f') + '\n',
    ]
    for s in settings:
        sections += [f'## Confusion, {NAMES[s]} (rows true, columns predicted)\n',
                     pd.DataFrame(cms[s], index=LABELS, columns=LABELS).rename_axis('true \\ pred').to_markdown() + '\n']

    text = '\n'.join(sections)
    out = os.path.join(args.results, f'comparison_{args.split}.md')
    with open(out, 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
