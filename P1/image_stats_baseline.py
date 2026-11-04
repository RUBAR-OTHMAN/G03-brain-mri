"""P1 step 6: image-stats baseline (shortcut check).

Trains simple classifiers on features that carry no tumor information, to measure how much of
the task can be solved from the image source alone (figshare tumors vs Kaggle no-tumor).
Fit on train, report on dev. The test split is never touched.

Feature sets:
  metadata      original width, height, aspect ratio, color mode, file size
                (removed by the planned preprocessing: grayscale + resize to 224)
  intensity     brightness statistics of the 224x224 grayscale image the model will see
  all           metadata + intensity
  intensity_eq  same statistics after per-image histogram equalization (a candidate mitigation)

Tasks: 4-class label; binary tumor vs no-tumor (= figshare vs Kaggle in this dataset);
       3-class tumors only (all figshare, so source cannot help; what is left is e.g. view mix or
       real brightness differences between tumor types)

Outputs: image_stats_features.csv, baseline_results.csv, baseline_results.md
Run after make_splits.py.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

project_root = os.path.dirname(os.path.abspath(__file__))
splits_csv = os.path.join(project_root, 'splits.csv')
features_csv = os.path.join(project_root, 'image_stats_features.csv')
results_csv = os.path.join(project_root, 'baseline_results.csv')
results_md = os.path.join(project_root, 'baseline_results.md')

IMG_SIZE = 224
SEED = 539
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def intensity_stats(arr, prefix):
    arr = arr.astype(np.float64)
    stats = {f'{prefix}mean': arr.mean(), f'{prefix}std': arr.std(),
             f'{prefix}frac_dark': (arr < 10).mean(), f'{prefix}frac_bright': (arr > 200).mean()}
    for p, v in zip(PERCENTILES, np.percentile(arr, PERCENTILES)):
        stats[f'{prefix}p{p}'] = v
    return stats


df = pd.read_csv(splits_csv, dtype={'patient_id': str}, keep_default_na=False)
df = df[df.split.isin(['train', 'dev'])].reset_index(drop=True)   # test stays held out

rows = []
for row in df.itertuples():
    feats = {
        'orig_w': row.orig_w,
        'orig_h': row.orig_h,
        'aspect': row.orig_w / row.orig_h,
        'was_color_file': int(row.mode != 'L'),
        'file_kb': os.path.getsize(os.path.join(project_root, row.path)) / 1024,
    }
    with Image.open(os.path.join(project_root, row.clean_path)) as im:
        small = im.convert('L').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    feats.update(intensity_stats(np.asarray(small), 'int_'))
    feats.update(intensity_stats(np.asarray(ImageOps.equalize(small)), 'eq_'))
    rows.append(feats)

features = pd.DataFrame(rows)
pd.concat([df[['path', 'label', 'source', 'split']], features], axis=1).to_csv(features_csv, index=False)

metadata_cols = ['orig_w', 'orig_h', 'aspect', 'was_color_file', 'file_kb']
intensity_cols = [c for c in features.columns if c.startswith('int_')]
FEATURE_SETS = {
    'metadata': metadata_cols,
    'intensity': intensity_cols,
    'all': metadata_cols + intensity_cols,
    'intensity_eq': [c for c in features.columns if c.startswith('eq_')],
}
MODELS = {
    'majority': lambda: DummyClassifier(strategy='most_frequent'),
    'logistic_regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    'decision_tree_d3': lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED),
    'random_forest': lambda: RandomForestClassifier(n_estimators=300, random_state=SEED),  # n_jobs=-1 stalls on Windows
}
all_rows = np.ones(len(df), dtype=bool)
TASKS = {   # name: (target, rows used)
    '4-class': (df.label.to_numpy(), all_rows),
    'tumor_vs_notumor': (np.where(df.label == 'notumor', 'notumor', 'tumor'), all_rows),
    '3-class_tumors_only': (df.label.to_numpy(), (df.label != 'notumor').to_numpy()),
}

train_split = (df.split == 'train').to_numpy()
dev_split = (df.split == 'dev').to_numpy()
results = []
for task, (y, rows) in TASKS.items():
    train, dev = train_split & rows, dev_split & rows
    for set_name, cols in FEATURE_SETS.items():
        if task == '3-class_tumors_only' and set_name in ('metadata', 'all'):
            continue   # metadata is constant within figshare (512x512 grayscale)
        X = features[cols].to_numpy()
        for model_name, make_model in MODELS.items():
            if model_name == 'majority' and set_name not in ('metadata', 'intensity'):
                continue   # the majority baseline ignores features; report it once per task
            if model_name == 'majority' and task != '3-class_tumors_only' and set_name != 'metadata':
                continue
            model = make_model().fit(X[train], y[train])
            pred = model.predict(X[dev])
            results.append({
                'task': task,
                'features': 'none' if model_name == 'majority' else set_name,
                'model': model_name,
                'dev_accuracy': accuracy_score(y[dev], pred),
                'dev_macro_f1': f1_score(y[dev], pred, average='macro'),
            })

res = pd.DataFrame(results)
res.to_csv(results_csv, index=False)

# Markdown: best model per task / feature set, plus the full grid
best = res.loc[res.groupby(['task', 'features']).dev_macro_f1.idxmax()]
order = {'none': 0, 'metadata': 1, 'intensity': 2, 'all': 3, 'intensity_eq': 4}
best = best.sort_values(['task', 'features'], key=lambda s: s.map(order) if s.name == 'features' else s)
lines = ['# Image-stats baseline (dev split)', '',
         'Simple classifiers on features with no tumor information. Fit on train, scored on dev; '
         'test untouched. Best model per row shown; full grid in baseline_results.csv.', '',
         '| Task | Features | Best model | Dev accuracy | Dev macro-F1 |', '|---|---|---|---:|---:|']
for r in best.itertuples():
    lines.append(f'| {r.task} | {r.features} | {r.model} | {r.dev_accuracy:.1%} | {r.dev_macro_f1:.3f} |')

# A depth-2 tree on metadata shows the rule a model could learn instead of looking at tumors
y_bin = TASKS['tumor_vs_notumor'][0]
tree = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(features.loc[train_split, metadata_cols], y_bin[train_split])
rules = export_text(tree, feature_names=metadata_cols)
tree_acc = accuracy_score(y_bin[dev_split], tree.predict(features.loc[dev_split, metadata_cols]))
lines += ['', f'Depth-2 tree on metadata, tumor vs no-tumor (dev accuracy {tree_acc:.1%}):', '', '```', rules.rstrip(), '```']

with open(results_md, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print('\n'.join(lines))
print(f"\nSaved {features_csv}, {results_csv}, {results_md}")
