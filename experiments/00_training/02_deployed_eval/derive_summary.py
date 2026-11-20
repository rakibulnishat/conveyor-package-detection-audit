"""Numbers quoted in the paper for the deployed checkpoint, its training log and the dataset audit.

Run from the repository root:  python experiments/02_deployed_eval/derive_summary.py [out.json]
Reads experiments/results/{deployed_eval/summary.json, deployed_eval/train_results.csv,
dataset/nearest_train_neighbour.csv, leakage/test_nn_by_condition.csv}; writes deployed_eval/derived.json.
Counts at tau = 0.25 are the Ultralytics confusion-matrix counts (IoU > 0.45 in Ultralytics 8.4.160).
"""
import json
import math
import re
import sys

import numpy as np
import pandas as pd

R = 'experiments/results'
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{R}/deployed_eval/derived.json'


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


S = json.load(open(f'{R}/deployed_eval/summary.json'))
out = {}
for sp in ('val', 'test'):
    (tp, fp), (fn, _) = S[sp]['confusion_matrix']
    tp, fp, fn = int(tp), int(fp), int(fn)
    out[sp] = dict(tp=tp, fp=fp, fn=fn, recall_025=tp / (tp + fn), recall_ci=wilson(tp, tp + fn),
                   precision_025=tp / (tp + fp), precision_ci=wilson(tp, tp + fp),
                   ultra_P=S[sp]['precision'], ultra_R=S[sp]['recall'], mAP50=S[sp]['mAP50'], mAP50_95=S[sp]['mAP50_95'])

fno = lambda s: int(re.search(r'frame_(\d+)', s).group(1))
N = pd.read_csv(f'{R}/dataset/nearest_train_neighbour.csv')
N['off'] = (N.file.map(fno) - N.nn_train.map(fno)).abs()
out['nn'] = {s: dict(n=int(len(g)), mae_median=float(round(g.mae.median(), 5)), mae_min=float(g.mae.min()), mae_max=float(g.mae.max()),
                     off1=int((g.off == 1).sum()), off2=int((g.off == 2).sum())) for s, g in N.groupby('split')}
out['nn'] = {k: out['nn'][k] for k in ('valid', 'test')}

T = pd.read_csv(f'{R}/leakage/test_nn_by_condition.csv')
out['leak_nn'] = {c: dict(mae_median=float(round(g.mae.median(), 5)), mae_min=float(g.mae.min()), mae_max=float(g.mae.max()),
                          off_median=float(g.offset.median()), off_min=int(g.offset.min()), off_max=int(g.offset.max()))
                  for c, g in T.groupby('condition', sort=False)}

L = pd.read_csv(f'{R}/deployed_eval/train_results.csv')
m = 'metrics/mAP50-95(B)'
out['trainlog'] = dict(best_epoch=int(L.loc[L[m].idxmax(), 'epoch']), best_map5095=float(L[m].max()),
                       nan_epochs=L[L['val/box_loss'].isna()].epoch.astype(int).tolist(),
                       zero_metric_epochs=L[L[m] == 0].epoch.astype(int).tolist(),
                       val_cls_ep2=float(L.loc[L.epoch == 2, 'val/cls_loss'].iloc[0]), time_s=float(L.time.iloc[-1]))
json.dump(out, open(OUT, 'w'))
print(json.dumps(out, indent=1))
