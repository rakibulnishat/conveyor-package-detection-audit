"""Leakage experiment read-out: test-block metrics per condition and per physical package (recall at conf 0.25)."""
import glob, json, math, os, re
import numpy as np, pandas as pd
R = 'experiments/results/leakage'
trk = json.load(open('experiments/results/reference/pt_track.json'))['frames']


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


gt = []
for p in sorted(glob.glob('experiments/07_leakage/work/blocked/test/labels/*.txt')):
    k = int(re.search(r'frame_(\d+)', p).group(1)); stem = os.path.basename(p)[:-4]
    for line in open(p):
        c, cx, cy, w, h = map(float, line.split())
        g512 = [(cx - w / 2) * 512, (cy - h / 2) * 512, (cx + w / 2) * 512, (cy + h / 2) * 512]
        g1020 = [(cx - w / 2) * 1020, (cy - h / 2) * 600, (cx + w / 2) * 1020, (cy + h / 2) * 600]
        cand = [(iou(g1020, b[:4]), int(b[6])) for b in trk[k - 1]['boxes'] if len(b) > 6]
        best = max(cand) if cand else (0, -1)
        gt.append(dict(frame=k, stem=stem, box=g512, pkg=best[1] if best[0] >= 0.3 else -1, pkg_iou=round(best[0], 3)))
G = pd.DataFrame(gt)
out = {'test_frames': int(G.frame.nunique()), 'test_boxes': len(G), 'boxes_per_package': G.pkg.value_counts().to_dict()}
for cond in ('leaky', 'blocked'):
    res = json.load(open(f'{R}/{cond}_test_result.json'))
    P = pd.read_csv(f'{R}/{cond}_test_predictions.csv'); P = P[P.conf >= 0.25]
    hit = []
    for _, g in G.iterrows():
        cand = P[P.file.str.startswith(g.stem)]
        hit.append(any(iou(g.box, [r.x1, r.y1, r.x2, r.y2]) >= 0.5 for r in cand.itertuples()))
    G[f'hit_{cond}'] = hit
    tp = int(sum(hit)); n = len(G)
    # false positives at 0.25: predictions not matching any GT at IoU>=0.5 (greedy by confidence)
    fp = 0
    for stem, grp in P.groupby(P.file.str.replace(r'\.jpg$', '', regex=True)):
        gts = [g for g in G[G.stem == stem].box]; used = set()
        for r in grp.sort_values('conf', ascending=False).itertuples():
            m = [(iou(gb, [r.x1, r.y1, r.x2, r.y2]), j) for j, gb in enumerate(gts) if j not in used]
            if m and max(m)[0] >= 0.5: used.add(max(m)[1])
            else: fp += 1
    per = {}
    for pk, grp in G.groupby('pkg'):
        k = int(grp[f'hit_{cond}'].sum()); per[int(pk)] = dict(recall=k / len(grp), k=k, n=len(grp), ci=wilson(k, len(grp)))
    out[cond] = dict(mAP50=res['mAP50'], mAP50_95=res['mAP50_95'], P_maxF1=res['precision'], R_maxF1=res['recall'],
                     recall_025=tp / n, k=tp, n=n, recall_ci=wilson(tp, n), fp_025=fp, per_package=per)
G.to_csv(f'{R}/test_block_gt_with_hits.csv', index=False)
json.dump(out, open(f'{R}/leakage_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
