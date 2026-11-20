"""Derived dataset files used by the figures and the leakage read-out.

Run from the repository root:  python experiments/01_dataset/derive_dataset_files.py [out_dir]
Writes (default out_dir experiments/results/dataset):
  gt_boxes_by_frame.csv      one row per annotated box: frame number, Roboflow split, normalised cx, cy, w, h
  dataset_vs_video.csv       for every dataset image, the best-matching frame of apps/web/public/co.mp4 (grey 512x512 MAE)
  test_nn_by_condition.csv   for every leakage test frame, its nearest training image in each condition (grey 128x128 MAE)
"""
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image

SRC = 'data/roboflow'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'experiments/results/dataset'
os.makedirs(OUT, exist_ok=True)
fno = lambda p: int(re.search(r'frame_(\d+)', os.path.basename(p)).group(1))

imgs = {}
for sp in ('train', 'valid', 'test'):
    for p in glob.glob(f'{SRC}/{sp}/images/*'):
        imgs[fno(p)] = (sp, p)

# 1) boxes per frame
with open(f'{OUT}/gt_boxes_by_frame.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['frame_no', 'split', 'cx', 'cy', 'w', 'h'])
    for k in sorted(imgs):
        sp, p = imgs[k]
        lab = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        for line in open(lab):
            if line.strip():
                w.writerow([k, sp] + line.split()[1:5])


def grey(a, s=128):
    return np.asarray(Image.fromarray(a).convert('L').resize((s, s), Image.BILINEAR), np.float32) / 255.


# 2) dataset image (already 512x512, read as grey) -> video frame (resized to 512x512 with INTER_AREA, then grey)
def grey512(bgr):
    return cv2.cvtColor(cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) / 255.


cap = cv2.VideoCapture('apps/web/public/co.mp4'); V = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    V.append(grey512(fr))
V = np.stack(V)
with open(f'{OUT}/dataset_vs_video.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['frame_no', 'split', 'best_video_frame(1-based)', 'mae_best', 'mae_2nd'])
    for k in sorted(imgs):
        sp, p = imgs[k]
        mae = np.abs(V - cv2.imread(p, cv2.IMREAD_GRAYSCALE) / 255.).mean((1, 2)); o = np.argsort(mae)
        w.writerow([k, sp, int(o[0]) + 1, round(float(mae[o[0]]), 4), round(float(mae[o[1]]), 4)])

# 3) leakage test frames -> nearest training image per condition
P = json.load(open('experiments/07_leakage/preregistration.json'))
G = {k: grey(np.asarray(Image.open(imgs[k][1]).convert('RGB'))) for k in imgs}
with open('experiments/results/leakage/test_nn_by_condition.csv' if len(sys.argv) < 2 else f'{OUT}/test_nn_by_condition.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['test_frame', 'condition', 'nn_frame', 'offset', 'mae'])
    for t in P['test_frames']:
        for cond in ('leaky', 'blocked'):
            tr = sorted(P[f'{cond}_train'])
            mae = np.array([np.abs(G[k] - G[t]).mean() for k in tr]); j = int(mae.argmin())
            w.writerow([t, cond, tr[j], abs(t - tr[j]), round(float(mae[j]), 4)])
print('written to', OUT)
