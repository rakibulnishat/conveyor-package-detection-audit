"""Full-frame Eigen-CAM (deployed ONNX path) for two clip frames, with detected boxes (what the operator's overlay keeps)
and ground-truth boxes. Frames chosen by rule: lowest and median share of heat inside the detected boxes
(exp/out_inbox/inbox_heat.csv). cam_example_low.png (a), cam_example_median.png (b)."""
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "experiments/figures")
sys.path.insert(0, "apps/web")
from figstyle import save  # noqa: E402
from pkginspect.engine import Detector, eigencam_grid, letterbox  # noqa: E402

h = pd.read_csv("experiments/results/eigencam_inbox/inbox_heat.csv").groupby("frame").first()
low = int(h.heat_share_in_boxes.idxmin())
med = int((h.heat_share_in_boxes - h.heat_share_in_boxes.median()).abs().idxmin())
gt = pd.read_csv("experiments/results/dataset/gt_boxes_by_frame.csv")
det = Detector()
cap = cv2.VideoCapture("apps/web/public/co.mp4")
frames = {}
k = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    k += 1
    if k in (low, med):
        frames[k] = cv2.resize(f, (1020, 600))
for k, name in ((low, "cam_example_low.png"), (med, "cam_example_median.png")):
    f = frames[k]
    r = det.detect(f)
    lb, rr, l, t = letterbox(f)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.)
    cam = eigencam_grid(det.sess.run(det.out_names, {det.inp: x})[1])
    hm = cv2.resize(cam, (1020, 600), interpolation=cv2.INTER_LINEAR)
    fig = plt.figure(figsize=(3.3, 3.3 * 600 / 1020))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(f[:, :, ::-1])
    ax.imshow(hm, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    for b in r["boxes"]:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="white", lw=1.6))
    for g in gt[gt.frame_no == k].itertuples():
        x1, y1 = (g.cx - g.w / 2) * 1020, (g.cy - g.h / 2) * 600
        ax.add_patch(plt.Rectangle((x1, y1), g.w * 1020, g.h * 600, fill=False, ec="black", lw=1.0, ls=(0, (3, 2))))
    ax.axvline(418, color="#1baf7a", lw=1.0)
    ax.set_xlim(0, 1020)
    ax.set_ylim(600, 0)
    ax.axis("off")
    print(k, name, "share", round(float(h.loc[k, "heat_share_in_boxes"]), 3), "area", round(float(h.loc[k, "union_area_frac"]), 3))
    save(fig, "figures/" + name)
