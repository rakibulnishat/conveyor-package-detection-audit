"""What the deployed overlay shows: the 12x20 Eigen-CAM upsampled to 1020x600 (bilinear, as camToValues/cv2.resize)
and masked to each detected box without renormalisation. Per detection: max and mean heat inside the box, and the
share of the frame's total heat that lies inside all detected boxes."""
import csv, sys
import cv2, numpy as np
ROOT, OUT = sys.argv[1:3]
sys.path.insert(0, ROOT)
from pkginspect.engine import Detector, eigencam_grid, letterbox
det = Detector()
cap = cv2.VideoCapture(ROOT + "/public/co.mp4"); rows = []; k = 0
while True:
    ok, f = cap.read()
    if not ok: break
    f = cv2.resize(f, (1020, 600)); k += 1
    r = det.detect(f)
    lb, rr, l, t = letterbox(f)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.)
    cam = eigencam_grid(det.sess.run(det.out_names, {det.inp: x})[1])
    hm = cv2.resize(cam, (1020, 600), interpolation=cv2.INTER_LINEAR)   # desktop TrackHook path
    union = np.zeros_like(hm, bool)
    for b in r["boxes"]:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]; union[y1:y2, x1:x2] = True
    share = float(hm[union].sum() / hm.sum()) if hm.sum() > 0 else float("nan")
    for b in r["boxes"]:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]; m = hm[y1:y2, x1:x2]
        rows.append(dict(frame=k, conf=b[4], box_area_frac=(x2 - x1) * (y2 - y1) / (1020 * 600), in_box_max=float(m.max()),
                         in_box_mean=float(m.mean()), frame_mean=float(hm.mean()), union_area_frac=float(union.mean()), heat_share_in_boxes=share))
with open(OUT + "/inbox_heat.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows); print(len(d)); print(d.describe().T[["mean", "50%", "min", "max"]])
print("frames:", d.frame.nunique(), "share<area:", (d.groupby("frame").first().eval("heat_share_in_boxes<union_area_frac")).sum())
