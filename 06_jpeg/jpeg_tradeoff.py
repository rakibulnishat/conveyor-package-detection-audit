"""JPEG quality vs payload vs agreement with the PyTorch reference (raw frames).
For q in {50,60,70,80,85,90,95,100}: payload bytes, boxes found, matched boxes, min/median IoU,
Eigen-CAM max abs diff, and the crossing events produced by the deployed browser tracker (run separately in Node).
usage: python -I jpeg_tradeoff.py <app_root> <ref_dir> <out_dir>
"""
import json
import sys

import cv2
import numpy as np

ROOT, REF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, ROOT)
from pkginspect.engine import Detector, eigencam_grid, letterbox  # noqa: E402

det = Detector()
cap = cv2.VideoCapture(ROOT + "/public/co.mp4")
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.resize(f, (1020, 600)))
ref = json.load(open(f"{REF}/pt_predict.json"))["frames"]
ref_cam = np.load(f"{REF}/pt_cam.npy")


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


rows = []
for q in (50, 60, 70, 80, 85, 90, 95, 100):
    nb = nm = 0; ious = []; camd = []; sizes = []; stream = []
    for k, f in enumerate(frames):
        buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1]; sizes.append(len(buf))
        img = cv2.imdecode(buf, 1)
        r = det.detect(img); B = [b[:4] for b in r["boxes"]]; A = [b[:4] for b in ref[k]["boxes"]]
        nb += len(B)
        used = set()
        for a in A:
            c = [(iou(a, b), j) for j, b in enumerate(B) if j not in used]
            if c and max(c)[0] >= 0.5:
                v, j = max(c); used.add(j); ious.append(v); nm += 1
        lb, rr, l, t = letterbox(img)
        x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.)
        _, act = det.sess.run(det.out_names, {det.inp: x})
        camd.append(float(np.abs(eigencam_grid(act) - ref_cam[k]).max()))
        stream.append({"i": k, "t": k / 29.97, "boxes": r["boxes"]})
    json.dump({"fps": 29.97, "frames": stream}, open(f"{OUT}/stream_q{q}.json", "w"))
    rows.append(dict(quality=q, kb_median=float(np.median(sizes)) / 1024, kb_max=max(sizes) / 1024, ref_boxes=sum(len(f["boxes"]) for f in ref),
                     onnx_boxes=nb, matched=nm, iou_min=float(min(ious)), iou_median=float(np.median(ious)),
                     cam_max=float(max(camd)), cam_median=float(np.median(camd))))
    print(rows[-1])
json.dump(rows, open(f"{OUT}/jpeg_tradeoff.json", "w"), indent=1)
