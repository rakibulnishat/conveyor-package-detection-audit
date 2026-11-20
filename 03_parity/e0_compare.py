"""E0 comparison: PyTorch reference vs ONNX (raw frames) vs ONNX (JPEG q=85 frames, the deployed transport).

Usage: python -I e0_compare.py <vercel_app_root> <ref_dir> <out_dir>
Writes: per_frame_parity.csv, box_pairs.csv, parity_summary.json, onnx_raw.json, onnx_jpeg85.json
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT, REF, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, ROOT)
from pkginspect.engine import Detector  # noqa: E402

det = Detector()
cap = cv2.VideoCapture(ROOT + "/public/co.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.resize(f, (1020, 600)))
assert len(frames) == 191

ref = json.load(open(REF / "pt_predict.json"))["frames"]
ref_cam = np.load(REF / "pt_cam.npy")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def greedy_match(A, B, thr=0.5):
    pairs, used = [], set()
    cand = sorted(((iou(a, b), i, j) for i, a in enumerate(A) for j, b in enumerate(B)), reverse=True)
    ua = set()
    for v, i, j in cand:
        if v < thr or i in ua or j in used:
            continue
        ua.add(i); used.add(j); pairs.append((i, j, v))
    return pairs, [i for i in range(len(A)) if i not in ua], [j for j in range(len(B)) if j not in used]


streams = {}
for mode in ("raw", "jpeg85"):
    fr_out, ms = [], []
    for k, f in enumerate(frames):
        img = f if mode == "raw" else cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], 1)
        t0 = time.perf_counter()
        r = det.detect(img)
        ms.append((time.perf_counter() - t0) * 1000)
        fr_out.append({"i": k, "t": k / fps, "boxes": r["boxes"], "cam": r["cam"]})
    streams[mode] = (fr_out, ms)
    json.dump({"fps": fps, "frames": fr_out}, open(OUT / f"onnx_{mode}.json", "w"))

rows, pairs_rows, summary = [], [], {"fps": fps, "n_frames": len(frames)}
for mode, (fr_out, ms) in streams.items():
    n_ref = n_onx = n_match = 0
    ious, dconf, camd = [], [], []
    extra, missing = [], []
    for k in range(len(frames)):
        A = [b[:4] for b in ref[k]["boxes"]]
        B = [b[:4] for b in fr_out[k]["boxes"]]
        p, ua, ub = greedy_match(A, B)
        n_ref += len(A); n_onx += len(B); n_match += len(p)
        for i, j, v in p:
            ious.append(v); dconf.append(abs(ref[k]["boxes"][i][4] - fr_out[k]["boxes"][j][4]))
            pairs_rows.append([mode, k, i, j, round(v, 6), ref[k]["boxes"][i][4], fr_out[k]["boxes"][j][4]])
        for j in ub:
            extra.append({"frame": k, "conf": fr_out[k]["boxes"][j][4]})
        for i in ua:
            missing.append({"frame": k, "conf": ref[k]["boxes"][i][4]})
        g = np.array(fr_out[k]["cam"]["data"], np.float32).reshape(fr_out[k]["cam"]["h"], fr_out[k]["cam"]["w"])
        d = float(np.abs(g - ref_cam[k]).max())
        camd.append(d)
        rows.append([mode, k, len(A), len(B), len(p), min([v for _, _, v in p], default=""), round(d, 6), round(ms[k], 2)])
    summary[mode] = {
        "ref_boxes": n_ref, "onnx_boxes": n_onx, "matched": n_match,
        "iou_min": float(min(ious)), "iou_p01": float(np.percentile(ious, 1)), "iou_median": float(np.median(ious)),
        "n_pairs_iou_lt_0999": int(sum(v < 0.999 for v in ious)),
        "conf_absdiff_max": float(max(dconf)),
        "unmatched_onnx": extra, "unmatched_ref": missing,
        "cam_maxabs_max": float(max(camd)), "cam_maxabs_median": float(np.median(camd)),
        "frames_cam_gt_0p01": int(sum(c > 0.01 for c in camd)),
        "onnx_cpu_ms_first": float(ms[0]), "onnx_cpu_ms_median": float(np.median(ms[1:])),
        "onnx_cpu_ms_p95": float(np.percentile(ms[1:], 95)),
    }
with open(OUT / "per_frame_parity.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["mode", "frame", "n_ref", "n_onnx", "n_matched", "iou_min", "cam_maxabs", "onnx_ms"]); w.writerows(rows)
with open(OUT / "box_pairs.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["mode", "frame", "ref_idx", "onnx_idx", "iou", "ref_conf", "onnx_conf"]); w.writerows(pairs_rows)
json.dump(summary, open(OUT / "parity_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
