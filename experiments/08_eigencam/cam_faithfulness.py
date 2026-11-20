"""Eigen-CAM checks on the 191 frames of co.mp4 (decoded, resized to 1020x600 as in both apps, no JPEG).

Protocol (fixed before running):
  A. Deletion / insertion on the deployed ONNX model.
     The 12x20 Eigen-CAM grid maps one-to-one onto 32x32-pixel cells of the 384x640
     letterboxed input. Cells are replaced by the letterbox grey (114/255) in order of
     decreasing CAM (deletion) or revealed on a grey canvas in that order (insertion),
     in steps of 10 % of the 240 cells (11 points, 0 % ... 100 %).
     Score at each step = maximum package score over all 5040 anchors (pre-NMS).
     Orders compared: Eigen-CAM, reversed Eigen-CAM, and 3 random orders (seeds 0, 1, 2).
     AUC by the trapezoidal rule on the normalised x-axis [0, 1].
  B. Localisation against ground truth (dataset labels; every frame of the clip is labelled).
     Pointing game: the arg-max cell centre lies inside a ground-truth box.
     Energy-in-box: share of CAM mass on cells whose centre lies inside any ground-truth box.
     Chance level = share of cells whose centre lies inside any ground-truth box.
  C. Model-parameter randomisation (Adebayo et al. 2018) with PyTorch on the same tensors:
     trained best.pt, COCO-pretrained yolo12s.pt, and randomly initialised YOLOv12s
     (seeds 0, 1, 2); plus a Sobel edge-magnitude map pooled to 12x20 as an image-only baseline.
     Spearman rank correlation of each map with the trained map, per frame.
Usage: python cam_faithfulness.py <app_root> <best.pt> <yolo12s.pt> <video> <labels_csv> <out_dir> <threads>
"""
import csv, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

app_root, best_pt, coco_pt, video, labels_csv, out_dir, threads = sys.argv[1:8]
threads = int(threads)
out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, app_root)
from pkginspect.engine import letterbox, eigencam_grid  # noqa: E402
import onnxruntime as ort  # noqa: E402

so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.log_severity_level = 3
sess = ort.InferenceSession(str(Path(app_root) / "model" / "best.onnx"), sess_options=so,
                            providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
outs = [o.name for o in sess.get_outputs()]
GH, GW, CELL = 12, 20, 32
GREY = 114.0 / 255.0
STEPS = np.linspace(0, 1, 11)


def run(x):
    o, a = sess.run(outs, {inp: x})
    return float(o[0, 4:, :].max()), a


def frames():
    cap = cv2.VideoCapture(video)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.resize(f, (1020, 600))  # the desktop and web pipelines both work on 1020x600 frames
    cap.release()


# ground-truth boxes per clip frame (normalised xywh, as labelled on the stretched 512x512 export)
gt = {}
with open(labels_csv) as fh:
    for r in csv.DictReader(fh):
        gt.setdefault(int(r["frame_no"]), []).append([float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"])])


def cell_mask_from_gt(boxes_n, W, H, r, left, top):
    yc = (np.arange(GH) + 0.5) * CELL
    xc = (np.arange(GW) + 0.5) * CELL
    X, Y = np.meshgrid(xc, yc)
    m = np.zeros((GH, GW), bool)
    for cx, cy, w, h in boxes_n:
        x1 = (cx - w / 2) * W * r + left; x2 = (cx + w / 2) * W * r + left
        y1 = (cy - h / 2) * H * r + top; y2 = (cy + h / 2) * H * r + top
        m |= (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
    return m


def curve(x, order, mode):
    k_all = len(order)
    vals = []
    for s in STEPS:
        k = int(round(s * k_all))
        cells = order[:k]
        if mode == "del":
            z = x.copy()
            for c in cells:
                i, j = divmod(int(c), GW)
                z[:, :, i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL] = GREY
        else:
            z = np.full_like(x, GREY)
            for c in cells:
                i, j = divmod(int(c), GW)
                z[:, :, i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL] = \
                    x[:, :, i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL]
        vals.append(run(z)[0])
    return vals


def auc(v):
    return float(np.trapezoid(v, STEPS)) if hasattr(np, "trapezoid") else float(np.trapz(v, STEPS))


# ---------- torch models for part C ----------
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
torch.set_num_threads(threads)
LAYER = "model.8.m.0.0.attn.pe.conv"


def torch_model(src, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        m = YOLO("yolo12s.yaml").model
    else:
        m = YOLO(src).model
    m = m.float().fuse().eval()  # fused, as in the Ultralytics predictor and the ONNX export
    store = {}
    layer = dict(m.named_modules())[LAYER]
    layer.register_forward_hook(lambda mod, i, o: store.__setitem__("a", o.detach().cpu().numpy()) or None)
    return m, store


models = {"trained": torch_model(best_pt), "coco": torch_model(coco_pt)}
for s in (0, 1, 2):
    models[f"random_s{s}"] = torch_model(None, seed=s)


def spearman(a, b):
    ra = a.reshape(-1).argsort().argsort().astype(float)
    rb = b.reshape(-1).argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


rows_del, rows_loc, rows_rand = [], [], []
t0 = time.time()
MAXF = int(os.getenv("MAXF", "0"))
for fi, f in enumerate(frames(), start=1):
    if MAXF and fi > MAXF:
        break
    H, W = f.shape[:2]
    lb, r, left, top = letterbox(f)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    s0, act = run(x)
    cam = eigencam_grid(act)
    order = np.argsort(-cam.reshape(-1), kind="stable")
    rec = {"frame": fi, "score_full": s0}
    for mode in ("del", "ins"):
        v = curve(x, order, mode); rec[f"cam_{mode}_auc"] = auc(v); rec[f"cam_{mode}_curve"] = v
        v = curve(x, order[::-1], mode); rec[f"inv_{mode}_auc"] = auc(v); rec[f"inv_{mode}_curve"] = v
        rv = []
        for seed in (0, 1, 2):
            ro = np.random.default_rng(seed + 1000 * fi).permutation(GH * GW)
            rv.append(curve(x, ro, mode))
        rv = np.mean(rv, 0).tolist(); rec[f"rand_{mode}_auc"] = auc(rv); rec[f"rand_{mode}_curve"] = rv
    rows_del.append(rec)

    m = cell_mask_from_gt(gt.get(fi, []), W, H, r, left, top)
    loc = {"frame": fi, "chance": float(m.mean())}
    maps = {"onnx": cam}
    # sign check: first principal component taken with the opposite sign (PCA sign is arbitrary)
    neg = eigencam_grid(-act[0][None] + 2 * act[0].reshape(act.shape[1], -1).mean(1)[None, :, None, None])
    maps["onnx_negsign"] = neg
    # plain channel-mean |activation| map (no PCA)
    am = np.abs(act[0]).mean(0); maps["absmean"] = am / max(am.max(), 1e-12)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        for name, (mdl, store) in models.items():
            mdl(xt); maps[name] = eigencam_grid(store["a"])
    g = cv2.cvtColor(lb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    maps["sobel"] = mag.reshape(GH, CELL, GW, CELL).mean((1, 3))
    rr = {"frame": fi}
    for name, mp in maps.items():
        a = np.unravel_index(int(np.argmax(mp)), mp.shape)
        loc[f"{name}_point"] = int(m[a])
        loc[f"{name}_energy"] = float(mp[m].sum() / max(mp.sum(), 1e-12))
        if name != "trained":
            rr[f"rho_{name}"] = spearman(maps["trained"], mp)
    rows_loc.append(loc); rows_rand.append(rr)
    if fi % 10 == 0:
        print(f"frame {fi}  {time.time() - t0:.0f}s", flush=True)

with open(out / "deletion_insertion.json", "w") as fh:
    json.dump({"steps": STEPS.tolist(), "frames": rows_del}, fh)
for name, rows in (("localisation.csv", rows_loc), ("randomisation_spearman.csv", rows_rand)):
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
summ = {"n_frames": len(rows_del), "threads": threads, "wall_s": time.time() - t0}
for k in ("cam_del_auc", "inv_del_auc", "rand_del_auc", "cam_ins_auc", "inv_ins_auc", "rand_ins_auc"):
    v = np.array([r[k] for r in rows_del]); summ[k] = {"mean": float(v.mean()), "median": float(np.median(v))}
d = np.array([r["rand_del_auc"] - r["cam_del_auc"] for r in rows_del])
i_ = np.array([r["cam_ins_auc"] - r["rand_ins_auc"] for r in rows_del])
summ["frames_cam_beats_random_deletion"] = int((d > 0).sum())
summ["frames_cam_beats_random_insertion"] = int((i_ > 0).sum())
for k in rows_loc[0]:
    if k != "frame":
        summ[f"loc_{k}"] = float(np.mean([r[k] for r in rows_loc]))
for k in rows_rand[0]:
    if k != "frame":
        v = np.array([r[k] for r in rows_rand]); summ[k] = {"mean": float(v.mean()), "median": float(np.median(v))}
json.dump(summ, open(out / "summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
