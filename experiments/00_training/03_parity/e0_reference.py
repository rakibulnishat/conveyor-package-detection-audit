"""E0: PyTorch reference for the deployed checkpoint on the bundled 191-frame clip.

Outputs (all in OUT dir):
  pt_predict.json   per-frame boxes from model.predict (conf 0.25, iou 0.7) = web operating point
  pt_track.json     per-frame boxes + track IDs from model.track(persist=True), library defaults
  pt_cam.npy        (191, 12, 20) Eigen-CAM grids from the desktop TrackHook formula, hooked during predict
  meta.json         versions, layer name, tracker config, defaults actually used
Usage: python e0_reference.py <best.pt> <co.mp4> <outdir>
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

PT, VIDEO, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (1020, 600)))  # identical to video_app.py
    return frames, fps


def eigencam(acts):
    """Verbatim port of TrackHook.get_heatmap up to the normalised low-resolution grid."""
    feat = acts.squeeze(0)
    C, Hf, Wf = feat.shape
    flat = feat.reshape(C, Hf * Wf)
    cam = flat.abs().mean(0).reshape(Hf, Wf).numpy().astype(np.float32)
    mat = flat.numpy().astype(np.float32)
    mat = mat - mat.mean(axis=1, keepdims=True)
    cov = mat @ mat.T
    v = np.ones(C, np.float32)
    for _ in range(30):
        v = cov @ v
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-8:
            break
        v /= nrm
    pca = np.maximum((v @ mat).reshape(Hf, Wf), 0)
    if pca.max() > 0:
        cam = pca / pca.max()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


frames, fps = read_frames(VIDEO)
assert len(frames) == 191, len(frames)

# ---------------- predict pass (web operating point) + Eigen-CAM hook ----------------
model = YOLO(PT)
convs = [(n, m) for n, m in model.model.named_modules() if isinstance(m, torch.nn.Conv2d)]
idx = max(0, int(len(convs) * 0.35))
for i in range(idx, -1, -1):
    if convs[i][1].kernel_size[0] > 1:
        idx = i
        break
layer_name_unfused = convs[idx][0]
n_convs_unfused = len(convs)

acts_store = {}
pred, cams, pt_ms = [], [], []
model.predict(frames[0], conf=0.25, iou=0.7, verbose=False, device="cpu")  # warm-up: predictor builds its fused copy
live_layer = dict(model.predictor.model.model.named_modules())[layer_name_unfused]  # module that actually runs


def _store(m, i, o):  # returns None so the forward output is never replaced
    acts_store["a"] = o.detach().cpu().float()


h = live_layer.register_forward_hook(_store)
for k, f in enumerate(frames):
    acts_store.clear()
    t0 = time.perf_counter()
    r = model.predict(f, conf=0.25, iou=0.7, verbose=False, device="cpu")[0]
    pt_ms.append((time.perf_counter() - t0) * 1000)
    assert "a" in acts_store, f"hook did not fire on frame {k}"
    b = r.boxes
    pred.append({"i": k, "t": k / fps, "imgsz": list(r.orig_shape),
                 "boxes": [[*map(float, xy), float(c), int(cl)] for xy, c, cl in
                           zip(b.xyxy.numpy(), b.conf.numpy(), b.cls.numpy())]})
    cams.append(eigencam(acts_store["a"]))
h.remove()
cams = np.stack(cams)
np.save(OUT / "pt_cam.npy", cams)
json.dump({"fps": fps, "frames": pred}, open(OUT / "pt_predict.json", "w"))

# letterbox shape actually used by ultralytics for 1020x600 at imgsz 640
lb_shape = list(acts_store["a"].shape)

# ---------------- track pass (desktop behaviour, library defaults) ----------------
tmodel = YOLO(PT)
# desktop TrackHook: layer object captured ONCE from the unfused model at start-up, as video_app.py does
tconvs = [(n, m) for n, m in tmodel.model.named_modules() if isinstance(m, torch.nn.Conv2d)]
desk_layer = tconvs[idx][1]
desk_store, desk_cams, desk_fired = {}, [], 0


def _desk_hook(m, i, o):  # same logic as TrackHook._hook: keep the first activation of the call, return None
    if "a" not in desk_store:
        desk_store["a"] = o.detach().cpu().float()


trk = []
for k, f in enumerate(frames):
    desk_store.clear()  # TrackHook.attach() resets _acts
    hh = desk_layer.register_forward_hook(_desk_hook)  # attach() on the start-up (unfused) module
    r = tmodel.track(f, persist=True, verbose=False, device="cpu")[0]
    hh.remove()  # detach()
    if "a" in desk_store:
        desk_fired += 1
        desk_cams.append(eigencam(desk_store["a"]))
    else:
        desk_cams.append(np.zeros_like(cams[0]))
    b = r.boxes
    ids = b.id.numpy().astype(int).tolist() if b.id is not None else [None] * len(b)
    trk.append({"i": k, "t": k / fps,
                "boxes": [[*map(float, xy), float(c), int(cl), tid] for xy, c, cl, tid in
                          zip(b.xyxy.numpy(), b.conf.numpy(), b.cls.numpy(), ids)]})
json.dump({"fps": fps, "frames": trk}, open(OUT / "pt_track.json", "w"))
np.save(OUT / "pt_cam_desktop_hook.npy", np.stack(desk_cams))

# what the track call really used
pr = tmodel.predictor
meta = {
    "ultralytics": ultralytics.__version__, "torch": torch.__version__,
    "eigencam_layer": layer_name_unfused, "n_conv2d_unfused": n_convs_unfused,
    "eigencam_act_shape": lb_shape,
    "track_conf_used": float(pr.args.conf), "track_iou_used": float(pr.args.iou),
    "tracker_cfg": str(pr.args.tracker), "track_imgsz": pr.args.imgsz,
    "pt_cpu_ms_median_predict": float(np.median(pt_ms[1:])),
    "pt_cpu_ms_first": float(pt_ms[0]),
    "fps": fps, "n_frames": len(frames),
    "desktop_hook_fired_frames": desk_fired,
    "desktop_layer_is_live_module": any(m is desk_layer for m in tmodel.model.modules()),
}
json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
