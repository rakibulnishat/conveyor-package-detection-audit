"""Server-side latency budget of the deployed ONNX path, per stage, on this CPU.

Stages timed for every frame of the bundled clip (191 frames, 3 repeats, after 5 warm-up calls):
  decode   cv2.imdecode of the JPEG the browser would send (quality 0.85, 1020x600)
  prep     letterbox to 384x640 + HWC->CHW float32
  infer    onnxruntime session.run (both outputs)
  post     score filter + box rescale + NMS
  cam      Eigen-CAM grid (power iteration, 30 steps)
Also: PyTorch (Ultralytics predict, same checkpoint) end-to-end per frame for reference,
JPEG payload bytes, and JSON response bytes.
usage: python -I latency_bench.py <app_root> <best.pt> <out_dir> <threads>
"""
import json
import sys
import time

import cv2
import numpy as np

ROOT, PT, OUT, THREADS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, ROOT)
import onnxruntime as ort  # noqa: E402
from pkginspect import engine  # noqa: E402

so = ort.SessionOptions()
so.intra_op_num_threads = THREADS
so.inter_op_num_threads = 1
so.log_severity_level = 3
sess = ort.InferenceSession(str(engine.MODEL_PATH), sess_options=so, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
outs = [o.name for o in sess.get_outputs()]

cap = cv2.VideoCapture(ROOT + "/public/co.mp4")
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.resize(f, (1020, 600)))
jpgs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() for f in frames]


def run(buf):
    t = [time.perf_counter()]
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR); t.append(time.perf_counter())
    lb, r, left, top = engine.letterbox(img)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0); t.append(time.perf_counter())
    out, act = sess.run(outs, {inp: x}); t.append(time.perf_counter())
    pred = out[0].T
    sc = pred[:, 4:].max(1); m = sc >= engine.CONF_THRES; pred, sc = pred[m], sc[m]
    boxes = np.zeros((len(pred), 4), np.float32)
    if len(pred):
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes[:, 0] = (cx - bw / 2 - left) / r; boxes[:, 1] = (cy - bh / 2 - top) / r
        boxes[:, 2] = (cx + bw / 2 - left) / r; boxes[:, 3] = (cy + bh / 2 - top) / r
        keep = engine.nms(boxes, sc, engine.IOU_THRES); boxes, sc = boxes[keep], sc[keep]
    t.append(time.perf_counter())
    cam = engine.eigencam_grid(act); t.append(time.perf_counter())
    resp = {"w": 1020, "h": 600, "boxes": [[round(float(v), 1) for v in b] + [round(float(s), 4), 0] for b, s in zip(boxes, sc)],
            "cam": {"h": 12, "w": 20, "data": [round(float(v), 3) for v in cam.reshape(-1)]}, "ms": 0}
    return np.diff(t) * 1000, len(json.dumps(resp).encode())


for b in jpgs[:5]:
    run(b)
rows, resp_bytes = [], []
for rep in range(3):
    for b in jpgs:
        st, rb = run(b); rows.append(st); resp_bytes.append(rb)
A = np.array(rows)
names = ["decode", "prep", "infer", "post", "cam"]
res = {"threads": THREADS, "n_calls": len(A), "stage_ms": {}}
for i, n in enumerate(names):
    res["stage_ms"][n] = dict(median=float(np.median(A[:, i])), p95=float(np.percentile(A[:, i], 95)), mean=float(A[:, i].mean()))
tot = A.sum(1)
res["total_ms"] = dict(median=float(np.median(tot)), p95=float(np.percentile(tot, 95)), mean=float(tot.mean()), max=float(tot.max()))
res["jpeg_bytes"] = dict(median=float(np.median([len(b) for b in jpgs])), min=min(len(b) for b in jpgs), max=max(len(b) for b in jpgs))
res["response_bytes"] = dict(median=float(np.median(resp_bytes)), max=max(resp_bytes))
np.savetxt(f"{OUT}/onnx_stage_ms_t{THREADS}.csv", A, delimiter=",", header=",".join(names), comments="", fmt="%.4f")

# PyTorch reference, same threads
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
torch.set_num_threads(THREADS)
model = YOLO(PT)
for f in frames[:5]:
    model.predict(f, conf=0.25, iou=0.7, verbose=False, device="cpu")
pt = []
for f in frames:
    t0 = time.perf_counter(); model.predict(f, conf=0.25, iou=0.7, verbose=False, device="cpu"); pt.append((time.perf_counter() - t0) * 1000)
res["pytorch_predict_ms"] = dict(median=float(np.median(pt)), p95=float(np.percentile(pt, 95)), mean=float(np.mean(pt)))
np.savetxt(f"{OUT}/pytorch_predict_ms_t{THREADS}.csv", np.array(pt), delimiter=",", header="ms", comments="", fmt="%.4f")
json.dump(res, open(f"{OUT}/latency_t{THREADS}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
