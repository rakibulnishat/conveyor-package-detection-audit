"""Per-frame unrounded Eigen-CAM difference, ONNX vs PyTorch reference, for raw and JPEG q85 frames."""
import csv, sys
import cv2, numpy as np
ROOT, REF, OUT = sys.argv[1:4]
sys.path.insert(0, ROOT)
from pkginspect.engine import Detector, eigencam_grid, letterbox
det = Detector()
cap = cv2.VideoCapture(ROOT + "/public/co.mp4"); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(cv2.resize(f, (1020, 600)))
ref = np.load(REF + "/pt_cam.npy")
def cam(img):
    lb, r, l, t = letterbox(img)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.)
    return eigencam_grid(det.sess.run(det.out_names, {det.inp: x})[1])
rows = []
for k, f in enumerate(fr):
    j = cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], 1)
    rows.append(dict(frame=k, raw=float(np.abs(cam(f) - ref[k]).max()), jpeg85=float(np.abs(cam(j) - ref[k]).max())))
with open(OUT + "/cam_unrounded_per_frame.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
a = np.array([[r["raw"], r["jpeg85"]] for r in rows]); print("max", a.max(0), "median", np.median(a, 0))
