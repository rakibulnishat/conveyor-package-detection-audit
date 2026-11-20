import sys, os, json, glob, shutil
import numpy as np, pandas as pd
from ultralytics import YOLO
import ultralytics
CKPT, DS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
yaml_path = f'{OUT}/data_abs.yaml'
open(yaml_path, 'w').write(f"path: {DS}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['object']\n")
model = YOLO(CKPT); summary = {'ultralytics': ultralytics.__version__}
for split in ('val', 'test'):
    m = model.val(data=yaml_path, split=split, imgsz=640, batch=16, iou=0.7, device='cpu', plots=True, save_json=True,
                  project=f'{OUT}/runs', name=split, exist_ok=True, verbose=False)
    summary[split] = dict(precision=float(m.box.mp), recall=float(m.box.mr), mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
                          confusion_matrix=m.confusion_matrix.matrix.tolist())
    for (x, y, xl, yl), name in zip(m.curves_results, m.curves):
        df = pd.DataFrame({'x': np.asarray(x).ravel()}); y = np.asarray(y)
        for c in range(y.shape[0]): df[f'y_class{c}'] = y[c]
        df.to_csv(f'{OUT}/{split}_curve_{name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")}.csv', index=False)
rows = []
for split in ('valid', 'test'):
    for p in sorted(glob.glob(f'{DS}/{split}/images/*')):
        r = model.predict(p, conf=0.001, iou=0.7, imgsz=640, verbose=False, device='cpu')[0]
        H, W = r.orig_shape
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy()):
            rows.append(dict(split=split, file=os.path.basename(p), kind='pred', x1=x1, y1=y1, x2=x2, y2=y2, conf=float(c)))
        lab = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        for l in open(lab).read().split('\n'):
            if not l.strip(): continue
            _, cx, cy, w, h = map(float, l.split()[:5])
            rows.append(dict(split=split, file=os.path.basename(p), kind='gt', x1=(cx - w / 2) * W, y1=(cy - h / 2) * H, x2=(cx + w / 2) * W, y2=(cy + h / 2) * H, conf=np.nan))
pd.DataFrame(rows).to_csv(f'{OUT}/predictions_and_gt_conf0001.csv', index=False)
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
