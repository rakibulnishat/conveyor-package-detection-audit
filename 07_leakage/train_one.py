"""Train one leakage-experiment condition and evaluate it. Resumable.
usage: python train_one.py <condition: leaky|blocked>
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from ultralytics import YOLO

# Run from the repository root. Splits, runs and the COCO-pretrained yolo12s.pt live in WORK;
# the files reported in the paper are written to OUT.
WORK = 'experiments/07_leakage/work'
OUT = 'experiments/results/leakage'
ROOT = WORK
cond = sys.argv[1]
data = f'{ROOT}/{cond}/data.yaml'
run_dir = f'{ROOT}/runs/{cond}'
last = f'{run_dir}/weights/last.pt'

args = json.load(open('experiments/07_leakage/deployed_train_args.json'))
keep = ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
        'box', 'cls', 'dfl', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective',
        'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste', 'erasing', 'close_mosaic', 'imgsz', 'seed',
        'deterministic', 'cos_lr', 'nbs', 'auto_augment', 'patience', 'pretrained']
cfg = {k: args[k] for k in keep if k in args}
cfg.update(epochs=30, batch=4, device='cpu', workers=2, amp=False, plots=True, val=True, verbose=False,
           project=f'{ROOT}/runs', name=cond, exist_ok=True)
json.dump(cfg, open(f'{OUT}/{cond}_train_cfg.json', 'w'), indent=1)

t0 = time.time()
if os.path.exists(last) and not os.path.exists(f'{run_dir}/weights/best.pt.done'):
    YOLO(last).train(resume=True)
elif not os.path.exists(f'{run_dir}/weights/best.pt.done'):
    YOLO(f'{ROOT}/yolo12s.pt').train(data=data, **cfg)
open(f'{run_dir}/weights/best.pt.done', 'w').write(str(time.time() - t0))

# ---- evaluation on the shared test block
m = YOLO(f'{run_dir}/weights/best.pt').val(data=data, split='test', imgsz=640, batch=4, iou=0.7, device='cpu',
                                            plots=False, project=f'{ROOT}/runs', name=f'{cond}_test', exist_ok=True,
                                            verbose=False)
res = dict(condition=cond, precision=float(m.box.mp), recall=float(m.box.mr), mAP50=float(m.box.map50),
           mAP50_95=float(m.box.map), confusion_matrix=m.confusion_matrix.matrix.tolist())
rows = []
model = YOLO(f'{run_dir}/weights/best.pt')
import glob
for p in sorted(glob.glob(f'{ROOT}/{cond}/test/images/*')):
    r = model.predict(p, conf=0.001, iou=0.7, imgsz=640, verbose=False, device='cpu')[0]
    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy()):
        rows.append(dict(file=os.path.basename(p), x1=x1, y1=y1, x2=x2, y2=y2, conf=float(c)))
pd.DataFrame(rows).to_csv(f'{OUT}/{cond}_test_predictions.csv', index=False)
json.dump(res, open(f'{OUT}/{cond}_test_result.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
