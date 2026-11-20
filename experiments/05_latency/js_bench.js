// node js_bench.js <pipeline.js> <onnx_jpeg85.json> <out.json>
// Times the browser-side work per detection in Node (V8, same engine as Chrome): tracker update + crossing rule
// per frame, and heat-map reconstruction (camToValues 1020x600, maskToBox, jetRGBA) per crossing event.
const fs = require('fs');
const PI = require(process.argv[2]);
const det = JSON.parse(fs.readFileSync(process.argv[3]));

const W = 1020, H = 600, LINE = Math.floor(W * 0.41), TOL = 18;
const now = () => Number(process.hrtime.bigint()) / 1e6;
const trk = [], heat = [];
for (let rep = 0; rep < 4; rep++) {
  const tr = new PI.Tracker();
  det.frames.forEach((f, k) => {
    const t0 = now();
    for (const t of tr.update(f.boxes.map((b) => b.slice(0, 6)), f.t, { 0: 'object' })) PI.isCrossing(t, LINE, TOL);
    const t1 = now();
    if (rep > 0) trk.push(t1 - t0);
    if (f.boxes.length) {
      const b = f.boxes[0];
      const s = now();
      const v = PI.camToValues(f.cam, W, H); const m = PI.maskToBox(v, W, H, b); PI.jetRGBA(m, W, H);
      const e = now();
      if (rep > 0) heat.push(e - s);
    }
  });
}
const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const res = { node: process.version, tracker_ms_median: q(trk, 0.5), tracker_ms_p95: q(trk, 0.95), heat_ms_median: q(heat, 0.5), heat_ms_p95: q(heat, 0.95), n_tracker: trk.length, n_heat: heat.length };
fs.writeFileSync(process.argv[4], JSON.stringify(res, null, 1)); console.log(res);
