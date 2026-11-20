// node tracker_web_rates.js <pipeline.js> <detections.json> <out.json>
// Emulates the web app's sampler: frame shown at time t_k = phase + k/rate is floor(t_k * fps) (a seek shows the
// frame whose interval contains t). Rates are the four options offered in the UI (4, 6, 10, 15 frames/s) plus 29.97.
// Every start phase on the frame grid inside one sampling interval is evaluated, so the result does not depend on
// where the clip happens to start.
const fs = require('fs');
const PI = require(process.argv[2]);
const data = JSON.parse(fs.readFileSync(process.argv[3]));
const LINE = Math.floor(1020 * 0.41), TOL = 18, N = data.frames.length, FPS = data.fps;
const rows = [];
for (const rate of [29.97, 15, 10, 6, 4]) {
  const nPhase = Math.max(1, Math.round(FPS / rate));
  for (let p = 0; p < nPhase; p++) {
    const phase = p / FPS;
    for (const rule of ['window_only', 'window_or_signchange']) {
      const tr = new PI.Tracker(), crossed = new Set(); let ev = 0, used = 0; const times = [];
      const ids = new Set();
      for (let k = 0; ; k++) {
        const t = phase + k / rate; const fi = Math.floor(t * FPS + 1e-9);
        if (fi >= N) break;
        used++;
        const f = data.frames[fi];
        for (const tt of tr.update(f.boxes.map((b) => b.slice(0, 6)), t, { 0: 'object' })) {
          ids.add(tt.id);
          const hit = rule === 'window_only' ? Math.abs(tt.cx - LINE) < TOL : PI.isCrossing(tt, LINE, TOL);
          if (hit && !crossed.has(tt.id)) { crossed.add(tt.id); ev++; times.push([tt.id, +t.toFixed(4), +tt.cx.toFixed(1)]); }
        }
      }
      rows.push({ rate, phase_frames: p, rule, samples: used, ids: ids.size, events: ev, times });
    }
  }
}
fs.writeFileSync(process.argv[4], JSON.stringify(rows, null, 1));
const agg = {};
for (const r of rows) { const k = r.rate + ' ' + r.rule; (agg[k] = agg[k] || []).push(r.events + '/' + r.ids); }
for (const k in agg) console.log(k.padEnd(28), agg[k].join(' '));
