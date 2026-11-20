# Deploy to Vercel (free Hobby plan) — step by step

No credit card is needed for Vercel's free **Hobby** plan (Vercel only asks for card details when you upgrade).
Hobby is for **personal, non-commercial use** (Vercel's fair-use rule). Sign up at <https://vercel.com/signup>.

## What was built (and why it looks different from the PyTorch version)

Vercel runs short, stateless serverless functions. I verified these limits in Vercel's docs (Oct 2026):
Hobby = 2 GB memory, 300 s max duration, **4.5 MB request/response body**, **500 MB** Python bundle.
PyTorch alone is too big for that, and a 30 MB video upload would be rejected, so the app was re-architected:

| Part | Where it runs | What it does |
|---|---|---|
| `/api/detect` | Vercel function (ONNX Runtime, **no PyTorch**) | one JPEG frame in → boxes + EigenCAM attention grid out |
| `/api/analyze` | Vercel function | one cropped package in → Groq vision (or local heuristic) → flap / damage |
| Video decoding, tracking, detection line, heatmap drawing, reports, ZIPs, charts, annotated-video recording | **your browser** | the video never leaves the device; only ~30–90 KB frames are sent |

Estimated function bundle: **~344 MB of 500 MB** (measured from the installed packages + model).

## Project files

```
index.py              Vercel entrypoint (exposes the FastAPI `app`)
pkginspect/api.py     API: /api/health, /api/detect, /api/analyze
pkginspect/engine.py  ONNX detector + EigenCAM (numpy)
pkginspect/vision.py  Groq / local analysis
model/best.onnx       your YOLO model exported to ONNX (EigenCAM layer included)
public/               dashboard (index.html, app.js, pipeline.js, style.css, alert.mp3, sample videos)
requirements.txt      pinned, tested dependencies
vercel.json           function settings
.vercelignore         keeps tests/tools out of the upload
tools/export_onnx.py  how best.onnx was produced (only needed if you retrain the model)
tests/                backend + browser-logic tests (not uploaded)
```

## 0. FIRST: rotate your Groq key

Your original `video_app.py` contained a hard-coded Groq API key. Treat it as leaked: open
<https://console.groq.com/keys>, **delete it, create a new one**. Never put the key in code or Git —
only in Vercel's Environment Variables (step 5).

## 1. Install the tools

* Node.js (LTS) from <https://nodejs.org> — check with `node -v`.
* Vercel CLI: `npm install -g vercel` — check with `vercel --version`
  (Vercel's FastAPI docs require CLI **48.1.8 or newer**).

## 2. Unzip and open the folder

```bash
unzip package-inspector-vercel.zip
cd package-inspector-vercel
```

## 3. (Recommended) Test locally first

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt uvicorn
uvicorn index:app --port 7861
# open http://127.0.0.1:7861  -> click "Try the sample video"
```
(Python 3.12. Optional automatic checks: `python tests/test_api.py` and `node tests/test_pipeline.js`.)

## 4. Log in and deploy

```bash
vercel login            # follow the email / GitHub / Google prompt
vercel                  # first deploy = a preview
```
Answer the prompts: *Set up and deploy?* **Y** → pick your account → *Link to existing project?* **N** →
project name (e.g. `package-inspector`) → directory `./` → accept detected settings.
The first build installs the Python packages and takes a few minutes.

## 5. Add your NEW Groq key (secret)

```bash
vercel env add GROQ_API_KEY production
```
Paste the new key when asked (also add it for *preview* if you use previews). Optional hardening:
`vercel env add SERVER_KEY_ENABLED production` with value `0` makes visitors paste **their own** Groq key,
so strangers cannot spend your Groq quota. (Without a key anywhere, the app uses the local flap-only heuristic.)

## 6. Deploy to production

```bash
vercel --prod
```
It prints your public URL (`https://package-inspector-xxxx.vercel.app`). Environment variables only apply to
deployments made **after** you add them, so always redeploy after changing them.

## 7. Verify

1. Open the URL; the *Server* pill should turn **ready** (the first request after idle can be slow — cold start).
2. The *Analysis* pill should say **Groq vision AI** (if you set the key).
3. Click **Try the sample video** → expect 3 packages, a live XAI panel, charts, and 5 downloads.

## Alternative: deploy from GitHub (no CLI)

Push this folder to a GitHub repo (`best.onnx` is ~36 MB and `co.mp4` ~12 MB — both under GitHub's 100 MB file limit),
then in Vercel: **Add New → Project → Import** the repo → **Deploy**. Add `GROQ_API_KEY` under
*Settings → Environment Variables* and **Redeploy**.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build error about size | Make sure you did not add extra packages; `vercel.json` excludes tests/tools/videos from the function. |
| `/` shows 404 | Confirm `public/index.html` was uploaded (`.vercelignore` must not list `public`). |
| "Server unreachable" / 504 on first use | Cold start; wait a few seconds and retry. |
| Analysis pill says Local heuristic | `GROQ_API_KEY` missing or added after the last deploy → add it and run `vercel --prod` again. |
| UNKNOWN verdicts | Groq call failed (bad key, rate limit, model change). The row's detail shows the reason. |
| Camera does not start | Needs HTTPS (Vercel provides it) and your permission; try another browser. |
| Video "cannot be decoded" | Your browser cannot play that codec; re-encode to H.264 MP4 or WebM. |

## What I tested here — and what I could NOT test

Tested in this sandbox (no PyTorch installed, only the packages in `requirements.txt`):
* ONNX detections vs your PyTorch model on all 191 sample frames: 359/359 boxes matched, IoU ≥ 0.999.
* EigenCAM attention vs your original hook on all frames: max difference 0.0005 (the model is exported *fused*, as ultralytics does internally).
* Tracker + line logic on real detections at 30, 15, 10, 6 and 4.3 fps: always exactly 3 packages (IDs 1–3), at the same times as the PyTorch run (1.4 / 4.5 / 5.8 s).
* Whole dashboard in headless Chromium: sample video, every download (txt/csv/json/zips opened and validated), annotated-video recording, image tab, live camera (fake device), Groq-key path (simulated responses), zero JS errors.

NOT tested (not possible from here):
* An actual Vercel build/deployment, cold-start time, or Vercel's CPU speed (locally the server takes ~100–150 ms per frame on one core).
* The real Groq API (no access from the sandbox). The Groq code path is the same code as before, tested with a simulated client; the model name `qwen/qwen3.8-27b` and the `reasoning_effort="none"` switch come from Groq's own docs.
* Hobby usage quotas: a 60 s video at 10 fps = ~600 function calls. Check <https://vercel.com/docs/limits>; if you exceed a limit Vercel pauses that feature until the period resets. Use a lower *Analysis rate* to save usage.

## Differences from the offline app (honest list)

* **Tracker:** a lightweight tracker in the browser replaces ultralytics BoT-SORT (needs PyTorch). It matched BoT-SORT on the sample, but results on very crowded belts may differ.
* **Line crossing:** besides your ±18 px window, a package also counts when it *moves across* the line between two analysed frames (the old rule missed a package at 6 fps).
* **Analysis rate:** video is analysed at 4–15 fps (default 10) instead of every frame; first 60 s only.
* **Annotated video:** no server-side MP4. A button replays the analysed frames and records a `.webm` in your browser (plus a ZIP of annotated snapshots at each crossing). Chrome's recorded WebM has no duration metadata, so some players cannot seek it.
* **Still images:** the model sees them at a fixed 384×640 letterbox; very tall photos are detected at lower resolution.
* **Grad-CAM** from `explain.py`/`infer.py` is not included: on your model it returns an all-zero heatmap. EigenCAM (what `video_app.py` uses) is used everywhere.
* **UNKNOWN verdict:** if the vision call fails the package is marked UNKNOWN (the offline app silently showed NORMAL).
* **Groq models:** the three fallback models in `video_app.py` are retired (Groq's deprecation page); the app prefers `qwen/qwen3.8-27b` and still auto-discovers models.
* Results are AI output and can be wrong — use them as decision support.
