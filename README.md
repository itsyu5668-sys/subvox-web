# Subvox Web

Silent lip-reading dictation as a public web API. Camera in the browser,
transcript out. FastAPI + torch CPU only.

- **VSR:** Auto-AVSR visual-only, LRS3 (19.1% WER), ESPnet beam search
- **Cleanup:** Qwen3-0.6B GGUF (CPU) over the top-3 beam hypotheses
- **Face:** MediaPipe FaceLandmarker, 96x96 mouth ROIs at 25 fps
- **Train:** LoRA rank 8, keep-only-if-better on held-out clips

## Endpoints

- `GET /` — service info
- `GET /health` — honest status: `ok` / `degraded` (files present, model
  not loaded yet) / `unavailable` (model files missing), never a fake ok
- `GET /doctor` — model inventory with real sizes + sha256
- `POST /transcribe` — video clip (mp4/webm/mov) or `.npz` ROIs
- `POST /train` — labelled clips, LoRA, keep-only-if-better
- `POST /train-lm` — import text, train personal LM
- `POST /test` — known Harvard sentence 3x, per-run WER + latency

## License note

LRS3 weights (Auto-AVSR visual-only model, `Amanvir/LRS3_V_WER19.1`)
are **non-commercial research only**. Do not use this server
commercially without replacing the VSR weights with a commercially
licensed model.

## Deploy

Live host: Render, Docker runtime — see `DEPLOY.md` for the current
plan, cost, and what still blocks full transcription.

Local run:

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
bash scripts/download-models-web.sh   # ~1.55 GB, public URLs, no token
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Model files are **not** committed: the Docker build downloads them from
the public sources listed in `scripts/download-models-web.sh`.
Vendored code (`src/subvox_local/`, `lipflow-upstream/`) is copied from
the Subvox Local repo; it is small (~2 MB) and is the same code the Mac
app runs.
