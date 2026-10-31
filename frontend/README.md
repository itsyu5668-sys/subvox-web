# Subvox Web — frontend

Duna-style web frontend for Subvox, **training first**. Plain `index.html` + `app.js` +
`styles.css`, no build step. Serve it as static files from the FastAPI backend
(or any static server) and point it at the API.

## Run it

```bash
# From this directory — quick local check only (API calls will fail
# unless the backend is running and reachable at /api):
python3 -m http.server 8080 --directory ~/workspace/subvox-web/frontend
# open http://localhost:8080

# The real way: FastAPI serves this directory as static files and
# mounts its own API under /api (same origin, the default API base).
```

No build, no npm. Fonts load from Google Fonts; if offline, the system
sans-serif fallback is used.

## API base

Default is same-origin `/api`. Override, in priority order:

1. Query string: `?api=http://localhost:8000`
2. `window.SUBVOX_API_BASE` set before `app.js` loads
3. The API settings box on the Models tab (saved in `localStorage` as `subvox_api_base`)

## Endpoints the UI expects

| Call | Purpose | What the UI sends | What the UI reads |
|---|---|---|---|
| `GET /health` | Status pill in the header | — | Any JSON; `model` shown if present |
| `POST /train` | Upload training takes + start training | `multipart/form-data`: 30 video files (`video_<sentence>_<take>`, `.webm`), a `manifest` JSON field (`[{sentence, sentence_index, take, stage, field}]`), `stage=1` | Either a final result (`stock_wer`, `adapted_wer`, `kept`, `loss`) **or** `{job_id}` — then it polls `GET /train/{job_id}` every 5 s for `status` / `step` / `total_steps` / `progress` / `loss` until `done`/`failed` or a WER appears |
| `POST /train` (JSON) | Keep / discard decision | `{action: "keep"}` or `{action: "discard"}` | `message` if present; failure is reported honestly and the choice is kept as local intent only |
| `POST /test` | Score one test take | `multipart/form-data`: `video` (`.webm`), `sentence`, `take` | `raw` / `cleaned` / `wer` / `wer_stock` / `wer_adapted` / `latency_ms` |
| `POST /transcribe` | Dictate one utterance | `multipart/form-data`: `video` (`.webm`) | `raw` / `cleaned` / `latency_ms`, `hypotheses` (array of strings or `{text, score}`) |
| `POST /train-lm` | Train the cleanup language model | JSON `{sentences: [...]}` | `message`, `perplexity` if present |
| `GET /doctor` | Models tab | — | Any JSON; `status`, `device`, `models` list (`{name, size, status, notes}`) rendered if present |

Field names are read leniently (e.g. `raw_text` or `raw`, `latencyMs` or
`latency_ms`, `stock_wer` or `baseline_wer`). Anything the server does not
report is shown as **"not reported"** — the UI never invents a number.

## Screens

1. **Train** — the 10 Harvard sentences from
   `~/workspace/subvox-local/src/subvox_local/sentences.py`, each recorded
   3× via webcam (`MediaRecorder`, webm, video only — **no audio is ever
   captured**). Live camera plus a 96×96 mouth-crop preview (fixed
   lower-centre crop as a framing guide; the backend does the real crop).
   Stage 1 = sentences 1, 3, 5, 6, 9, 10 (quick adapt); Stage 2 = 2, 4, 7, 8.
   Upload needs all 30 takes. After training: stock WER vs adapted WER on
   held-out sentences, kept yes/no, keep/discard buttons.
2. **Test** — fixed known sentence *"The birch canoe slid on the smooth
   planks"*, 3 takes. Per-run raw / cleaned / WER / latency. Every test is
   logged in a table (date, stock vs adapted WER, latency, raw output),
   stored in this browser's `localStorage` only.
3. **Dictate** — hold-to-talk (pointer or Space), live transcript, raw vs
   cleaned, top-3 hypotheses, optional pasted target with a WER computed
   **locally in the browser** (never sent anywhere).
4. **Models / Doctor** — live `GET /doctor` data, model sizes, raw JSON
   dump, language-model training form, API-base setting, and the
   **LRS3 non-commercial research-only warning banner** (always visible,
   not dismissible).

## Design tokens (Duna)

Linen `#f7f7f5` canvas, Paper White `#ffffff` cards, Bone `#edece7` panels,
Aubergine Ink `#1b0624` text, Espresso `#160f0c` buttons, Warm Ash `#766a7c`,
Graphite `#444444`, Stone `#898683`, Mist `#eeeeee` inputs. Inter (GT America
substitute), display tracking −0.06em, pill buttons 999px, cards 16px radius,
images 24px radius, inputs 8px. No blue/green accent, no shadow beyond a 1px
inset ring. The sunset watercolour gradient strip is the only colour moment.

## What is still untested

- **Camera grant** — no webcam was available in the build environment, so
  `getUserMedia`, the live preview, the mouth-crop canvas loop, and
  `MediaRecorder` have not run against a real camera. The code paths are
  written defensively (empty-blob check, permission-error notice).
- **Real API calls** — the FastAPI backend was being built in parallel and
  was not running when this frontend was written, so `/health`, `/train`,
  `/test`, `/transcribe`, `/train-lm`, and `/doctor` have not been exercised
  end to end. The request/response shapes above are the contract the UI
  assumes; if the backend differs, the UI will show the backend's error
  message rather than fake a result, but the field mapping may need a tweak.
- **`/train/{job_id}` polling** — only used if `/train` returns a bare
  `job_id`; untested against a real job.
- **Keep/discard** — the `POST /train {action}` contract is assumed; if the
  backend exposes a different endpoint, the buttons will report the failure
  and keep the choice local only.

## Verified

- `node --check app.js` passes.
- All three files serve over plain HTTP and load (200, correct content types).
- No console errors could be checked without a browser — open the page and
  watch the console on first run against the real backend.
