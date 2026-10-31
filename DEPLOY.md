# Subvox Web — deploy status

## Current deploy (2026-10-31)

- **Host:** Render, Docker runtime, service `subvox-web`
- **URL:** https://subvox-web.onrender.com (verify with `curl .../health`)
- **Plan:** free — 512 MB RAM, 0.5 vCPU, sleeps after 15 min idle
- **Image:** same image that the paid plans will run (models baked in,
  ~1.55 GB weights + torch CPU). No rebuild needed to upgrade.
- **Deploy:** `dep-dbj31ucs728c738n3ofg`, `live` since 18:21:37 UTC.

## Measured (curl, 2026-10-31)

- `/health` warm: HTTP 200 in 0.48 s, `"status": "degraded"` (all 7
  required files present, model lazy — not loaded yet),
  `missing_required: []`.
- `/` warm: HTTP 200 in 0.68 s.
- `/doctor`: torch 2.15.0+cpu, python 3.12.16. (Its RAM/disk/CPU
  numbers are the Render HOST machine's, not the container's 512 MB.)
- **Cold start** (first request after 16 min idle): HTTP 200 in
  **22.2 s**; container uptime was 10.6 s at response time.
  Second (warm) request: 0.94 s.
- `/transcribe` is NOT tested on this plan: loading the model would
  OOM the 512 MB container (see below). It needs a paid plan first.

## What works on the free plan

- `/`, `/health`, `/doctor` — honest status. Expect
  `"status": "degraded"` (files present, model not loaded — lazy load)
  or `"unavailable"` if a model file is missing.
- The deploy pipeline itself (GitHub push -> Render build -> public URL).

## What does NOT work on the free plan

- `/transcribe`, `/train`, `/test` — loading torch + the 956 MB VSR
  model needs ~2-4 GB RAM. On 512 MB the container gets OOM-killed
  and Render restarts it. `/health` recovers (lazy load), but no
  transcription happens. This is a RAM limit, not a code bug.

## To unlock transcription (needs Francis's go-ahead — paid)

Cheapest option that actually runs it:

1. **Hugging Face Spaces + PRO, $9/mo** — Docker Space, CPU Basic
   (2 vCPU, 16 GB RAM, 50 GB disk). Best specs per dollar. Needs
   Francis's HF account upgraded to PRO (new Docker Spaces require PRO
   since 2026; free accounts can only create Static Spaces).
2. **Render Standard, $25/mo** — 1 vCPU, 2 GB RAM. Tight; likely
   needs **Pro, $85/mo** (2 vCPU, 4 GB) to be safe.
3. **Cloudflare Containers, $5/mo + usage** — standard-1 (0.5 vCPU,
   4 GiB, 8 GB disk) fits, but needs wrangler + Worker/DO
   orchestration and runs ~$30/mo always-on.

Upgrade path on Render: change the plan on the existing service —
same image, no rebuild, no new repo.

## Build note (2026-10-31)

Deploys 1–2 failed (`build_failed`); Render's API exposes no build
log. The fix that made deploy 3 go live: llama-cpp-python 0.3.36
ships sdist-only on PyPI (no wheels) — pip was building llama.cpp
from source on the builder. The Dockerfile installs the project's
own prebuilt manylinux wheel from its GitHub release instead, and
`llama-cpp-python` is NOT in `requirements.txt`. Do not put it back
or the Render build will likely fail again.
