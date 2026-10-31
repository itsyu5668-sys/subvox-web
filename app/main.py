"""Subvox web backend API (FastAPI, torch CPU only).

Reuses the lip model and weights from ~/workspace/subvox-local/ — no
model rebuilt, no weights copied (the 1.7 GB AV model and 1.8 GB
AV-HuBERT backup are deliberately excluded).

Endpoints:
  GET  /health      model loaded? which files present — never a fake ok
  POST /transcribe  video clip (mp4/webm/mov) or .npz ROIs -> transcript
  POST /train       labelled clips -> LoRA rank 8, keep-only-if-better
  POST /train-lm    import text, train personal LM
  GET  /doctor      model inventory with real sizes + sha256
  POST /test        known Harvard sentence 3x, per-run WER + latency

LRS3 weights are NON-COMMERCIAL RESEARCH ONLY — see README and the
`license_note` field on /health and /doctor.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import model as M
from .paths import (
    DATA, LORA_DIR, LRS3_LICENSE_NOTE, MAC_PERSONAL_LORA, PERSONAL_DIR,
    TEST_CLIPS_DIR, UPLOAD_DIR, disk_and_ram, inventory,
)
from .model import MAX_CLIP_SECONDS

_started_at = time.time()
TEST_SENTENCE = "The birch canoe slid on the smooth planks"

app = FastAPI(
    title="Subvox Web Backend",
    description="Lip-reading dictation API. Auto-AVSR visual-only (LRS3) + "
                "Qwen3-0.6B GGUF cleanup, torch CPU only. "
                "LRS3 weights are non-commercial research only.",
    version="0.1.0",
)
# Local-dev frontend (Duna HTML / future web UI) talks to this API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

for _d in (DATA, LORA_DIR, PERSONAL_DIR, TEST_CLIPS_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- helpers
def _save_upload(f: UploadFile) -> Path:
    suffix = Path(f.filename or "upload").suffix or ".bin"
    fd, path = tempfile.mkstemp(prefix="subvox-", suffix=suffix, dir=str(UPLOAD_DIR))
    with os.fdopen(fd, "wb") as out:
        shutil.copyfileobj(f.file, out)
    return Path(path)


def _decode_upload_to_rois(f: UploadFile):
    """UploadFile -> (rois, info). .npz with a 'rois' array, else video."""
    path = _save_upload(f)
    try:
        if path.suffix.lower() == ".npz":
            rois = M.rois_from_npz(str(path))
            info = {"frames": int(len(rois)), "fps": 25,
                    "face_frames": int(len(rois)), "truncated": False,
                    "source_frames": int(len(rois)), "input": "npz-rois"}
        else:
            rois, info = M.video_to_rois(str(path))
            info["input"] = "video"
        return rois, info
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def _parse_json_field(raw: str | None, name: str):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Field {name!r} is not valid JSON: {e}")


# ---------------------------------------------------------------- root/health/doctor
@app.get("/")
def root():
    return {
        "service": "subvox-web",
        "version": "0.1.0",
        "endpoints": ["/health", "/transcribe", "/train", "/train-lm",
                      "/doctor", "/test"],
        "license_note": LRS3_LICENSE_NOTE,
    }


@app.get("/health")
def health():
    inv = inventory(with_hash=False)
    st = M.try_get_state()
    loaded = st["reader"] is not None
    if not inv["all_required_present"]:
        status = "unavailable"
    elif loaded:
        status = "ok"
    else:
        status = "degraded"   # files present, model not loaded yet (lazy load)
    return {
        "status": status,
        "model_loaded": loaded,
        "load_error": st.get("load_error"),
        "load_seconds": st.get("load_seconds"),
        "cleanup_backend": M.cleanup_backend() if loaded else "not loaded yet",
        "files_present": {k: v["present"] for k, v in inv["files"].items()},
        "files_size_mb": {k: v["size_mb"] for k, v in inv["files"].items()
                          if v["present"]},
        "missing_required": inv["missing_required"],
        "uptime_seconds": round(time.time() - _started_at, 1),
        "license_note": LRS3_LICENSE_NOTE,
    }


@app.get("/doctor")
def doctor():
    inv = inventory(with_hash=True)   # real sha256, takes a few seconds on 1.5 GB
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = "not importable"
    import sys
    personal_lora = PERSONAL_DIR / "models" / "vsr_face_lora.pt"
    phrases = PERSONAL_DIR / "phrases.txt"
    upstream_link = Path(M.LOCAL_UPSTREAM) / "models"
    return {
        "inventory": inv,
        "license_note": LRS3_LICENSE_NOTE,
        "python": sys.version.split()[0],
        "torch": torch_version,
        "device": "cpu",
        "system": disk_and_ram(),
        "upstream_models_symlink": {
            "path": str(upstream_link),
            "is_symlink": upstream_link.is_symlink(),
            "target": os.readlink(upstream_link) if upstream_link.is_symlink() else None,
        },
        "personal_lora": {
            "web_data_lora": str(LORA_DIR / "vsr_face_lora.pt"),
            "web_data_lora_present": (LORA_DIR / "vsr_face_lora.pt").exists(),
            "web_personal_lora": str(personal_lora),
            "web_personal_lora_present": personal_lora.exists(),
            "mac_app_lora_path": str(MAC_PERSONAL_LORA),
            "mac_app_lora_present_here": MAC_PERSONAL_LORA.exists(),
        },
        "phrases_file": str(phrases),
        "phrases_lines": (sum(1 for _ in open(phrases, encoding="utf-8"))
                          if phrases.exists() else 0),
        "max_clip_seconds": MAX_CLIP_SECONDS,
    }


# ---------------------------------------------------------------- transcribe
@app.post("/transcribe")
async def transcribe(
    video: UploadFile = File(..., description="mp4/webm/mov clip, or .npz with 'rois' (T,96,96)"),
    recent: str | None = Form(None, description="JSON list of recent sentences for cleanup context"),
    custom_words: str | None = Form(None, description="JSON list of custom words"),
):
    recent_list = _parse_json_field(recent, "recent") or []
    words_list = _parse_json_field(custom_words, "custom_words") or []
    try:
        rois, info = _decode_upload_to_rois(video)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, f"Model could not load: {e}")
    try:
        result = M.transcribe_rois(rois, recent=recent_list, custom_words=words_list)
    except RuntimeError as e:
        raise HTTPException(503, f"Model could not load: {e}")
    if not result["raw"]:
        # Beam search returned nothing — report it, never invent text.
        raise HTTPException(422, "Beam search returned no hypothesis for this clip.")
    result["video"] = info
    return result


# ---------------------------------------------------------------- train (LoRA)
@app.post("/train")
async def train(
    clips: list[UploadFile] = File(..., description="Labelled video clips (or .npz ROIs)"),
    manifest: str | None = Form(None, description='JSON: [{"file": name, "text": "...", "split": "train"|"heldout"}]'),
    transcripts: str | None = Form(None, description="Alt to manifest: JSON {filename: text}"),
    held_out: str | None = Form(None, description="Alt to manifest: JSON [filenames held out]"),
    steps: int = Form(150, description="Max LoRA steps, capped at 150 on CPU"),
):
    steps = max(1, min(int(steps), 150))
    manifest_list = _parse_json_field(manifest, "manifest")
    if manifest_list is None:
        tr = _parse_json_field(transcripts, "transcripts") or {}
        ho = set(_parse_json_field(held_out, "held_out") or [])
        manifest_list = [{"file": k, "text": v,
                          "split": "heldout" if k in ho else "train"}
                         for k, v in tr.items()]
    by_file = {m["file"]: m for m in manifest_list if m.get("file") and m.get("text")}
    if not by_file:
        raise HTTPException(400, "No labelled clips: supply a manifest (or transcripts) "
                                 "mapping each clip filename to its text.")
    train_clips, held_out_clips = [], []
    try:
        for f in clips:
            m = by_file.get(f.filename)
            if m is None:
                continue  # unlabelled upload: skipped, not guessed
            rois, _info = _decode_upload_to_rois(f)
            clip = {"rois": rois, "text": m["text"]}
            (held_out_clips if m.get("split") == "heldout" else train_clips).append(clip)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, f"Model could not load: {e}")
    if not train_clips:
        raise HTTPException(400, "No train-split clips decoded. Check filenames match the manifest.")
    if not held_out_clips:
        raise HTTPException(422, "Keep-only-if-better needs held-out clips: mark at least "
                                 "one clip split='heldout' (never trained on). Refusing to "
                                 "train without an evaluation set.")
    # NOTE: holds the reader lock for the whole run (CPU training is slow,
    # up to ~10+ min for 150 steps). /transcribe waits rather than racing
    # a model whose weights are being mutated.
    try:
        with M._lock:
            st = M.get_state()
            from subvox_local.adapt import LORA_R, fast_setup
            from subvox_local.paths import PERSONAL_VSR_LORA
            report = fast_setup(st["reader"], train_clips, held_out_clips,
                                steps=steps, log=print)
    except RuntimeError as e:
        raise HTTPException(503, f"Model could not load: {e}")
    train_info = report.get("train") or {}
    out = {
        "stock_wer": report.get("stock_wer"),
        "adapted_wer": report.get("adapted_wer"),
        "kept": bool(report.get("kept")),
        "steps": train_info.get("steps", 0),
        "seconds": train_info.get("seconds"),
        "loss": train_info.get("loss_end"),
        "loss_start": train_info.get("loss_start"),
        "trainable_params": report.get("trainable_params"),
        "lora_rank": LORA_R,
        "n_train_clips": len(train_clips),
        "n_held_out_clips": len(held_out_clips),
        "stock_rows": report.get("stock_rows"),
        "adapted_rows": report.get("adapted_rows"),
        "lora_path": None,
        "lora_copies": [],
    }
    if report.get("kept") and report.get("lora_path"):
        src = Path(report["lora_path"])  # data/personal/models/vsr_face_lora.pt
        copies = []
        for dest in (LORA_DIR / "vsr_face_lora.pt", MAC_PERSONAL_LORA):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            copies.append(str(dest))
        out["lora_path"] = str(src)
        out["lora_copies"] = copies
    return out


# ---------------------------------------------------------------- train-lm
@app.post("/train-lm")
async def train_lm(
    text_file: UploadFile | None = File(None, description=".txt of your writing / phrases"),
    text: str | None = Form(None, description="Raw text to import (alternative to a file)"),
):
    if text_file is None and not (text or "").strip():
        raise HTTPException(400, "Supply text_file (.txt upload) or text (form field).")
    tmp = None
    try:
        if text_file is not None:
            tmp = _save_upload(text_file)
            path = str(tmp)
        else:
            fd, p = tempfile.mkstemp(prefix="subvox-text-", suffix=".txt", dir=str(UPLOAD_DIR))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            tmp = Path(p)
            path = str(tmp)
        from subvox_local.lm_train import import_text, train_personal_lm
        added = import_text(path)
        if added == 0:
            raise HTTPException(422, "No usable lines imported (need lines of 2+ words).")
        report = train_personal_lm()
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001 - training failures reported, not hidden
        raise HTTPException(503, f"Personal LM training failed: {type(e).__name__}: {e}")
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
    report["phrases_added"] = added
    report["phrases_file"] = str(PERSONAL_DIR / "phrases.txt")
    report["personal_lm_path"] = str(PERSONAL_DIR / "models" / "lm_phrasing.pth")
    return report


# ---------------------------------------------------------------- test
@app.post("/test")
async def test(
    clips: list[UploadFile] | None = File(None, description="Up to 3 clips of the test sentence (else data/test_clips/ is used)"),
    expected: str = Form(TEST_SENTENCE),
):
    paths: list[Path] = []
    tmp_paths: list[Path] = []
    if clips:
        for f in clips[:3]:
            p = _save_upload(f)
            paths.append(p)
            tmp_paths.append(p)
    else:
        for ext in ("*.mp4", "*.webm", "*.mov", "*.npz"):
            paths.extend(sorted(TEST_CLIPS_DIR.glob(ext))[:3])
    if not paths:
        raise HTTPException(
            503,
            "No test clips available: upload up to 3 clips of "
            f"'{expected}' (field 'clips'), or place them in data/test_clips/. "
            "Nothing was fabricated — record the sentence 3x on camera first.")
    runs = []
    try:
        for p in paths:
            t0 = time.time()
            if p.suffix.lower() == ".npz":
                rois = M.rois_from_npz(str(p))
                info = {"frames": int(len(rois)), "input": "npz-rois"}
            else:
                rois, info = M.video_to_rois(str(p))
            result = M.transcribe_rois(rois)
            runs.append({
                "clip": p.name,
                "expected": expected,
                "raw": result["raw"],
                "cleaned": result["cleaned"],
                "wer_raw": round(M.wer(expected, result["raw"]), 3) if result["raw"] else None,
                "wer_cleaned": round(M.wer(expected, result["cleaned"]), 3) if result["cleaned"] else None,
                "latency_seconds": result["latency_seconds"],
                "frames": result["frames"],
            })
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, f"Model could not load: {e}")
    finally:
        for p in tmp_paths:
            try:
                p.unlink()
            except OSError:
                pass
    wers = [r["wer_raw"] for r in runs if r["wer_raw"] is not None]
    return {
        "sentence": expected,
        "runs": runs,
        "n_runs": len(runs),
        "mean_wer_raw": round(sum(wers) / len(wers), 3) if wers else None,
        "mean_latency_seconds": round(
            sum(r["latency_seconds"] for r in runs) / len(runs), 2) if runs else None,
    }


@app.on_event("startup")
def _startup():
    if os.environ.get("SUBVOX_PRELOAD") == "1":
        threading.Thread(target=_preload, daemon=True).start()


def _preload():
    try:
        M.get_state()
        print("Subvox: model preloaded.")
    except Exception as e:  # noqa: BLE001 - preload failure is visible in /health
        print(f"Subvox: preload failed (see /health): {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
