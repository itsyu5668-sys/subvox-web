"""Model loading and the video -> mouth ROIs -> transcript pipeline.

Reuses subvox-local code, nothing reimplemented:
  * face tracking / 96x96 mouth crops at 25 fps: subvox_local.face
    (MediaPipe FaceLandmarker over lipflow-upstream/lipflow/face.py)
  * VSR: subvox_local.vsr.load_reader / read_top3 (Auto-AVSR visual-only,
    ESPnet beam search, vendored in lipflow-upstream/)
  * cleanup: subvox_local.cleanup.LocalCleaner (Qwen3-0.6B GGUF, CPU)
  * WER: subvox_local.wer.word_error_rate
  * LoRA adapt: subvox_local.adapt (rank 8, 589,824 params)

Models are loaded LAZILY on first use and cached in this module.
Nothing here ever fabricates a transcript: if the model is missing or
a clip has no detectable face, the caller gets an exception / empty
result, never invented text.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .paths import (
    DATA, LOCAL_MODELS, LOCAL_SRC, LOCAL_UPSTREAM, PERSONAL_DIR,
)

# Personal artefacts for THIS server live under data/personal, so the
# web LoRA / phrases never clobber the Mac app's local home. Must be set
# BEFORE importing subvox_local.paths (it reads the env at import time).
os.environ.setdefault("SUBVOX_LOCAL_HOME", str(PERSONAL_DIR))

for _p in (str(LOCAL_SRC), str(LOCAL_UPSTREAM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_FPS = 25
MAX_CLIP_SECONDS = 30          # longer clips are truncated, never silently
MAX_FRAMES = MODEL_FPS * MAX_CLIP_SECONDS
MIN_ROI_FRAMES = 5             # fewer usable frames than this: refuse, don't guess

_state = {
    "reader": None,
    "tracker": None,
    "cleaner": None,
    "load_error": None,
    "load_seconds": None,
    "loaded_at": None,
}
_lock = threading.RLock()      # serialises reader use (train mutates it)


def _load_all():
    """Load tracker + VSR reader + cleanup cleaner. Raises on failure;
    the exception text is stored in _state['load_error'] for /health."""
    t0 = time.time()
    from subvox_local.face import load_tracker
    from subvox_local.vsr import load_reader
    from subvox_local.cleanup import LocalCleaner

    tracker = load_tracker()                       # raises FileNotFoundError
    reader = load_reader(beam_size=10, device="cpu", personal=True)
    cleaner = LocalCleaner()
    _state.update({
        "tracker": tracker, "reader": reader, "cleaner": cleaner,
        "load_seconds": round(time.time() - t0, 1),
        "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load_error": None,
    })
    return _state


def get_state():
    """Load on first use (thread-safe). Returns _state dict."""
    with _lock:
        if _state["reader"] is not None:
            return _state
        if _state["load_error"] is not None:
            raise RuntimeError(_state["load_error"])
        try:
            return _load_all()
        except Exception as e:  # noqa: BLE001 - surfaced honestly via /health
            _state["load_error"] = f"{type(e).__name__}: {e}"
            raise


def try_get_state():
    """_state without triggering a load (for /health before first use)."""
    return _state


def cleanup_backend() -> str:
    c = _state.get("cleaner")
    if c is None:
        return "unknown (model not loaded yet)"
    try:
        return "gguf-qwen3-0.6b" if c.available() else "basic-rules"
    except Exception:
        return "basic-rules"


def video_to_rois(video_path: str, tracker=None):
    """Decode a video file to (T,96,96) uint8 mouth ROIs at 25 fps.

    Returns (rois, info). Raises ValueError with a plain explanation if
    the video cannot be decoded or no face is found in any frame.
    info: {frames, fps, face_frames, truncated}
    """
    import cv2
    from subvox_local.face import mouth_rois
    from lipflow.vsr import LipReader

    tr = tracker or get_state()["tracker"]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 240:
        fps = 25.0
    frames, anchors, stamps = [], [], []
    idx = 0
    truncated = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= MAX_FRAMES:
            truncated = True
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ts_ms = int(idx * 1000.0 / fps)
        try:
            obs = tr.detect(frame, ts_ms)
        except Exception:
            obs = None
        frames.append(gray)
        anchors.append(obs.anchors if obs is not None else None)
        stamps.append(idx / fps)
        idx += 1
    cap.release()
    face_frames = sum(1 for a in anchors if a is not None)
    if not frames:
        raise ValueError("Video decoded to zero frames — empty or corrupt file.")
    if face_frames == 0:
        raise ValueError(
            f"No face detected in any of {len(frames)} frames — cannot lip-read "
            "this clip. Re-record facing the camera, well lit, mouth visible.")
    resampled_idx = LipReader.resample(stamps, len(frames))
    frames = [frames[i] for i in resampled_idx]
    anchors = [anchors[i] for i in resampled_idx]
    rois = mouth_rois(frames, anchors)
    if rois is None or len(rois) < MIN_ROI_FRAMES:
        raise ValueError(
            f"Only {0 if rois is None else len(rois)} usable mouth frames — "
            "need at least 5. Face was lost for most of the clip.")
    return rois.astype(np.uint8), {
        "frames": int(len(rois)), "fps": round(float(fps), 2),
        "face_frames": int(face_frames), "truncated": truncated,
        "source_frames": len(stamps)}


def rois_from_npz(npz_path: str) -> np.ndarray:
    """Accept pre-extracted ROIs (T,96,96) uint8 saved as .npz {'rois': ...}."""
    data = np.load(npz_path)
    if "rois" not in data:
        raise ValueError("NPZ must contain a 'rois' array of shape (T, 96, 96).")
    rois = np.asarray(data["rois"])
    if rois.ndim != 3 or rois.shape[1:] != (96, 96):
        raise ValueError(f"rois must be (T, 96, 96), got shape {rois.shape}.")
    if len(rois) < MIN_ROI_FRAMES:
        raise ValueError(f"Only {len(rois)} ROI frames — need at least {MIN_ROI_FRAMES}.")
    return rois.astype(np.uint8)


def beam_top3_with_scores(reader, rois):
    """Encode + beam search, returning [{text, score}] best-first.

    Scores are the raw ESPnet beam log-scores (higher / less negative =
    better). They are NOT probabilities; confidence is computed from
    them by the caller via softmax over the top-3.
    """
    from espnet.asr.asr_utils import add_results_to_json
    enc = reader.encode(rois)
    for module in reader.beam.full_scorers["decoder"].modules():
        if hasattr(module, "_mem_kv"):
            del module._mem_kv
    hyps = reader.beam(enc)
    out = []
    seen = set()
    for h in hyps[:3]:
        text = reader._clean(add_results_to_json([h.asdict()], reader.token_list))
        if text in seen:
            continue
        seen.add(text)
        try:
            score = float(h.score)
        except Exception:
            score = None
        out.append({"text": text, "score": score})
    return out


def transcribe_rois(rois, recent=None, custom_words=None):
    """Run VSR + cleanup on already-extracted ROIs.

    Returns dict with raw/cleaned/top3/confidence/latency. Never invents
    text: empty beam output yields empty strings.
    """
    t0 = time.time()
    with _lock:
        st = get_state()
        reader, cleaner = st["reader"], st["cleaner"]
        top3 = beam_top3_with_scores(reader, rois)
        raw = top3[0]["text"] if top3 else ""
        try:
            cleaned = cleaner.clean([t["text"] for t in top3] or [raw],
                                    recent=recent, custom_words=custom_words)
        except Exception:
            from subvox_local.cleanup import basic_cleanup
            cleaned = basic_cleanup(raw)
    confidence = None
    scores = [t["score"] for t in top3 if t.get("score") is not None]
    if len(scores) >= 2:
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        confidence = round(exps[0] / sum(exps), 3)
    return {
        "raw": raw,
        "cleaned": cleaned,
        "top3": top3,
        "confidence": confidence,   # softmax over top-3 beam scores; None if <2 hyps
        "latency_seconds": round(time.time() - t0, 2),
        "cleanup_backend": cleanup_backend(),
        "frames": int(len(rois)),
    }


def wer(ref: str, hyp: str) -> float:
    from subvox_local.wer import word_error_rate
    return word_error_rate(ref, hyp)
