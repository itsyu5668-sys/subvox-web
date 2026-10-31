"""Learning from corrections (Lipflow's rule, kept as-is).

When you fix a word Subvox Local typed — within about 30 seconds — it
saves that clip with your corrected sentence, and the next
`subvox-local adapt-more` round trains on it. Only the text field it
pasted into is read. Nothing is uploaded. Switch off with
settings.json: {"learn_from_corrections": false}.

Storage: corrections.jsonl in the user data dir, one JSON object per line:
{"ts": ..., "expected": "...", "got": "...", "clip": "<path to .npz or null>"}
"""
from __future__ import annotations

import json
import os
import time

from .paths import CORRECTIONS

WINDOW_SECONDS = 30


def record_correction(expected: str, got: str, clip_path: str | None = None) -> str | None:
    """Append a correction. Returns the line written, or None if empty."""
    expected, got = (expected or "").strip(), (got or "").strip()
    if not expected or expected == got:
        return None
    os.makedirs(os.path.dirname(CORRECTIONS), exist_ok=True)
    row = {"ts": time.time(), "expected": expected, "got": got, "clip": clip_path}
    with open(CORRECTIONS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return json.dumps(row)


def recent_corrections(within_seconds: int = WINDOW_SECONDS) -> list[dict]:
    if not os.path.exists(CORRECTIONS):
        return []
    cutoff = time.time() - within_seconds
    out = []
    for line in open(CORRECTIONS, encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ts", 0) >= cutoff:
            out.append(row)
    return out


def all_correction_clips() -> list[dict]:
    """Clips with corrected text, for the next adapt round.
    Format matches adapt clips: {'rois': ..., 'text': ...} — rois are
    loaded lazily by the caller from the .npz path."""
    if not os.path.exists(CORRECTIONS):
        return []
    out = []
    for line in open(CORRECTIONS, encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("clip") and row.get("expected"):
            out.append({"path": row["clip"], "text": row["expected"]})
    return out
