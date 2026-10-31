"""Personal language model: train on YOUR phrasing, with zero mouthing.

Wraps Lipflow's train_lm.train() (3 epochs, lr 3e-5). It fine-tunes a copy
of the subword Transformer LM on phrases.txt, runs it alongside the
general LM in beam search (weight 0.2), and saves it ONLY if your
held-out perplexity improves without general text getting much worse
(max 1.15x). That guard is in Lipflow's code and is kept here.

Where phrases come from (local only, in order):
 1. `subvox-local import-text my-writing.txt` — any .txt you point at.
 2. `subvox-local import-wispr` — reads Wispr Flow's LOCAL database
    read-only (macOS only, same as Lipflow's import_wispr). Prints only
    counts. Nothing leaves the Mac. Optional; skip it and the LM still
    trains on whatever text you import in step 1, or on the 10 setup
    sentences alone (weak, but better than nothing).
"""
from __future__ import annotations

import os
import shutil

from .paths import PHRASES


def import_text(path: str) -> int:
    """Append a .txt file's lines to phrases.txt. Returns #lines added."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if len(l.strip().split()) >= 2]
    os.makedirs(os.path.dirname(PHRASES), exist_ok=True)
    with open(PHRASES, "a", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
    return len(lines)


def import_wispr() -> dict:
    """Read Wispr Flow's local DB read-only via Lipflow's importer.
    macOS only. Returns Lipflow's stats dict (counts only).
    Lipflow's importer writes to lipflow.personal.PHRASES — redirect it to
    Subvox's phrases.txt first, or the text lands in Lipflow's home and
    train_personal_lm() never sees it."""
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    upstream = os.path.join(root, "lipflow-upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from lipflow import personal as _personal
    _personal.PHRASES = PHRASES
    from lipflow.personal import import_wispr as _import
    return _import()


def train_personal_lm(log=print) -> dict:
    """Fine-tune the personal LM. Returns Lipflow's train() report dict."""
    if not os.path.exists(PHRASES):
        raise FileNotFoundError(
            f"No phrases file yet: {PHRASES}\n"
            "Import some of your writing first: "
            "subvox-local import-text my-writing.txt  (or: subvox-local import-wispr)")
    # Lipflow's train_lm computes SPM + model paths from
    # <lipflow-upstream>/models at import time. Make sure the symlink to our
    # <repo>/models exists BEFORE importing it (load_reader() does this for
    # dictation, but train-lm never calls load_reader()).
    from .vsr import ensure_upstream_models_link
    ensure_upstream_models_link()
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    upstream = os.path.join(root, "lipflow-upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    # Lipflow's train_lm reads PHRASES from lipflow.personal — point it at ours.
    from lipflow import personal as _personal
    _personal.PHRASES = PHRASES
    import lipflow.train_lm as _tl
    _tl.PHRASES = PHRASES
    # train_lm saves to PERSONAL_LM from lipflow.paths; redirect it.
    from .paths import PERSONAL_LM as _PLM
    import lipflow.paths as _lp
    _lp.PERSONAL_LM = _PLM
    _tl.PERSONAL_LM = _PLM
    result = _tl.train()
    log(f"Personal LM: {result['sentences']} sentences, "
        f"saved={result['saved']}, {result['seconds']:.0f}s")
    return result
