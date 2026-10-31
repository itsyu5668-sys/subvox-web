"""VSR wrapper: load Lipflow's Auto-AVSR LipReader from the vendored copy.

The heavy model code (ESPnet beam search, Conformer encoder) lives in
lipflow-upstream/ (MIT, Amy Zhou 2026, vendored ESPnet Apache 2.0).
This module only wires it to Subvox Local's model paths and adds the
top-3 beam interface the cleanup step needs.

INTEL MAC: Francis's Mac is an Intel Core i5, NOT Apple Silicon.
There is no MPS and no CUDA on it, so torch is forced to CPU here
(device="cpu" is the default; "auto" resolves to CPU, never MPS).
Lipflow's measured numbers on an M4 Pro for a 9 s utterance
(lipflow-upstream/README.md, "Speed"): encoder 0.16 s on the Apple
GPU vs 1.7 s on CPU, beam search about 0.8-1.6 s on CPU, about 1-2 s
from release to text on the M4 Pro. On an Intel i5 the encoder will
be at least that 1.7 s CPU figure and plausibly several seconds —
the i5 is a much weaker CPU than an M4 Pro — so expect noticeably
slower dictation than on Apple Silicon. No i5 latency number is
claimed here; measure it on Francis's Mac (docs/INTEL_MAC.md).

UNTESTED ON INTEL MAC: LipReader loads torch + the 955 MB checkpoint.
It has not been run on an Intel Mac in this build (Linux VM, CPU only).
"""
from __future__ import annotations

import os
import sys

from .paths import MODELS

# Intel i5: CPU only. MPS exists only on Apple Silicon, CUDA only on
# NVIDIA GPUs — neither is present on Francis's Mac, so never ask for them.
DEFAULT_DEVICE = "cpu"


def _add_upstream():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    upstream = os.path.join(root, "lipflow-upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    return upstream


def ensure_upstream_models_link():
    """Lipflow's code reads models from <lipflow-upstream>/models.
    Our downloads live in <repo>/models, so link them (no copy, no 1 GB duplicate).
    Also used by train-lm, which never calls load_reader()."""
    upstream = _add_upstream()
    link = os.path.join(upstream, "models")
    if os.path.islink(link):
        if os.readlink(link) != MODELS:
            raise RuntimeError(f"{link} points elsewhere: {os.readlink(link)}")
    elif os.path.lexists(link):
        # A real directory (or broken symlink) — never silently use the wrong models.
        raise RuntimeError(f"{link} exists and is not a symlink to {MODELS}. "
                           "Move it aside and re-run.")
    else:
        os.symlink(MODELS, link)
    return upstream


def load_reader(beam_size: int = 10, device: str = "cpu", personal: bool = True):
    """Return a Lipflow LipReader using models/ in this repo.
    Raises FileNotFoundError with the exact missing file if a model is absent.

    device: "cpu" (default, and the only real option on an Intel Mac).
    "auto" is accepted for backwards compatibility and resolves to CPU.
    Any MPS/CUDA request is rejected loudly rather than silently falling
    back to a slower path the caller did not ask for.

    Before constructing the reader, Lipflow's personal-model paths are
    redirected to Subvox Local's home: otherwise LipReader would load a
    personal LM / full-frontend VSR from Lipflow's home and never see ours.
    If a stage-1/2 LoRA file exists it is applied on top of the base model.
    """
    from .paths import PERSONAL_LM, PERSONAL_VSR, PERSONAL_VSR_LORA
    if device == "auto":
        device = "cpu"
    if device not in ("cpu",):
        raise ValueError(
            f"device={device!r} is not available on an Intel Mac (no MPS/CUDA). "
            "Use device='cpu'.")
    ensure_upstream_models_link()
    for need in ("vsr/model.json", "vsr/model.pth", "lm/model.json", "lm/model.pth"):
        p = os.path.join(MODELS, need)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing model file: {p}\n"
                                    "Run: bash scripts/download-models.sh")
    # Lipflow reads PERSONAL_VSR / PERSONAL_LM from lipflow.paths at
    # LipReader construction time — redirect before constructing.
    import lipflow.paths as _lp
    _lp.PERSONAL_VSR = PERSONAL_VSR
    _lp.PERSONAL_LM = PERSONAL_LM
    from lipflow.vsr import LipReader
    reader = LipReader(device=device, beam_size=beam_size, personal=personal)
    if personal and os.path.exists(PERSONAL_VSR_LORA):
        from .adapt import apply_lora, load_lora
        adapters, _n = apply_lora(reader)
        load_lora(reader, adapters, PERSONAL_VSR_LORA)
        reader.lora_adapters = adapters
    return reader


def read_top3(reader, rois) -> list[str]:
    """Beam-search top-3 hypotheses, best first. Used by cleanup rerank.
    Beam size is fixed at load_reader() time; there is no per-call override."""
    enc = reader.encode(rois)
    out = reader.beam_search(enc, nbest=3)
    if isinstance(out, str):
        return [out]
    return list(out)
