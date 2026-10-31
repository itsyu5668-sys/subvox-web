"""CLI: subvox-local doctor | setup-fast | adapt-more | train-lm |
import-text | import-wispr | file

All commands are local-only. No network calls except model downloads
in `setup` (explicit) — dictation itself never touches the network.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _print_report(d: dict):
    print(json.dumps(d, indent=2, default=str))


def cmd_doctor(_args):
    """Check models, camera, and platform. No inference.
    INTEL MAC: checks the GGUF cleanup model + llama-cpp, not MLX
    (MLX is Apple Silicon only and does not run on Francis's Intel i5).
    The old MLX dirs are listed for reference only."""
    from .paths import MODELS, FACE_LANDMARKER, QWEN_06B, QWEN_17B, QWEN_06B_GGUF, QWEN_17B_GGUF
    checks = {}
    for label, p in [
        ("vsr/model.json", os.path.join(MODELS, "vsr", "model.json")),
        ("vsr/model.pth", os.path.join(MODELS, "vsr", "model.pth")),
        ("lm/model.json", os.path.join(MODELS, "lm", "model.json")),
        ("lm/model.pth", os.path.join(MODELS, "lm", "model.pth")),
        ("lm/unigram5000.model", os.path.join(MODELS, "lm", "unigram5000.model")),
        ("face_landmarker.task", FACE_LANDMARKER),
        ("av/config.json", os.path.join(MODELS, "av", "config.json")),
        ("av/model.safetensors", os.path.join(MODELS, "av", "model.safetensors")),
        ("qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf (Intel cleanup, primary)", QWEN_06B_GGUF),
        ("qwen3-1.7b-gguf/Qwen3-1.7B-Q4_K_M.gguf (Intel cleanup, fallback)", QWEN_17B_GGUF),
        ("qwen3-0.6b-4bit-mlx (Apple Silicon only, NOT usable on Intel)", QWEN_06B),
        ("qwen3-1.7b-4bit-mlx (Apple Silicon only, NOT usable on Intel)", QWEN_17B),
        ("avhubert/model.pt", os.path.join(MODELS, "avhubert", "model.pt")),
    ]:
        if os.path.isdir(p):
            n = len(os.listdir(p))
            size = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)
                       if os.path.isfile(os.path.join(p, f)))
            checks[label] = f"dir, {n} files, {size / 1e6:.0f} MB"
        elif os.path.exists(p):
            checks[label] = f"ok, {os.path.getsize(p) / 1e6:.1f} MB"
        else:
            checks[label] = "MISSING — run: bash scripts/download-models.sh"
    checks["platform"] = sys.platform
    checks["python"] = sys.version.split()[0]
    try:
        import torch
        checks["torch"] = torch.__version__
        checks["mps_available"] = str(torch.backends.mps.is_available())
        checks["cuda_available"] = str(torch.cuda.is_available())
    except Exception as e:
        checks["torch"] = f"not importable ({e})"
    try:
        import llama_cpp  # noqa: F401
        checks["llama_cpp"] = "importable (GGUF cleanup backend, CPU)"
    except Exception:
        checks["llama_cpp"] = "not importable — run: uv sync (llama-cpp-python)"
    checks["mlx_lm"] = ("not used on Intel Mac — MLX is Apple Silicon only; "
                        "cleanup is GGUF via llama-cpp (see llama_cpp above)")
    _print_report(checks)
    return 0


def cmd_setup_fast(args):
    """Interactive fast setup: 6 sentences -> LoRA adapt ->
    held-out check -> personal LM. Stage 2 continues in background.
    INTEL CPU: this is SLOW on an i5 — the old "~2 min" claim does NOT
    apply (see docs/INTEL_MAC.md). Requires camera + models + torch CPU."""
    from .sentences import QUICK_6, HELD_OUT_STAGE1
    print("Stage 1 — quick adapt: mouth these 6 sentences, one at a time.")
    from .sentences import QUICK_6, HELD_OUT_STAGE1
    print("Stage 1 — quick adapt: mouth these 6 sentences, one at a time.")
    for i, s in enumerate(QUICK_6, 1):
        print(f"  {i}. {s}")
    print("Held out (never trained on in stage 1):")
    for s in HELD_OUT_STAGE1:
        print(f"  - {s}")
    if args.dry_run:
        print("[dry-run] No camera, no training. Sentences listed above.")
        return 0
    # Heavy imports (torch, cv2) only after the dry-run return.
    from .capture import Capture
    from .face import load_tracker
    from .vsr import load_reader
    from .adapt import fast_setup
    from .paths import CLIPS_DIR
    import numpy as np
    os.makedirs(CLIPS_DIR, exist_ok=True)
    tracker = load_tracker()
    reader = load_reader(beam_size=10)
    cap = Capture(camera=args.camera).open()
    clips, held = [], []
    try:
        for s in QUICK_6:
            input(f"\nPress Enter, then mouth: {s}\n")
            rois, _ = cap.record(6.0, tracker)
            if rois is not None:
                clips.append({"rois": rois, "text": s})
                np.savez_compressed(os.path.join(CLIPS_DIR, f"q{len(clips)}.npz"),
                                    rois=rois, text=s)
        for s in HELD_OUT_STAGE1:
            input(f"\nHeld-out — press Enter, then mouth: {s}\n")
            rois, _ = cap.record(6.0, tracker)
            if rois is not None:
                held.append({"rois": rois, "text": s})
                np.savez_compressed(os.path.join(CLIPS_DIR, f"h{len(held)}.npz"),
                                    rois=rois, text=s)
    finally:
        cap.close()
    report = fast_setup(reader, clips, held, log=print)
    _print_report(report)
    print("\nStage 2 (background): the other 4 sentences + corrections.")
    print("Run: subvox-local adapt-more   (continues LoRA from stage 1)")
    print("Then train the personal LM: subvox-local train-lm  (needs phrases.txt)")
    return 0


def _load_npz_clip(path):
    import numpy as np
    d = np.load(path, allow_pickle=True)
    return {"rois": d["rois"], "text": str(d["text"])}


def _saved_clips(prefix):
    import glob
    from .paths import CLIPS_DIR
    return [_load_npz_clip(p) for p in sorted(glob.glob(os.path.join(CLIPS_DIR, f"{prefix}*.npz")))]


def cmd_adapt_more(args):
    """Stage 2: continue LoRA from stage 1 on the remaining 4 sentences
    + stage-1 clips (to avoid forgetting) + any correction clips,
    re-evaluate on the saved held-out clips, keep only if better.
    INTEL CPU: runs at the 150-step CPU default; use --steps to change.
    UNTESTED ON INTEL MAC: needs camera + torch CPU on Francis's i5."""
    from .sentences import STAGE2
    print("Stage 2 — background adapt: mouth these 4 sentences, one at a time.")
    for i, s in enumerate(STAGE2, 1):
        print(f"  {i}. {s}")
    if args.dry_run:
        print("[dry-run] No camera, no training. Sentences listed above.")
        return 0
    # Heavy imports (torch, cv2) only after the dry-run return.
    from .capture import Capture
    from .face import load_tracker
    from .vsr import load_reader
    from .adapt import apply_lora, load_lora, train_lora, evaluate, save_lora, remove_lora
    from .corrections import all_correction_clips
    from .paths import CLIPS_DIR, PERSONAL_VSR_LORA
    import numpy as np

    os.makedirs(CLIPS_DIR, exist_ok=True)
    tracker = load_tracker()
    reader = load_reader(beam_size=10)
    cap = Capture(camera=args.camera).open()
    stage2 = []
    try:
        for s in STAGE2:
            input(f"\nPress Enter, then mouth: {s}\n")
            rois, _ = cap.record(6.0, tracker)
            if rois is not None:
                stage2.append({"rois": rois, "text": s})
                np.savez_compressed(os.path.join(CLIPS_DIR, f"s{len(stage2)}.npz"),
                                    rois=rois, text=s)
    finally:
        cap.close()
    # Training set: new stage-2 clips + saved stage-1 clips + corrections.
    clips = list(stage2) + _saved_clips("q")
    for c in all_correction_clips():
        try:
            clips.append(_load_npz_clip(c["path"]))
        except Exception as e:
            print(f"Skipping correction clip {c['path']}: {e}")
    if not clips:
        print("No clips to train on — run setup-fast first.")
        return 1
    held = _saved_clips("h")
    adapters, n_params = apply_lora(reader)
    if os.path.exists(PERSONAL_VSR_LORA):
        load_lora(reader, adapters, PERSONAL_VSR_LORA)
        print(f"Continuing from stage-1 LoRA: {PERSONAL_VSR_LORA}")
    else:
        print("No stage-1 LoRA found — starting fresh from the stock model.")
    print(f"LoRA adapters: {len(adapters)} linears, {n_params:,} trainable params")
    stock = evaluate(reader, held) if held else None
    if stock:
        print(f"Before stage 2, WER on {stock['n']} held-out clips: {stock['wer_mean']:.3f}")
    train_info = train_lora(reader, clips, adapters, steps=args.steps, log=print)
    print(f"Trained {train_info['steps']} steps in {train_info['seconds']:.0f}s "
          f"on {len(clips)} clips.")
    if held:
        adapted = evaluate(reader, held)
        print(f"After stage 2, WER on held-out clips: {adapted['wer_mean']:.3f}")
        if adapted["wer_mean"] < stock["wer_mean"]:
            path = save_lora(adapters, PERSONAL_VSR_LORA)
            print(f"Kept — improved held-out WER -> {path}")
        else:
            remove_lora(reader, adapters)
            print("No improvement on held-out clips — stage-2 LoRA discarded, "
                  "stage-1 file (if any) untouched.")
            return 0
    else:
        path = save_lora(adapters, PERSONAL_VSR_LORA)
        print("No held-out clips saved (run setup-fast first for a keep-gate). "
              f"Saved unconditionally -> {path}")
    _print_report({"steps": train_info["steps"], "seconds": round(train_info["seconds"], 1),
                   "clips": len(clips), "held_out_n": len(held),
                   "lora_path": PERSONAL_VSR_LORA})
    return 0


def cmd_train_lm(_args):
    from .lm_train import train_personal_lm
    _print_report(train_personal_lm())
    return 0


def cmd_import_text(args):
    from .lm_train import import_text
    n = import_text(args.path)
    print(f"Imported {n} lines from {args.path} into phrases.txt")
    return 0


def cmd_import_wispr(_args):
    from .lm_train import import_wispr
    result = import_wispr()
    _print_report(result)
    print("Imported into phrases.txt — next: subvox-local train-lm")
    return 0


def cmd_file(args):
    """Lip-read a video file (no camera). For testing on a Mac."""
    from .face import load_tracker, mouth_rois
    from .vsr import load_reader
    from .dictation import Dictation
    import cv2
    tracker = load_tracker()
    reader = load_reader(beam_size=args.beam)
    cap = cv2.VideoCapture(args.path)
    frames, anchors, stamps = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        obs = tracker.detect(frame, int(t_s * 1000))
        frames.append(gray)
        anchors.append(obs.anchors if obs else None)
        stamps.append(t_s)
    cap.release()
    if not frames:
        print("No frames in video.")
        return 1
    # Resample to steady 25 fps first — the model was trained on 25 fps,
    # and a video file is usually 24/30/60 fps.
    from lipflow.vsr import LipReader
    idx = LipReader.resample(stamps, len(frames))
    frames = [frames[i] for i in idx]
    anchors = [anchors[i] for i in idx]
    rois = mouth_rois(frames, anchors)
    if rois is None:
        print("No face found in video.")
        return 1
    d = Dictation(reader, tracker, camera=None, cleaner_backend=args.cleanup)
    print(d.decode_rois(rois))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="subvox-local",
                                description="Silent lip-reading dictation, 100% on-device.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="Check models / platform (no inference)")
    s = sub.add_parser("setup-fast", help="Quick adapt, 6 sentences (Intel CPU: slow, see docs/INTEL_MAC.md)")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("adapt-more", help="Stage 2 background adapt (continues stage 1)")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--steps", type=int, default=150)
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("train-lm", help="Train personal LM on phrases.txt")
    s = sub.add_parser("import-text", help="Add a .txt file to phrases.txt")
    s.add_argument("path")
    sub.add_parser("import-wispr", help="Import Wispr Flow history (macOS, local read-only)")
    s = sub.add_parser("file", help="Lip-read a video file")
    s.add_argument("path")
    s.add_argument("--beam", type=int, default=10)
    s.add_argument("--cleanup", default="auto", choices=["auto", "local", "basic"])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    fn = {"doctor": cmd_doctor, "setup-fast": cmd_setup_fast,
          "adapt-more": cmd_adapt_more, "train-lm": cmd_train_lm,
          "import-text": cmd_import_text, "import-wispr": cmd_import_wispr,
          "file": cmd_file}[args.cmd]
    return fn(args)
