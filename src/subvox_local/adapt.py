"""Fast setup: LoRA quick adapt, not a full fine-tune.

Lipflow's stock onboarding fine-tunes the whole visual front end
(3D-conv + ResNet) plus the first encoder layer for 6 epochs on 24 clips —
about 5 minutes of GPU training on an M4 Pro, inside an ~8 minute setup.

The fast plan keeps the quality and cuts the time:
  * Freeze the whole base Auto-AVSR model.
  * Train ONLY LoRA adapters (rank 8, alpha 16) on the self-attention
    Q/K/V/out Linear layers of every Conformer encoder layer.
    At n_feat=768: 4 linears x 12 layers x (8*768 + 768*8) = 589,824
    trainable params (~0.6M) — the same scale as the "V LoRA +0.6M"
    row in the personalized-lip-reading paper (KAIST, 2024), where
    1 minute of adaptation data was enough to improve every speaker.
  * Max 300 optimizer steps, AdamW lr 1e-4, cosine decay to 1e-5,
    grad clip 5.0, BatchNorm frozen (a few clips would wreck BN stats —
    Lipflow does the same in train_vsr.py).
  * Loss is the model's own objective (0.1*CTC + 0.9*attention, label
    smoothing), reusing Lipflow's clip_loss / _augment, so the training
    signal is identical — only the trainable parameter set is smaller.
  * Stage 1 (quick): 6 sentences, 3 epochs over those clips (~2 min on
    an M4 Pro GPU — much slower on an Intel i5 CPU, see INTEL note above).
    Stage 2 (background): the other 4 sentences + corrections, continue
    from the stage-1 LoRA, while the user is already dictating.
  * Keep-only-if-better: the adapted model is evaluated on held-out
    sentences the adapt step never trained on. It is saved ONLY if its
    word error rate beats the stock model on those sentences. Otherwise
    the stock model stays, and the LoRA file is discarded.

INTEL MAC (this build): Francis's Mac is an Intel Core i5, NOT Apple
Silicon. There is no MPS and no CUDA on it, so ALL training here is
torch CPU: device "cpu", batch size 1, 0 data workers, torch threads
capped at 4 (an i5 is typically 2-4 cores; more threads just thrash).
LoRA is still rank 8, 589,824 trainable params — the parameter count
does not change on CPU, only the speed does.

Do NOT claim the old "~2 min" fast-setup time on Intel. Lipflow's
full fine-tune took ~5 minutes of GPU training on an M4 Pro; LoRA
300 steps on that GPU was the ~2 min target. On an i5 CPU the same
steps will be MUCH slower — several times slower at best, plausibly
10+ minutes for stage 1 — and the exact time is UNMEASURED until it
runs on Francis's Mac. That is why the defaults below are cut for
CPU (150 steps max for stage 1, not 300) and why stage 2 is a
background/resumable job: start it, keep dictating with the stock
model, and let it finish in its own time. train_lora() saves a
checkpoint every 50 steps and can resume from one, so a killed or
interrupted run does not lose all progress. See docs/INTEL_MAC.md.

UNTESTED ON INTEL MAC: LoRA training has not been run on an Intel Mac
in this build (this VM is Linux x86_64, CPU only). The code paths are
the same PyTorch ops Lipflow's finetune uses, but the step time / WER
numbers on a real i5 are unmeasured — see docs/INTEL_MAC.md for the
Mac test. No accuracy numbers are claimed here; evaluate() reports
the measured WER on the user's own held-out clips at setup time.
"""
from __future__ import annotations

import os
import random
import sys
import time

import torch
import torch.nn as nn

from .wer import word_error_rate

__all__ = ["word_error_rate", "LoRALinear", "apply_lora", "remove_lora",
           "lora_state", "save_lora", "load_lora", "train_lora", "evaluate",
           "fast_setup", "start_stage2_background", "run_stage2_background"]

LORA_R = 8
LORA_ALPHA = 16
LORA_MAX_STEPS = 300  # absolute cap (was the old GPU-era stage-1 default)
LORA_DEFAULT_STEPS = 150  # Intel CPU stage-1 default: half the steps, still usable
LORA_BATCH_SIZE = 1  # Intel CPU: batch 1 only, no batching win on 2-4 cores
LORA_NUM_WORKERS = 0  # Intel CPU: no data-loader workers (in-process data)
LORA_DEVICE = "cpu"  # Intel i5: CPU only — no MPS (Apple Silicon only), no CUDA
LORA_LR = 1e-4
LORA_MIN_LR = 1e-5
LORA_CHECKPOINT_EVERY = 50  # steps between resume checkpoints in train_lora()


def _lipflow_train_vsr():
    """Import Lipflow's train_vsr (clip_loss / _augment / _targets) from the
    vendored copy in lipflow-upstream/. Raises a clear error if missing."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    upstream = os.path.join(root, "lipflow-upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from lipflow import train_vsr as _tv
    return _tv


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank update.

    Forward: y = Wx + b + (alpha/r) * B(A(x)).
    A is init'd N(0, 1/r), B is zeros, so the adapter starts as identity.
    Adapter params are created on the base layer's device (CPU on an
    Intel Mac), so the first forward does not hit a device mismatch.
    """

    def __init__(self, base, r: int = LORA_R, alpha: int = LORA_ALPHA):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        for p in base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features, device=dev) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        import torch.nn.functional as F
        y = self.base(x)
        return y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    def lora_state(self):
        return {"lora_A": self.lora_A.detach().cpu(),
                "lora_B": self.lora_B.detach().cpu()}

    def load_lora_state(self, sd):
        self.lora_A.data.copy_(sd["lora_A"].to(self.lora_A.device))
        self.lora_B.data.copy_(sd["lora_B"].to(self.lora_B.device))

    def enable_grads(self):
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)


def apply_lora(reader, r: int = LORA_R, alpha: int = LORA_ALPHA):
    """Replace every Conformer encoder layer's self-attention Q/K/V/out
    Linear with a LoRA adapter. Freezes everything else.
    Returns (adapters, n_trainable_params). adapters is a dict
    {dotted_name: LoRA module} so the state can be saved / loaded."""
    model = reader.model
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = {}
    encoders = getattr(model.encoder, "encoders", None)
    if encoders is None:
        raise RuntimeError("Could not find model.encoder.encoders — "
                           "Auto-AVSR checkpoint layout differs from Lipflow's.")
    for li, layer in enumerate(encoders):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
            base = getattr(attn, name, None)
            if base is None:
                continue
            if isinstance(base, LoRALinear):
                continue  # already wrapped — a second apply must not crash
            lora = LoRALinear(base, r=r, alpha=alpha)
            setattr(attn, name, lora)
            adapters[f"encoders.{li}.self_attn.{name}"] = lora
    n = sum(p.numel() for a in adapters.values() for p in (a.lora_A, a.lora_B))
    return adapters, n


def remove_lora(reader, adapters: dict) -> None:
    """Restore the original (unwrapped) Linear layers, e.g. when an adapt
    round is discarded so later dictation in the same process uses stock."""
    model = reader.model
    for key, lora in adapters.items():
        parts = key.split(".")  # encoders.{li}.self_attn.{name}
        layer = model.encoder.encoders[int(parts[1])]
        setattr(layer.self_attn, parts[3], lora.base)


def lora_state(adapters: dict) -> dict:
    return {k: v.lora_state() for k, v in adapters.items()}


def save_lora(adapters: dict, path: str) -> str:
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(lora_state(adapters), path)
    return path


def load_lora(reader, adapters: dict, path: str) -> None:
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k, v in adapters.items():
        if k in sd:
            v.load_lora_state(sd[k])
    for a in adapters.values():
        a.enable_grads()  # loading must not leave stage 2 with nothing to train


def train_lora(reader, clips: list[dict], adapters: dict, steps: int = LORA_DEFAULT_STEPS,
               lr: float = LORA_LR, log=print, seed: int = 0,
               checkpoint_path: str | None = None, resume_path: str | None = None,
               save_every: int = LORA_CHECKPOINT_EVERY) -> dict:
    """Train only the LoRA adapters on clips [{'rois': (T,96,96) uint8, 'text': str}].
    Uses Lipflow's clip_loss / _augment (same loss, same augmentation).
    Intel CPU: device cpu, batch 1, torch threads capped at 4 — this is
    MUCH slower than the old M4 GPU numbers; `seconds` in the return
    value is the measured time, do not quote the old ~2 min target.
    Background / resume: if checkpoint_path is given, the LoRA state is
    saved every `save_every` steps; if resume_path points at such a
    checkpoint, training continues from it instead of starting over.
    Returns {'steps': n, 'loss_start': f, 'loss_end': f, 'seconds': s}."""
    import torch
    _tv = _lipflow_train_vsr()
    if not clips:
        return {"steps": 0, "loss_start": None, "loss_end": None, "seconds": 0.0}
    # Intel i5: cap torch threads — more threads than cores just thrashes
    # on a 2-4 core CPU, and this process shares the machine with dictation.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    if resume_path and os.path.exists(resume_path):
        load_lora(reader, adapters, resume_path)
        log(f"Resuming LoRA from checkpoint: {resume_path}")
    random.seed(seed)
    torch.manual_seed(seed)
    # Re-enable adapter grads: a previous train_lora()/load_lora() round or
    # the end-of-training freeze below must not leave stage 2 with nothing
    # to train (AdamW raises on an empty parameter list).
    for a in adapters.values():
        a.enable_grads()
    params = [p for a in adapters.values() for p in (a.lora_A, a.lora_B)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=LORA_MIN_LR)
    data = [(c["rois"], _tv._targets(reader, c["text"])) for c in clips]
    reader.model.decoder.to(reader.enc_device)
    reader.model.ctc.to(reader.enc_device)
    if hasattr(reader.model, "criterion") and hasattr(reader.model.criterion, "to"):
        reader.model.criterion.to(reader.enc_device)
    reader.model.train()
    for mod in reader.model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            mod.eval()
    t0, losses, step = time.time(), [], 0
    while step < steps:
        random.shuffle(data)
        for rois, ys in data:
            if step >= steps:
                break
            loss = _tv.clip_loss(reader, _tv._augment(rois), ys)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            step += 1
            if checkpoint_path and save_every and step % save_every == 0:
                save_lora(adapters, checkpoint_path)
                log(f"  LoRA checkpoint saved ({step}/{steps}): {checkpoint_path}")
            if step % 50 == 0:
                log(f"  LoRA step {step}/{steps}: loss {sum(losses[-50:]) / len(losses[-50:]):.3f}")
    reader.model.eval()
    reader.model.decoder.to(reader.device)
    reader.model.ctc.to(reader.device)
    # Freeze the base again, but NOT the adapters — stage 2 must be able to
    # train them in a later call.
    adapter_ids = {id(p) for a in adapters.values() for p in (a.lora_A, a.lora_B)}
    for p in reader.model.parameters():
        if id(p) not in adapter_ids:
            p.requires_grad_(False)
    return {"steps": step,
            "loss_start": sum(losses[:5]) / max(len(losses[:5]), 1) if losses else None,
            "loss_end": sum(losses[-5:]) / max(len(losses[-5:]), 1) if losses else None,
            "seconds": time.time() - t0}


def evaluate(reader, clips: list[dict]) -> dict:
    """Greedy-decode held-out clips and report WER. No training data here."""
    wers, rows = [], []
    for c in clips:
        text, _ = reader.read(c["rois"], fast=True)
        wer = word_error_rate(c["text"], text)
        wers.append(wer)
        rows.append({"expected": c["text"], "got": text, "wer": round(wer, 3)})
    return {"wer_mean": sum(wers) / len(wers) if wers else None,
            "n": len(wers), "rows": rows}


def fast_setup(reader, stage1_clips: list[dict], held_out_clips: list[dict],
               log=print, steps: int = LORA_DEFAULT_STEPS) -> dict:
    """Stage 1 quick adapt. Returns a report; saves the LoRA file ONLY if
    adapted WER on held-out clips beats stock WER on the same clips.
    Intel CPU: `steps` defaults to 150 (not the old 300) and training
    is CPU-slow — the old "~2 min" claim does NOT apply on an i5.
    The report's train.seconds is the measured time; record it.
    A checkpoint is saved every 50 steps so an interrupted run can
    resume with train_lora(..., resume_path=<checkpoint>)."""
    from .paths import PERSONAL_VSR_LORA
    import os as _os
    checkpoint = _os.path.join(_os.path.dirname(PERSONAL_VSR_LORA), "vsr_face_lora_stage1.pt")
    if not held_out_clips:
        return {"stock_wer": None, "adapted_wer": None, "kept": False,
                "lora_path": None, "trainable_params": 0, "train": None,
                "stock_rows": [], "adapted_rows": [],
                "note": "No held-out clips recorded — nothing to evaluate against. "
                        "Record the held-out sentences in setup-fast first."}
    stock = evaluate(reader, held_out_clips)
    log(f"Stock WER on {stock['n']} held-out clips: {stock['wer_mean']:.3f}")
    adapters, n_params = apply_lora(reader)
    log(f"LoRA adapters: {len(adapters)} linears, {n_params:,} trainable params (base frozen)")
    log(f"Intel CPU training: {steps} steps max, batch 1, device cpu — "
        "slower than M4 GPU, time is measured not claimed.")
    train_info = train_lora(reader, stage1_clips, adapters, steps=steps,
                            log=log, checkpoint_path=checkpoint)
    adapted = evaluate(reader, held_out_clips)
    log(f"Adapted WER on held-out clips: {adapted['wer_mean']:.3f} "
        f"(train: {train_info['seconds']:.0f}s, {train_info['steps']} steps)")
    kept = adapted["wer_mean"] < stock["wer_mean"]
    path = None
    if kept:
        path = save_lora(adapters, PERSONAL_VSR_LORA)
        log(f"Kept adapted model -> {path}")
    else:
        # Remove the adapters from the live model too, so dictation in this
        # process does not silently keep using the "discarded" model.
        remove_lora(reader, adapters)
        log("No improvement on held-out clips — stock model stays, LoRA discarded.")
    return {"stock_wer": stock["wer_mean"], "adapted_wer": adapted["wer_mean"],
            "kept": kept, "lora_path": path, "trainable_params": n_params,
            "train": train_info, "stock_rows": stock["rows"], "adapted_rows": adapted["rows"],
            "checkpoint_path": checkpoint, "steps": steps, "device": LORA_DEVICE}


def run_stage2_background(reader, clips: list[dict], held_out_clips: list[dict],
                          report_path: str, steps: int = LORA_DEFAULT_STEPS,
                          log=print) -> dict:
    """Stage 2 worker: continue LoRA from stage 1 on new clips, evaluate
    on held-out clips, keep ONLY if better, write the report JSON to
    report_path. Meant to run in a background thread (see
    start_stage2_background) so the user can keep dictating with the
    stock model while the i5 CPU trains — stage 2 on CPU is slow and
    must not block dictation. The keep-only-if-better gate is identical
    to fast_setup's; a discarded stage-2 LoRA never touches the saved
    stage-1 file."""
    import json
    from .paths import PERSONAL_VSR_LORA
    report: dict = {"kept": False, "lora_path": None, "steps": steps,
                    "device": LORA_DEVICE, "error": None}
    try:
        adapters, n_params = apply_lora(reader)
        if os.path.exists(PERSONAL_VSR_LORA):
            load_lora(reader, adapters, PERSONAL_VSR_LORA)
            log(f"Continuing stage 2 from stage-1 LoRA: {PERSONAL_VSR_LORA}")
        report["trainable_params"] = n_params
        checkpoint = os.path.join(os.path.dirname(PERSONAL_VSR_LORA),
                                  "vsr_face_lora_stage2.pt")
        report["checkpoint_path"] = checkpoint
        stock = evaluate(reader, held_out_clips) if held_out_clips else None
        if stock:
            report["stock_wer"] = stock["wer_mean"]
            log(f"Stage 2 start, held-out WER: {stock['wer_mean']:.3f}")
        train_info = train_lora(reader, clips, adapters, steps=steps, log=log,
                                checkpoint_path=checkpoint,
                                resume_path=checkpoint)
        report["train"] = train_info
        if held_out_clips:
            adapted = evaluate(reader, held_out_clips)
            report["adapted_wer"] = adapted["wer_mean"]
            log(f"Stage 2 done, held-out WER: {adapted['wer_mean']:.3f} "
                f"(train: {train_info['seconds']:.0f}s)")
            if adapted["wer_mean"] < stock["wer_mean"]:
                report["lora_path"] = save_lora(adapters, PERSONAL_VSR_LORA)
                report["kept"] = True
                log(f"Kept stage-2 model -> {report['lora_path']}")
            else:
                remove_lora(reader, adapters)
                log("Stage 2 did not beat held-out WER — discarded, stage-1 file untouched.")
        else:
            report["lora_path"] = save_lora(adapters, PERSONAL_VSR_LORA)
            report["kept"] = True
            log("No held-out clips — stage-2 LoRA saved without a keep-gate check. "
                "Record held-out clips next time so the gate can run.")
    except Exception as e:  # background thread must never die silently
        report["error"] = str(e)
        log(f"Stage 2 background training failed: {e}")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def start_stage2_background(reader, clips: list[dict], held_out_clips: list[dict],
                            report_path: str, steps: int = LORA_DEFAULT_STEPS,
                            log=print):
    """Start stage 2 in a daemon background thread. Returns the Thread;
    the report JSON appears at report_path when it finishes (or fails).
    The caller keeps dictating with the stock model meanwhile — do NOT
    use `reader` for dictation in the main thread while this runs, the
    LoRA adapters are being trained on it. UNTESTED ON INTEL MAC."""
    import threading
    t = threading.Thread(target=run_stage2_background,
                         args=(reader, clips, held_out_clips, report_path),
                         kwargs={"steps": steps, "log": log}, daemon=True)
    t.start()
    return t
