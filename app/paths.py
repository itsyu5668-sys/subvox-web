"""Paths and model inventory for the Subvox web backend.

Model weights are NOT copied into this repo. They live in
~/workspace/subvox-local/models/ (already downloaded) and are read
from there. Only personal artefacts (LoRA adapters, phrases, personal
LM, uploaded clips) live under this repo's data/ dir.

Override the local repo location with SUBVOX_LOCAL_ROOT.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parent.parent
DATA = WEB_ROOT / "data"
LORA_DIR = DATA / "lora"
PERSONAL_DIR = DATA / "personal"          # SUBVOX_LOCAL_HOME for subvox_local
TEST_CLIPS_DIR = DATA / "test_clips"
UPLOAD_DIR = DATA / "uploads"

SUBVOX_LOCAL_ROOT = Path(
    os.environ.get("SUBVOX_LOCAL_ROOT", "")
    or Path.home() / "workspace" / "subvox-local"
)
LOCAL_MODELS = SUBVOX_LOCAL_ROOT / "models"
LOCAL_SRC = SUBVOX_LOCAL_ROOT / "src"
LOCAL_UPSTREAM = SUBVOX_LOCAL_ROOT / "lipflow-upstream"

# Where the Mac app reads its personal LoRA from (subvox_local.paths on
# macOS: ~/Library/Application Support/SubvoxLocal/models/vsr_face_lora.pt).
# On this Linux VM the same paths.py logic resolves to
# ~/.local/share/SubvoxLocal/models/vsr_face_lora.pt. /train copies a kept
# adapter here as well as to data/lora/, so the file can be moved to the
# Mac verbatim. See README "LoRA hand-off to the Mac app".
if os.name == "posix" and Path("/System/Library").exists():
    MAC_PERSONAL_LORA = Path.home() / "Library/Application Support/SubvoxLocal/models/vsr_face_lora.pt"
else:
    MAC_PERSONAL_LORA = Path.home() / ".local/share/SubvoxLocal/models/vsr_face_lora.pt"

# (relative path under LOCAL_MODELS, source HF repo, description, required?)
CATALOG = [
    ("vsr/model.json", "Amanvir/LRS3_V_WER19.1", "Auto-AVSR visual-only config", True),
    ("vsr/model.pth", "Amanvir/LRS3_V_WER19.1", "Auto-AVSR visual-only weights (LRS3, 19.1% WER)", True),
    ("lm/model.json", "Amanvir/lm_en_subword", "Subword LM config", True),
    ("lm/model.pth", "Amanvir/lm_en_subword", "Subword RNN LM weights", True),
    ("lm/unigram5000.model", "mpc001/auto_avsr", "SentencePiece tokenizer", True),
    ("face_landmarker.task", "Google MediaPipe", "FaceLandmarker", True),
    ("qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf", "unsloth/Qwen3-0.6B-GGUF",
     "Qwen3-0.6B cleanup model (GGUF, CPU)", True),
    # NOT copied / NOT used by this server (documented so /doctor can say so):
    ("av/model.safetensors", "nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base",
     "AV (audio+visual) weights — NOT used, excluded on purpose", False),
    ("avhubert/model.pt", "vumichien/AV-HuBERT",
     "AV-HuBERT Large backup — NOT used, excluded on purpose", False),
]

LRS3_LICENSE_NOTE = (
    "LRS3 weights (Auto-AVSR visual-only model, Amanvir/LRS3_V_WER19.1) are "
    "NON-COMMERCIAL RESEARCH ONLY. Do not use this server commercially "
    "without replacing the VSR weights with a commercially licensed model."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(rel: str, with_hash: bool = False) -> dict:
    p = LOCAL_MODELS / rel
    entry: dict = {"path": str(p), "present": p.exists()}
    if p.exists():
        entry["size_bytes"] = p.stat().st_size
        entry["size_mb"] = round(p.stat().st_size / 1e6, 1)
        if with_hash:
            entry["sha256"] = sha256(p)
    else:
        entry["size_bytes"] = None
        entry["size_mb"] = None
    return entry


def inventory(with_hash: bool = False) -> dict:
    files = {}
    missing_required = []
    for rel, repo, desc, required in CATALOG:
        e = file_entry(rel, with_hash=with_hash)
        e.update({"source": repo, "description": desc, "required": required})
        files[rel] = e
        if required and not e["present"]:
            missing_required.append(rel)
    return {"files": files, "missing_required": missing_required,
            "all_required_present": not missing_required}


def disk_and_ram() -> dict:
    total, used, free = shutil.disk_usage(WEB_ROOT)
    out = {"disk_free_gb": round(free / 1e9, 2), "disk_total_gb": round(total / 1e9, 2),
           "cpu_count": os.cpu_count()}
    try:
        import resource
        out["ram_total_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 2)
    except Exception:
        out["ram_total_gb"] = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    out["ram_available_gb"] = round(int(line.split()[1]) / 1e6, 2)
                    break
    except Exception:
        pass
    return out
