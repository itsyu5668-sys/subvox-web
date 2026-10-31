"""Where Subvox Local keeps things.

Shared model weights live in <repo>/models/ (downloaded once, never uploaded).
Everything learned from *you* (clips, phrases, personal models, settings) lives
in the user data dir — macOS: ~/Library/Application Support/SubvoxLocal,
Linux: $XDG_DATA_HOME/SubvoxLocal (~/.local/share/SubvoxLocal).
Override with SUBVOX_LOCAL_HOME (tests use a temp dir).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS = os.path.join(ROOT, "models")
VSR_DIR = os.path.join(MODELS, "vsr")
LM_DIR = os.path.join(MODELS, "lm")
AV_DIR = os.path.join(MODELS, "av")
FACE_LANDMARKER = os.path.join(MODELS, "face_landmarker.task")
QWEN_06B = os.path.join(MODELS, "qwen3-0.6b-4bit-mlx")  # Apple Silicon only — NOT usable on Intel Mac
QWEN_17B = os.path.join(MODELS, "qwen3-1.7b-4bit-mlx")  # Apple Silicon only — NOT usable on Intel Mac
# Intel Mac (and Linux) cleanup models: GGUF via llama-cpp-python, CPU.
QWEN_06B_GGUF = os.path.join(MODELS, "qwen3-0.6b-gguf", "Qwen3-0.6B-Q4_K_M.gguf")
QWEN_17B_GGUF = os.path.join(MODELS, "qwen3-1.7b-gguf", "Qwen3-1.7B-Q4_K_M.gguf")

if sys.platform == "win32":
    _DEFAULT_HOME = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "SubvoxLocal")
elif sys.platform == "linux":
    _DEFAULT_HOME = os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")), "SubvoxLocal")
else:  # darwin
    _DEFAULT_HOME = os.path.expanduser("~/Library/Application Support/SubvoxLocal")

HOME = os.environ.get("SUBVOX_LOCAL_HOME") or _DEFAULT_HOME
PERSONAL_MODELS = os.path.join(HOME, "models")
PERSONAL_VSR_LORA = os.path.join(PERSONAL_MODELS, "vsr_face_lora.pt")
PERSONAL_VSR = os.path.join(PERSONAL_MODELS, "vsr_face.pth")  # Lipflow-compat full-frontend file
PERSONAL_LM = os.path.join(PERSONAL_MODELS, "lm_phrasing.pth")
CLIPS_DIR = os.path.join(HOME, "clips", "fast_setup")
PHRASES = os.path.join(HOME, "phrases.txt")
WORDS = os.path.join(HOME, "words.txt")
CORRECTIONS = os.path.join(HOME, "corrections.jsonl")
SETTINGS = os.path.join(HOME, "settings.json")
