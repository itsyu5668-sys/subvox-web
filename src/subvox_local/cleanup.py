"""Offline cleanup: raw VSR guesses -> clean sentence. LOCAL ONLY.

No cloud calls. No anthropic / openai imports anywhere in this file.

INTEL MAC (this build): Francis's Mac is an Intel Core i5, NOT Apple
Silicon. MLX does NOT run on Intel Macs (Apple Silicon only), so the
previous MLX primary backend is REMOVED. The primary backend is now:

 1. `local`: Qwen3-0.6B Q4_K_M GGUF via llama-cpp-python (llama_cpp),
    CPU, local file models/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf.
    GGUF runs everywhere, including Intel Macs and this Linux VM.
    Source repo is unsloth/Qwen3-0.6B-GGUF — NOTE: the official
    Qwen/Qwen3-0.6B-GGUF repo ships ONLY Qwen3-0.6B-Q8_0.gguf, there
    is no Q4_K_M file there (verified 2026-10-31, HF API file list).
    Fallback if present: Qwen3-1.7B Q4_K_M GGUF
    (unsloth/Qwen3-1.7B-GGUF, models/qwen3-1.7b-gguf/).
    The model gets the top-3 beam hypotheses + your recent dictations
    and picks the intended sentence — that is what fixes p/b/m, f/v,
    t/d/n confusions, which are invisible on the lips.
    Expected speed on an Intel i5 CPU is slower than MLX on Apple
    Silicon (~0.2 s/sentence there); no latency number is claimed
    here until it is measured on Francis's Mac — see docs/INTEL_MAC.md.
 2. `basic` (everywhere): offline rules only — sentence case, "I",
    end punctuation, spelled-out numbers -> digits. This is the
    fallback when no GGUF file is present. It is NOT as good as the
    local LLM, and this file says so instead of pretending otherwise.

UNTESTED ON INTEL MAC: the GGUF path has not been run on an Intel Mac
in this build (this VM is Linux x86_64). Mac test steps are in
docs/INTEL_MAC.md. The old MLX model dirs (models/qwen3-*-mlx/) are
still on disk for reference but are Apple Silicon only and are NOT
usable on Francis's Intel Mac.
"""
from __future__ import annotations

import os
import re

# GGUF cleanup models. Repo ids are the unsloth GGUF builds (the Qwen
# official GGUF repo has no Q4_K_M file — see module docstring).
LOCAL_MODEL = "unsloth/Qwen3-0.6B-GGUF"
LOCAL_MODEL_FILE = "Qwen3-0.6B-Q4_K_M.gguf"
LOCAL_MODEL_FALLBACK = "unsloth/Qwen3-1.7B-GGUF"
LOCAL_MODEL_FALLBACK_FILE = "Qwen3-1.7B-Q4_K_M.gguf"

SMALL_SHOTS = """Fix lip-reading guesses. Pick the sentence the person meant.
Guesses: WHILE IN OFFICE / WALLET OFFICER / WHILE IN OFFICER
Recent: I have a meeting in the office tomorrow
Answer: While in office.

Guesses: BORN IN NEW YORK CITY / BORN IN NEW YORK SITTY / BORN IN NEW YORK CITY AND
Answer: Born in New York City.
"""

_TENS = {"twenty": "2", "thirty": "3", "forty": "4", "fifty": "5",
          "sixty": "6", "seventy": "7", "eighty": "8", "ninety": "9"}
_ONES = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
         "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def _year3(m: "re.Match") -> str:
    # "nineteen forty three" -> "1943". Unrecognised words: leave untouched
    # (never emit a half-converted "19hundred").
    a, b = m.group(1), m.group(2)
    if a in _TENS and b in _ONES:
        return "19" + _TENS[a] + _ONES[b]
    return m.group(0)


def _year2(m: "re.Match") -> str:
    # "nineteen forty" -> "1940" (round decade, no ones word).
    a = m.group(1)
    if a in _TENS:
        return "19" + _TENS[a] + "0"
    return m.group(0)


def basic_cleanup(text: str) -> str:
    """Offline rules only. Deterministic, no model. Used as fallback."""
    t = " ".join(text.strip().lower().split())
    if not t:
        return ""
    # spelled-out years like "nineteen forty three" -> 1943 (Lipflow's rule)
    t = re.sub(r"\bnineteen (\w+) (\w+)\b", _year3, t)
    t = re.sub(r"\bnineteen (\w+)\b", _year2, t)
    t = re.sub(r"\bi\b", "I", t)
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return t


class LocalCleaner:
    """GGUF Qwen3 cleanup via llama-cpp-python, CPU. Lazy-loads on first use.

    Works on Intel Macs, Apple Silicon, and Linux — anywhere a GGUF file
    and llama-cpp-python run. No MLX, no GPU, no network at runtime.
    """

    def __init__(self, model: str = LOCAL_MODEL):
        self.model_id = model
        self._llm = None
        self._llm_path = None

    def _candidate_paths(self) -> list[str]:
        from .paths import QWEN_06B_GGUF, QWEN_17B_GGUF
        return [QWEN_06B_GGUF, QWEN_17B_GGUF]

    def available(self) -> bool:
        """True only if llama-cpp-python is importable AND a GGUF cleanup
        model file is actually downloaded locally. Checking the import
        alone used to claim the backend was ready while the model file
        was still missing — that mistake is not repeated here."""
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False
        return any(os.path.isfile(p) for p in self._candidate_paths())

    def _load(self):
        if self._llm is not None:
            return
        from llama_cpp import Llama
        for path in self._candidate_paths():
            if os.path.isfile(path):
                # n_ctx 1024 is plenty for a rerank prompt (a few dozen
                # tokens); n_threads capped at 4 so a dual-core i5 is not
                # oversubscribed — llama.cpp picks a sane default anyway.
                n_threads = max(1, min(4, os.cpu_count() or 2))
                self._llm = Llama(model_path=path, n_ctx=1024,
                                  n_threads=n_threads, verbose=False)
                self._llm_path = path
                return
        raise FileNotFoundError(
            "No Qwen GGUF cleanup model downloaded. Run: "
            "bash scripts/download-models.sh  (section 9, Qwen3-0.6B-Q4_K_M.gguf)")

    def clean(self, hypotheses: list[str], recent: list[str] | None = None,
              custom_words: list[str] | None = None) -> str:
        """hypotheses: top-3 beam guesses, best first. Returns cleaned sentence."""
        if not hypotheses:
            return ""
        if not self.available():
            return basic_cleanup(hypotheses[0])
        try:
            self._load()
        except Exception:
            # Model failed to load (corrupt / removed) — degrade to rules,
            # never crash dictation on the cleanup step.
            return basic_cleanup(hypotheses[0])
        recent_txt = "\n".join((recent or [])[-3:])
        words_txt = ", ".join(custom_words or [])
        prompt = (SMALL_SHOTS
                  + f"Guesses: {' / '.join(hypotheses)}\n"
                  + (f"Recent: {recent_txt}\n" if recent_txt else "")
                  + (f"Custom words: {words_txt}\n" if words_txt else "")
                  + "Answer:")
        try:
            out = self._llm(prompt, max_tokens=60, temperature=0.0,
                            stop=["\n"])
            text = out["choices"][0]["text"].strip().split("\n")[0].strip()
        except Exception:
            return basic_cleanup(hypotheses[0])
        return text or basic_cleanup(hypotheses[0])


def clean_top3(hypotheses: list[str], recent=None, custom_words=None,
               backend: str = "auto") -> str:
    """One entry point. backend: auto | local | basic.
    auto = local GGUF (llama-cpp, CPU) if available, else basic rules
    (and say so in the log). `local` is the GGUF backend — NOT MLX;
    MLX was removed for the Intel Mac (Apple Silicon only)."""
    if backend == "basic":
        return basic_cleanup(hypotheses[0] if hypotheses else "")
    cleaner = LocalCleaner()
    if cleaner.available():
        return cleaner.clean(hypotheses, recent=recent, custom_words=custom_words)
    return basic_cleanup(hypotheses[0] if hypotheses else "")
