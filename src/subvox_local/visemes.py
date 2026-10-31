"""Viseme mapping (lip-shape classes), adapted from Lipflow's lipflow/visemes.py
(MIT, Amy Zhou 2026) — which maps spelling to coarse viseme classes so words
that *look* the same on the lips can be compared.

Classes: M = p/b/m, F = f/v, T = t/d/n/s/z/x, K = k/g/c/q, J = ch/sh/j/y,
W = r/w, A = a/e, I = i/ee/ea, U = o/u/ou/oo, D = th, (h is silent on lips).
Used here to (a) verify the fast-setup sentences cover every class, and
(b) snap mis-read names to likely names by lip shape.
"""
from __future__ import annotations

import re

_DIGRAPHS = [("ch", "J"), ("sh", "J"), ("th", "D"), ("ph", "F"), ("ck", "K"),
             ("qu", "KW"), ("gu", "K"), ("mc", "MK"), ("wh", "W"), ("ng", "K"),
             ("ee", "I"), ("oo", "U"), ("ou", "U"), ("ea", "I")]
_SINGLE = {**dict.fromkeys("pbm", "M"), **dict.fromkeys("fv", "F"),
           **dict.fromkeys("tdnszlx", "T"), **dict.fromkeys("kgcq", "K"),
           **dict.fromkeys("jy", "J"), "r": "W", "w": "W",
           **dict.fromkeys("ae", "A"), "i": "I", **dict.fromkeys("ou", "U"), "h": ""}

ALL_CLASSES = set("MFTKJWA IUD".replace(" ", ""))


def visemes(word: str) -> str:
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]
    out, i = [], 0
    while i < len(w):
        for dg, v in _DIGRAPHS:
            if w.startswith(dg, i):
                out.append(v)
                i += len(dg)
                break
        else:
            out.append(_SINGLE.get(w[i], ""))
            i += 1
    s = "".join(out)
    return re.sub(r"(.)\1+", r"\1", s)


def sentence_visemes(sentence: str) -> set[str]:
    out: set[str] = set()
    for w in sentence.split():
        out.update(visemes(w))
    return out


def coverage(sentences: list[str]) -> dict:
    """Return {covered classes, missing classes, per-sentence classes}."""
    covered: set[str] = set()
    per = []
    for s in sentences:
        c = sentence_visemes(s)
        per.append(c)
        covered |= c
    return {"covered": sorted(covered), "missing": sorted(ALL_CLASSES - covered), "per_sentence": per}
