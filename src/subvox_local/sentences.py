"""Fast-setup sentences: 10 phonetically balanced Harvard sentences.

Lipflow uses 24 sentences (12 of your own + 12 Harvard, IEEE 1969 lists 1-6).
That is where its ~8 minutes go: 24 takes + ~5 min of full-frontend training.

The fast plan uses 10 Harvard sentences instead of 24, chosen so that
together they cover ALL 10 viseme classes (M, F, T, K, J, W, A, I, U, D).
Verified with src/subvox_local/visemes.py — see tests/test_sentences.py.
Each sentence is 7-9 words, 5-12 words is Lipflow's own range for practice.

Why these 10 (what each one adds):
 1. birch canoe / smooth planks — M (b, p, m), K (c), T (s, t), W (r)
 2. glue sheet / dark blue background — J (sh), K (g, k), W (l->W? no, w in blue), U (oo, u)
 3. easy tell depth well — I (ea, y), T (t, d), D (th in "the depth"), F (f in "of" is silent-ish, but "depth" th)
 4. chicken leg rare dish — J (ch, sh in dish), K (c, k), T (t, d), I (ea)
 5. rice served round bowls — W (r, w), T (s, d), U (ou, ow), M (m in "round"? no — bowls b)
 6. juice lemons fine punch — J (j), F (f), M (m), T (s, n), U (u)
 7. box thrown parked truck — T (th in thrown, t, d), K (k, x->K), M (m in "thrown"? no — b in box)
 8. four hours steady work — F (f), W (w, r, h silent), T (s, t), U (ou, ow)
 9. kick ball straight follow — K (k, c), T (t, s), F (f), W (ll->W? "follow" f + w sound)
10. salt breeze across sea — T (s, t), K (c, k), F (f? "across" — no, breeze z->T), I (ee), A (a)

Quick-adapt stage 1 uses sentences 1, 3, 5, 6, 9, 10 (6 takes, ~2 min
of mouthing — the takes, not the training; training on an Intel i5 CPU
is much slower than the old M4 GPU target, see docs/INTEL_MAC.md).
Those 6 alone also cover all 10 classes (verified in tests) — that is what
makes the 2-minute stage usable instead of a toy. Stage 2 adds 2, 4, 7, 8
in the background while you use the app.
"""
from __future__ import annotations

FAST_10 = [
    "The birch canoe slid on the smooth planks",
    "Glue the sheet to the dark blue background",
    "It's easy to tell the depth of a well",
    "These days a chicken leg is a rare dish",
    "Rice is often served in round bowls",
    "The juice of lemons makes fine punch",
    "The box was thrown beside the parked truck",
    "Four hours of steady work faced us",
    "Kick the ball straight and follow through",
    "The salt breeze came across from the sea",
]

# Stage 1 (quick adapt, ~2 min): indices into FAST_10
QUICK_6_IDX = [0, 2, 4, 5, 8, 9]
QUICK_6 = [FAST_10[i] for i in QUICK_6_IDX]
# Stage 2 (background, adds the rest): indices into FAST_10
STAGE2_IDX = [1, 3, 6, 7]
STAGE2 = [FAST_10[i] for i in STAGE2_IDX]

# Held-out check: 2 sentences from FAST_10 are NEVER trained on in stage 1.
# They are used only to decide "keep adapted model vs keep stock".
# We use indices 3 and 7 as held-out in stage 1; stage 2 trains on them only
# after the stage-1 keep/drop decision has been made.
HELD_OUT_STAGE1_IDX = [3, 7]
HELD_OUT_STAGE1 = [FAST_10[i] for i in HELD_OUT_STAGE1_IDX]
