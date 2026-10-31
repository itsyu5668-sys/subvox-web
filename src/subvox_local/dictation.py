"""Push-to-talk dictation loop: hold key -> record -> decode -> cleanup -> paste.

Flow (Lipflow's, kept):
  hold Right Option (macOS) -> webcam + live FaceLandmarker -> mouth crops
  at 25 fps -> on release, keep filming 0.4 s (trailing frames) -> Auto-AVSR
  beam search, top-3 -> offline Qwen cleanup rerank (or basic rules) ->
  paste at cursor in whatever app has focus.
Live preview: greedy CTC every 0.45 s while you talk (fast, ~1 ms).

This module is the orchestrator; it does not contain model code.
UNTESTED ON INTEL MAC: hotkey + paste + GGUF cleanup paths need a real Mac.
"""
from __future__ import annotations

from .capture import TAIL_SECONDS


class Dictation:
    def __init__(self, reader, tracker, camera, cleaner_backend: str = "auto",
                 beam_size: int = 10, copy_only: bool = False):
        self.reader = reader
        self.tracker = tracker
        self.camera = camera
        self.cleaner_backend = cleaner_backend
        self.beam_size = beam_size
        self.copy_only = copy_only
        self.recent: list[str] = []
        self.custom_words: list[str] = []

    def decode_rois(self, rois) -> str:
        """ROIs (T,96,96) -> cleaned sentence."""
        from .cleanup import clean_top3
        from .vsr import read_top3
        hyps = read_top3(self.reader, rois)  # beam size was fixed at load time
        text = clean_top3(hyps, recent=self.recent, custom_words=self.custom_words,
                          backend=self.cleaner_backend)
        if text:
            self.recent.append(text)
            self.recent = self.recent[-5:]
        return text

    def dictate_once(self, seconds: float = 10.0) -> str:
        """Record up to `seconds` (or until the caller stops it), decode, paste.
        Used by the CLI `file` command and by the menu-bar app.
        The 0.4 s of trailing frames is part of the record window itself —
        sleeping after record() returns would only add latency, no frames."""
        from .paste import copy_only as _copy, paste as _paste
        rois, _stamps = self.camera.record(seconds + TAIL_SECONDS, self.tracker)
        if rois is None or len(rois) == 0:
            return ""
        text = self.decode_rois(rois)
        if not text:
            return ""
        if self.copy_only:
            _copy(text)
        else:
            _paste(text)
        return text
