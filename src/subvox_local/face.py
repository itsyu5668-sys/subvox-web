"""Face tracking + mouth crops. Thin wrapper over Lipflow's face.py.

Pipeline (Lipflow / Auto-AVSR, Apache 2.0): MediaPipe FaceLandmarker runs
on every frame while recording -> 4 anchors (right eye, left eye, nose
base, mouth centre) are aligned to the training mean face -> 96x96
grayscale mouth crops, resampled to 25 fps, the exact input Auto-AVSR
was trained on.

This module does not reimplement that math; it re-exports Lipflow's
FaceTracker / mouth_rois so Subvox Local's capture code has one import.
"""
from __future__ import annotations

import os
import sys


def _add_upstream():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    upstream = os.path.join(root, "lipflow-upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    return upstream


def load_tracker(model_path: str | None = None):
    from .paths import FACE_LANDMARKER
    _add_upstream()
    from lipflow.face import FaceTracker
    path = model_path or FACE_LANDMARKER
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing MediaPipe model: {path}\n"
                                "Run: bash scripts/download-models.sh")
    return FaceTracker(model_path=path)


def mouth_rois(*args, **kwargs):
    _add_upstream()
    from lipflow.face import mouth_rois as _mr
    return _mr(*args, **kwargs)
