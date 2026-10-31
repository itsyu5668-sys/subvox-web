"""Webcam capture: frames + timestamps -> mouth ROIs at 25 fps.

OpenCV VideoCapture (camera index 0 by default). Frames are kept as
grayscale + MediaPipe anchors; mouth_rois() aligns and crops them.
No video is saved unless the user is in a practice/adapt round, in
which case ROIs (96x96 uint8, no full face, no audio) are saved as
compressed .npz under the user data dir — same as Lipflow's practice clips.
Nothing leaves the machine.
"""
from __future__ import annotations

import time

import numpy as np

MODEL_FPS = 25
TAIL_SECONDS = 0.4  # keep filming 0.4 s after key release (Lipflow: model needs trailing frames)


class Capture:
    def __init__(self, camera: int = 0, width: int = 1280, height: int = 720):
        self.camera = camera
        self.width = width
        self.height = height
        self._cap = None

    def open(self):
        import cv2
        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera}. "
                               "On macOS, grant Camera permission to the app, not the terminal.")
        self._cap = cap
        return self

    def read_frame(self):
        ok, frame = self._cap.read()
        if not ok:
            return None, None
        return frame, time.time()

    def record(self, seconds: float, tracker, on_frame=None):
        """Record for `seconds`, running FaceLandmarker live per frame.
        Returns (rois, stamps): mouth crops resampled to a steady 25 fps
        (the rate Auto-AVSR was trained on) + the resampled timestamps."""
        from .face import _add_upstream, mouth_rois
        import cv2
        frames, anchors, stamps = [], [], []
        t_end = time.time() + seconds
        while time.time() < t_end:
            frame, t = self.read_frame()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obs = tracker.detect(frame, int(t * 1000))
            frames.append(gray)
            anchors.append(obs.anchors if obs else None)
            stamps.append(t)
            if on_frame:
                on_frame(frame)
        if not frames:
            return None, []
        # Resample FIRST (Lipflow does this in dictation.py/offline.py):
        # the camera runs ~30 fps, the model was trained on 25 fps.
        _add_upstream()
        from lipflow.vsr import LipReader
        idx = LipReader.resample(stamps, len(frames))
        frames = [frames[i] for i in idx]
        anchors = [anchors[i] for i in idx]
        stamps = [stamps[i] for i in idx]
        rois = mouth_rois(frames, anchors)
        return rois, stamps

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
