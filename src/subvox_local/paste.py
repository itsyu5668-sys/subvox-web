"""Paste at cursor, system-wide (macOS). Linux fallback: clipboard only.

macOS: copy text to the clipboard with pbcopy, then simulate Cmd+V with
Quartz CGEvent — the same approach Lipflow's paste.py uses. Requires the
Accessibility permission for the app (asked once in setup).
Linux: xclip / wl-copy if present, otherwise print the text.
UNTESTED ON MAC.
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def copy_only(text: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    elif shutil.which("wl-copy"):
        subprocess.run(["wl-copy"], input=text.encode(), check=True)
    elif shutil.which("xclip"):
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
    else:
        print(text)


def paste(text: str) -> None:
    """Copy + Cmd+V. On macOS only; elsewhere, clipboard only."""
    copy_only(text)
    if sys.platform != "darwin":
        # copy_only already printed the text if no clipboard tool exists;
        # do not print it a second time here.
        return
    # Quartz key event: Cmd+V. UNTESTED ON MAC in this build.
    import Quartz
    from Quartz import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap
    key_v = 9  # kVK_ANSI_V
    flags = Quartz.kCGEventFlagMaskCommand
    down = CGEventCreateKeyboardEvent(None, key_v, True)
    Quartz.CGEventSetFlags(down, flags)
    CGEventPost(kCGHIDEventTap, down)
    up = CGEventCreateKeyboardEvent(None, key_v, False)
    Quartz.CGEventSetFlags(up, flags)
    CGEventPost(kCGHIDEventTap, up)
