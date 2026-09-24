"""Output targets for live dictation: apply an Edit (backspace N, then insert).

A target is pinned to what had focus when it was created; still_focused()
reports whether that is still where the operator is working.
"""

import json
import subprocess


def wezterm_focused_pane(run=subprocess.run):
    """Pane focused in the most recently active WezTerm client, or None."""
    try:
        res = run(["wezterm", "cli", "list-clients", "--format", "json"],
                  capture_output=True, text=True, timeout=2)
    except Exception:
        return None
    if res.returncode != 0:
        return None
    try:
        entries = json.loads(res.stdout or "[]")
    except ValueError:
        return None
    best = None
    for entry in entries:
        pane = entry.get("focused_pane_id")
        if pane is None:
            continue
        idle = entry.get("idle_time") or {}
        idle_s = float(idle.get("secs", 0)) + float(idle.get("nanos", 0)) / 1e9
        if best is None or idle_s < best[0]:
            best = (idle_s, int(pane))
    return None if best is None else best[1]


def xdotool_active_window(run=subprocess.run):
    """X id of the active window as a string, or None."""
    try:
        res = run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=1)
    except Exception:
        return None
    if res.returncode != 0:
        return None
    return (res.stdout or "").strip() or None


class WezTermTarget:
    """Types into one WezTerm pane via `wezterm cli send-text --no-paste`."""

    name = "wezterm"

    def __init__(self, pane_id, window_id, run=subprocess.run):
        self.pane_id = int(pane_id)
        self.window_id = str(window_id)
        self._run = run

    @classmethod
    def detect(cls, run=subprocess.run):
        window = xdotool_active_window(run)
        pane = wezterm_focused_pane(run)
        if window is None or pane is None:
            return None
        return cls(pane, window, run)

    def send(self, edit):
        text = "\x7f" * edit.backspace + edit.insert
        if not text:
            return True
        try:
            res = self._run(
                ["wezterm", "cli", "send-text", "--pane-id", str(self.pane_id), "--no-paste"],
                input=text.encode("utf-8"), capture_output=True, timeout=5,
            )
        except Exception:
            return False
        return res.returncode == 0

    def still_focused(self):
        if xdotool_active_window(self._run) != self.window_id:
            return False
        return wezterm_focused_pane(self._run) == self.pane_id


class XdotoolTarget:
    """Types into the active X window with xdotool."""

    name = "xdotool"

    def __init__(self, window_id, run=subprocess.run):
        self.window_id = str(window_id)
        self._run = run

    @classmethod
    def detect(cls, run=subprocess.run):
        window = xdotool_active_window(run)
        return cls(window, run) if window else None

    def send(self, edit):
        try:
            if edit.backspace > 0:
                res = self._run(
                    ["xdotool", "key", "--clearmodifiers", "--delay", "8",
                     "--repeat", str(edit.backspace), "BackSpace"],
                    capture_output=True, timeout=10,
                )
                if res.returncode != 0:
                    return False
            if edit.insert:
                res = self._run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "8", "--", edit.insert],
                    capture_output=True, timeout=30,
                )
                if res.returncode != 0:
                    return False
        except Exception:
            return False
        return True

    def still_focused(self):
        return xdotool_active_window(self._run) == self.window_id


def choose_target(wm_class, run=subprocess.run):
    """WezTerm pane target when a WezTerm window is focused, else xdotool."""
    if "wezterm" in (wm_class or "").lower():
        target = WezTermTarget.detect(run)
        if target is not None:
            return target
    return XdotoolTarget.detect(run)
