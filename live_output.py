"""Output targets for live dictation: apply an Edit (backspace N, then insert).

A target is pinned to what had focus when it was created; still_focused()
reports whether that is still where the operator is working.
"""

import json
import subprocess
import unicodedata


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


_PROMPT_MARKS = ("$ ", "# ", "% ", "> ", "\u276f ", "\u276f\u00a0", "\u203a ", "\u203a\u00a0")


def _cells(ch):
    """Terminal cells a character takes (wide East Asian and emoji take two)."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def wezterm_line_context(pane_id, run=subprocess.run):
    """(text before the cursor, text after it) on the pane's cursor row, or None.

    Read off the screen, so it is approximate: a prompt symbol at the start
    (`❯`, `›`, `$`) is dropped, and a wrapped input only shows its last row.
    Good enough to decide case and spacing, never used to decide backspaces.
    """
    try:
        res = run(["wezterm", "cli", "list", "--format", "json"],
                  capture_output=True, text=True, timeout=2)
        pane = next((p for p in json.loads(res.stdout or "[]") if p.get("pane_id") == pane_id), None)
        if res.returncode != 0 or pane is None:
            return None
        row = int(pane["cursor_y"])
        res = run(["wezterm", "cli", "get-text", "--pane-id", str(pane_id),
                   "--start-line", str(row), "--end-line", str(row)],
                  capture_output=True, text=True, timeout=2)
        if res.returncode != 0:
            return None
        line = (res.stdout or "").rstrip("\n")
        cursor_x = int(pane["cursor_x"])
    except Exception:
        return None
    cells, split = 0, len(line)
    for i, ch in enumerate(line):
        if cells >= cursor_x:
            split = i
            break
        cells += _cells(ch)
    before, after = line[:split], line[split:].rstrip()
    # Drop the prompt: everything up to the first prompt marker (a shell's
    # "user@host:~$ "), then leading symbols up to the first word character.
    marks = [i + len(m) for m in _PROMPT_MARKS for i in [before.find(m)] if i >= 0]
    if marks:
        before = before[min(marks):]
    start = next((i for i, ch in enumerate(before) if ch.isalnum()), len(before))
    return before[start:], after


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

    def line_context(self):
        """Text before the cursor, read off the screen. The text after it is left
        out: a TUI's hardware cursor can sit off the input row, so what follows
        it on screen is not reliably what follows the operator's cursor."""
        context = wezterm_line_context(self.pane_id, self._run)
        return None if context is None else (context[0], "")


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


def choose_target(wm_class, run=subprocess.run, achat_detect=None):
    """Best target for the focused window.

    In WezTerm: the pane's `achat run` input socket when there is one, else
    the pane itself. Anything else: xdotool.
    """
    if "wezterm" in (wm_class or "").lower():
        if achat_detect is None:
            from live_achat import AchatTarget
            achat_detect = AchatTarget.detect
        target = achat_detect(run)
        if target is not None:
            return target
        target = WezTermTarget.detect(run)
        if target is not None:
            return target
    return XdotoolTarget.detect(run)
