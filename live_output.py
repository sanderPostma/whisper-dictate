"""Output targets for live dictation: apply an Edit (backspace N, then insert).

A target is pinned to what had focus when it was created; still_focused()
reports whether that is still where the operator is working.
"""

import json
import re
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


_TERMINALS = ("wezterm", "terminal", "konsole", "kitty", "alacritty", "xterm", "terminator",
              "tilix", "urxvt", "foot", "ghostty")


def is_terminal(wm_class):
    return any(t in (wm_class or "").lower() for t in _TERMINALS)


def clipboard_key(action, terminal):
    """The key for "paste" / "copy" / "cut", or None where there is none.

    Terminals: Ctrl+Shift+V / C (Ctrl+C would interrupt the program there),
    and no cut: terminal text can only be selected, not cut."""
    if terminal:
        return {"paste": "ctrl+shift+v", "copy": "ctrl+shift+c"}.get(action)
    return {"paste": "ctrl+v", "copy": "ctrl+c", "cut": "ctrl+x"}.get(action)


def press_key(key, run=subprocess.run):
    """Press a key chord in the focused window."""
    try:
        res = run(["xdotool", "key", "--clearmodifiers", key], capture_output=True, timeout=5)
    except Exception:
        return False
    return res.returncode == 0


_PROMPT_MARKS = ("$ ", "# ", "% ", "> ", "\u276f ", "\u276f\u00a0", "\u203a ", "\u203a\u00a0")


def _cells(ch):
    """Terminal cells a character takes (wide East Asian and emoji take two)."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


_SGR = re.compile(r"\x1b\[([0-9;:]*)m|\x1b\([0-9A-Za-z]|\x1b\[[0-9;?]*[A-Za-z]")
_RULE = "\u2500"
_PROMPT = "\u276f"


def _visible(row):
    """(all text, text not drawn muted) of one row with escapes.

    Muted: dim, or the grey Claude Code uses for placeholders and hints."""
    plain, typed, muted = [], [], False
    pos = 0
    for m in _SGR.finditer(row):
        chunk = row[pos:m.start()]
        plain.append(chunk)
        if not muted:
            typed.append(chunk)
        pos = m.end()
        params = m.group(1)
        if params is None:
            continue
        parts = params.replace(":", ";").split(";") if params else ["0"]
        if params in ("", "0", "39", "22") or params.startswith(("0;", "39;")):
            muted = False
        if "2" == parts[0] or params.startswith("0;2") or params.startswith("38;2;;153;153;153") \
                or params.startswith("38:2::153:153:153"):
            muted = True
        elif params.startswith("38") and not params.startswith(("38:2::153", "38;2;;153")):
            muted = False
    plain.append(row[pos:])
    if not muted:
        typed.append(row[pos:])
    return "".join(plain), "".join(typed)


def claude_input_before(screen, cursor_x, cursor_y):
    """Text before the cursor in a Claude Code style input box, or None.

    The box is the last "❯" row right under a full-width rule, down to the
    next rule. Its text is read off the screen (placeholders left out), so
    it does not depend on the terminal's cursor, which such TUIs park
    anywhere. The cursor only counts when it sits inside the box.
    """
    rows = [_visible(r.rstrip("\r")) for r in (screen or "").split("\n")]
    is_rule = [len(p.strip()) >= 20 and set(p.strip()) == {_RULE} for p, _ in rows]
    for top in range(len(rows) - 2, -1, -1):
        if not is_rule[top] or not rows[top + 1][0].lstrip().startswith(_PROMPT):
            continue
        bottom = next((i for i in range(top + 2, len(rows)) if is_rule[i]), None)
        if bottom is None:
            continue
        parts = []
        for i in range(top + 1, bottom):
            plain, typed = rows[i]
            at_cursor = top < cursor_y < bottom and i == cursor_y
            if at_cursor and typed.strip(" \u00a0" + _PROMPT):
                cells, cut = 0, len(plain)
                for j, ch in enumerate(plain):
                    if cells >= cursor_x:
                        cut = j
                        break
                    cells += _cells(ch)
                typed = plain[:cut]
            text = typed.lstrip().lstrip(_PROMPT).lstrip(" \u00a0") if i == top + 1 else typed.lstrip()
            if at_cursor:
                head = " ".join(p for p in parts if p)
                return head + " " + text if head and text else head + text
            parts.append(text.rstrip())
        return " ".join(p for p in parts if p)
    return None


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
        cursor_x = int(pane["cursor_x"])
        res = run(["wezterm", "cli", "get-text", "--pane-id", str(pane_id), "--escapes"],
                  capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            boxed = claude_input_before(res.stdout, cursor_x, row)
            if boxed is not None:
                return boxed, ""
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

    def press_enter(self):
        return self.send_keys(b"\r")

    def clear_line(self):
        """Readline: Ctrl+E to the end, Ctrl+U to kill back to the start."""
        return self.send_keys(b"\x05\x15")

    def move_cursor(self, move):
        """Readline keys: Alt+B / Alt+F per word, Ctrl+A / Ctrl+E for the ends."""
        if move.kind == "start":
            return self.send_keys(b"\x01")
        if move.kind == "end":
            return self.send_keys(b"\x05")
        return self.send_keys((b"\x1bb" if move.direction < 0 else b"\x1bf") * move.count)

    def send_keys(self, data):
        """Raw bytes to the pane as typed input (\r is Enter)."""
        try:
            res = self._run(
                ["wezterm", "cli", "send-text", "--pane-id", str(self.pane_id), "--no-paste"],
                input=data, capture_output=True, timeout=5,
            )
        except Exception:
            return False
        return res.returncode == 0

    def clipboard(self, action):
        """Paste / copy with the terminal's own keys (cut: none)."""
        key = clipboard_key(action, terminal=True)
        return bool(key) and press_key(key, self._run)

    SCREEN_ROWS = 15  # the input box sits near the bottom of the pane

    @staticmethod
    def line_end_in(screen, text, rows=SCREEN_ROWS, tail_chars=24, min_chars=6):
        """Whether one of the last rows of screen ends with the end of text."""
        tail = (text or "").split("\n")[-1][-tail_chars:].strip()
        if len(tail) < min_chars:
            return False
        lines = [r.rstrip() for r in (screen or "").split("\n") if r.strip()]
        return any(r.endswith(tail) for r in lines[-rows:])

    def shows_line_end(self, text, tail_chars=24):
        """Whether the pane shows text at the end of a row near the bottom.

        For small repairs only (a punctuation mark): a TUI like Claude Code
        draws its own cursor, so the cursor row read by line_context can be
        another row than the input line.
        """
        try:
            res = self._run(["wezterm", "cli", "get-text", "--pane-id", str(self.pane_id)],
                            capture_output=True, text=True, timeout=2)
        except Exception:
            return False
        return res.returncode == 0 and self.line_end_in(res.stdout, text, tail_chars=tail_chars)

    def line_context(self):
        """Text before the cursor, read off the screen. The text after it is left
        out: a TUI's hardware cursor can sit off the input row, so what follows
        it on screen is not reliably what follows the operator's cursor."""
        context = wezterm_line_context(self.pane_id, self._run)
        return None if context is None else (context[0], "")


class XdotoolTarget:
    """Types into the active X window with xdotool."""

    name = "xdotool"

    def __init__(self, window_id, run=subprocess.run, terminal=False):
        self.window_id = str(window_id)
        self._run = run
        self.terminal = terminal  # a terminal other than WezTerm: its clipboard keys

    def clipboard(self, action):
        key = clipboard_key(action, self.terminal)
        return bool(key) and press_key(key, self._run)

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

    def press_enter(self):
        return self._key("Return")

    def move_cursor(self, move):
        """Editor keys: Ctrl+Left / Ctrl+Right per word, Home / End."""
        if move.kind in ("start", "end"):
            return self._key("Home" if move.kind == "start" else "End")
        return self._key("ctrl+Left" if move.direction < 0 else "ctrl+Right", repeat=move.count)

    def _key(self, key, repeat=1):
        cmd = ["xdotool", "key", "--clearmodifiers"]
        if repeat > 1:
            cmd += ["--repeat", str(repeat)]
        try:
            res = self._run(cmd + [key], capture_output=True, timeout=10)
        except Exception:
            return False
        return res.returncode == 0


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
    target = XdotoolTarget.detect(run)
    if target is not None:
        target.terminal = is_terminal(wm_class)
    return target
