"""Live dictation into an `achat run` session through its input-control socket.

`achat run` models the agent's prompt line and accepts guarded edits: an
edit is a compare-and-swap against the line's revision, so a correction can
never backspace over text the operator typed or over a nudge in flight.
Protocol: agentic-chat docs/operator-guide.md, "The input-control socket".
"""

import glob
import json
import os
import socket
import subprocess
import tempfile
import time

from live_output import wezterm_focused_pane, xdotool_active_window

# Rejections meaning the operator (or a nudge) touched the line.
_LINE_MOVED = ("conflict", "cursor_not_at_end", "unknown_line")
BUSY_RETRIES = 8
BUSY_BACKOFF_S = 0.25


def _rev(value):
    """Extract rev if it's a valid integer (not bool), else None.

    achat treats null rev as 'skip the check', which disables the guard.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def achat_input_dir(env=None):
    """Directory holding `<pid>.sock` / `<pid>.json`, as `achat run` picks it."""
    env = os.environ if env is None else env
    if env.get("ACHAT_INPUT_DIR"):
        return env["ACHAT_INPUT_DIR"]
    if env.get("XDG_RUNTIME_DIR"):
        return os.path.join(env["XDG_RUNTIME_DIR"], "achat", "input")
    return os.path.join(tempfile.gettempdir(), f"achat-{os.getuid()}", "input")


def socket_request(sock_path, payload, timeout=2.0):
    """Send one JSON request line, return the parsed response line."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        with s.makefile("rb") as f:
            line = f.readline()
    if not line:
        raise ConnectionError("achat socket closed without a response")
    return json.loads(line)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def find_session_socket(tty, input_dir, pid_alive=_pid_alive):
    """Socket of the live `achat run` whose own terminal is `tty`, or None.

    Matched on the tty, not the sidecar's `wezterm_pane`: that value is
    inherited from the environment, so a session nested inside another
    terminal (tmux, script, an agent's shell) claims the pane too.
    Sidecars left behind by a killed wrapper are skipped.
    """
    if not tty:
        return None
    best = None
    for sidecar in glob.glob(os.path.join(input_dir, "*.json")):
        try:
            with open(sidecar, encoding="utf-8") as f:
                info = json.load(f)
            mtime = os.path.getmtime(sidecar)
        except (OSError, ValueError):
            continue
        if not isinstance(info, dict) or info.get("tty") != tty:
            continue
        if not pid_alive(info.get("pid")):
            continue
        sock = sidecar[:-len(".json")] + ".sock"
        if not os.path.exists(sock):
            continue
        if best is None or mtime > best[0]:
            best = (mtime, sock)
    return None if best is None else best[1]


def wezterm_pane_tty(pane_id, run=subprocess.run):
    """tty_name of a WezTerm pane, or None."""
    try:
        res = run(["wezterm", "cli", "list", "--format", "json"],
                  capture_output=True, text=True, timeout=2)
    except Exception:
        return None
    if res.returncode != 0:
        return None
    try:
        panes = json.loads(res.stdout or "[]")
    except ValueError:
        return None
    for pane in panes:
        if pane.get("pane_id") == pane_id:
            return pane.get("tty_name")
    return None


def _clean_insert(text):
    """The socket refuses control bytes; keep lengths so the session stays in sync."""
    return "".join(" " if ord(c) < 0x20 or ord(c) == 0x7F else c for c in text)


class AchatTarget:
    """Types into the prompt line of an `achat run` session in a WezTerm pane."""

    name = "achat"
    # A rejected edit means the operator touched the line: commit the window
    # and carry on, instead of giving up on corrections for the session.
    recoverable_rejections = True

    def __init__(self, pane_id, window_id, sock_path, rev, run=subprocess.run,
                 request=socket_request, sleep=time.sleep, log=print):
        self.pane_id = int(pane_id)
        self.window_id = str(window_id)
        self.sock_path = sock_path
        self.rev = rev
        self.dead = False
        self._run = run
        self._request = request
        self._sleep = sleep
        self._log = log

    @classmethod
    def detect(cls, run=subprocess.run, input_dir=None, request=socket_request,
               pid_alive=_pid_alive, **kwargs):
        window = xdotool_active_window(run)
        pane = wezterm_focused_pane(run)
        if window is None or pane is None:
            return None
        sock = find_session_socket(wezterm_pane_tty(pane, run),
                                   input_dir or achat_input_dir(), pid_alive)
        if sock is None:
            return None
        try:
            state = request(sock, {"op": "state"})
        except (OSError, ValueError):
            return None
        if not state.get("ok"):
            return None
        rev = _rev(state.get("rev"))
        if rev is None:
            return None
        return cls(pane, window, sock, rev, run, request, **kwargs)

    def _call(self, payload):
        try:
            return self._request(self.sock_path, payload)
        except (OSError, ValueError) as e:
            self._log(f"[live] achat socket gone: {e}")
            self.dead = True
            return None

    def _edit(self, backspace, insert):
        """One edit, retrying while a nudge is mid-steal. Returns the response."""
        payload = {"op": "edit", "expect_rev": self.rev, "backspace": backspace, "insert": insert}
        for _ in range(BUSY_RETRIES):
            resp = self._call(payload)
            if resp is None or resp.get("error") != "busy":
                return resp
            self._sleep(BUSY_BACKOFF_S)
        return resp

    def _refresh(self):
        """Re-read the line; return the state if an append would be safe now."""
        state = self._call({"op": "state"})
        if not state or not state.get("ok"):
            return None
        rev = _rev(state.get("rev"))
        if rev is None:
            return None
        self.rev = rev
        text = state.get("text") or ""
        if state.get("known") and state.get("cursor") == len(text):
            return state
        return None

    def send(self, edit):
        """Apply the edit. False means the typed window no longer matches the
        line (the operator touched it), so the caller must commit the window."""
        if not edit.backspace and not edit.insert:
            return True
        insert = _clean_insert(edit.insert)
        resp = self._edit(edit.backspace, insert)
        if resp is None:
            return False
        if resp.get("ok"):
            rev = _rev(resp.get("rev"))
            if rev is None:
                self._log("[live] achat reply without rev")
                self.dead = True
                return False
            self.rev = rev
            return True
        self._log(f"[live] achat edit rejected: {resp.get('error')}")
        if "rev" in resp:
            rev = _rev(resp["rev"])
            if rev is not None:
                self.rev = rev
        if resp.get("error") in _LINE_MOVED and edit.backspace == 0 and self._refresh():
            # Appending after the operator's text is fine (backspacing over it
            # never is). Still report False: the window now ends in our text
            # but a correction must not reach back past theirs.
            resp = self._edit(0, insert)
            if resp is not None and resp.get("ok"):
                rev = _rev(resp.get("rev"))
                if rev is not None:
                    self.rev = rev
                else:
                    self._log("[live] achat reply without rev")
                    self.dead = True
        return False

    def still_focused(self):
        if self.dead:
            return False
        if xdotool_active_window(self._run) != self.window_id:
            return False
        return wezterm_focused_pane(self._run) == self.pane_id
