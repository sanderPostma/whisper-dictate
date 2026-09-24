# Live Dictation Phase 2: achat Input Socket — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the focused WezTerm pane runs an `achat run` session, live dictation types into the agent's prompt line through achat's input-control socket. Every edit is a compare-and-swap, so a correction can never backspace over text the operator typed or over a nudge in flight.

**Architecture:** One new module, `live_achat.py`, holding `AchatTarget` and the socket discovery. `choose_target` prefers it inside WezTerm. `LiveController._send` learns that an achat rejection means "commit the window", not "output is broken", so corrections carry on.

**Tech Stack:** Python 3 stdlib (`socket`, `json`, `glob`). Tests use `unittest` via `venv/bin/python -m unittest`.

**Spec:** `docs/superpowers/specs/2026-09-24-live-dictation-design.md`, section "Phase 2". The protocol is documented in agentic-chat's `docs/operator-guide.md`, section "The input-control socket".

**Verification status:**
- All code below ran in a scratch copy: 108 live tests pass.
- It was also driven end to end against a real `achat run -- bash` socket (achat build 40a9e99). A scripted controller typed, corrected "brunch" to "branch" in place, then carried on appending after injected operator keystrokes without touching them.

## Deviations from the spec (deliberate)

1. **Discovery matches the sidecar's `tty` against the pane's `tty_name`** (from `wezterm cli list`), not `wezterm_pane`. `wezterm_pane` is inherited from the environment, so an `achat run` nested inside a pane (tmux, `script`, an agent's own shell) claims the pane too. This was observed on the real machine. A session under tmux therefore falls back to `WezTermTarget` keystrokes.
2. **A rejected append is retried once on a fresh revision** when the line is Known with the cursor at the end: the operator typed, so dictation continues after their text. `send` still returns False so the window is committed, and a later correction cannot reach back past the operator's text. That case was caught in the real-socket run: the compare-and-swap is checked against our own newest revision, so it cannot guard it.
3. **`busy` (nudge mid-steal) is retried** 8 times with a 0.25 s back-off, as achat's guide asks. After that the edit is dropped and the window committed.
4. **Control bytes in the text become spaces** before sending. The socket refuses them with `invalid_insert`. One character is swapped for one, so the session's character counts stay right.
5. **No `subscribe`** yet. The compare-and-swap already makes corrections safe, and `subscribe` only matters for the voice commands planned later.

## Global Constraints

- achat socket: `$ACHAT_INPUT_DIR` or `$XDG_RUNTIME_DIR/achat/input` or `<tmp>/achat-<uid>/input`. It holds `<pid>.sock` and `<pid>.json`, and speaks newline-delimited JSON with one response per request.
- `edit` = `{"op":"edit","expect_rev":R,"backspace":N,"insert":S}`. Success is `{"ok":true,"rev":R2}`. Errors: `conflict`, `unknown_line`, `cursor_not_at_end`, `busy`, `range`, `invalid_insert`, `bad_request`.
- Stale sidecars (dead pid, missing socket) are skipped, never deleted: they belong to achat.
- Never backspace over text whisper-dictate did not type.

## Review Focus

- The operator types in the agent prompt between a fast pass and its correction. Expect: the correction is dropped, the operator's text is untouched, and dictation continues after it.
- A nudge arrives mid-dictation (`busy`). Expect: the edit waits, then lands; nothing interleaves.
- `achat run` exits mid-session. Expect: the socket error marks the target dead, `still_focused` goes False, and the controller retargets to plain WezTerm.
- A sidecar is left by a killed wrapper. Expect: it is skipped, with no connect attempt beyond the liveness check.
- Two `achat run` sessions claim the same `wezterm_pane`. Expect: only the one on the pane's own tty is used.

---

### Task 6: AchatTarget

**Files:**
- Create: `live_achat.py`
- Create: `tests/test_live_achat.py`
- Modify: `live_output.py` (`choose_target`)
- Modify: `live_controller.py` (`_send`)
- Modify: `tests/test_live_controller.py` (one test)
- Modify: `README.md` (Live dictation section)

**Interfaces:**
- Consumes: `live_output.wezterm_focused_pane(run)` returns the pane id as an int or None; `live_output.xdotool_active_window(run)` returns the window id as a str or None; `live_session.Edit(backspace, insert)`; `LiveController._send(edit)`.
- Produces:
  - `AchatTarget(pane_id, window_id, sock_path, rev, run, request, sleep, log)` has `name="achat"`, `recoverable_rejections=True`, `send(edit) -> bool` and `still_focused() -> bool`.
  - `AchatTarget.detect(run, input_dir=None, request=socket_request, pid_alive=...)` returns the target or None.
  - `choose_target(wm_class, run, achat_detect=None)`.
- Contract change for `send`: False means "commit the window". A target with `recoverable_rejections` keeps corrections on; any other target disables them, as before.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_achat.py`:

```python
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_achat import (AchatTarget, achat_input_dir, find_session_socket, socket_request,
                        wezterm_pane_tty)
from live_output import WezTermTarget, choose_target
from live_session import Edit

LIST_CLIENTS = ("wezterm", "cli", "list-clients")
LIST_PANES = ("wezterm", "cli", "list")
ACTIVE_WINDOW = ("xdotool", "getactivewindow")


class FakeRun:
    """Stands in for subprocess.run: answers stdout by longest matching command prefix."""

    def __init__(self, answers):
        self.answers = answers

    def __call__(self, cmd, **kwargs):
        stdout, best = "", -1
        for prefix, out in self.answers.items():
            if tuple(cmd[:len(prefix)]) == prefix and len(prefix) > best:
                stdout, best = out, len(prefix)
        return SimpleNamespace(returncode=0, stdout=stdout)


def desktop(pane=7, tty="/dev/pts/4", window="42"):
    return FakeRun({
        LIST_CLIENTS: json.dumps([{"focused_pane_id": pane, "idle_time": {"secs": 0, "nanos": 0}}]),
        LIST_PANES + ("--format",): json.dumps([{"pane_id": pane, "tty_name": tty}]),
        ACTIVE_WINDOW: window + "\n",
    })


class FakeAchat:
    """Scripted socket: pops one response per request, records the requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, sock_path, payload):
        self.requests.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def target(achat, rev=10):
    return AchatTarget(7, "42", "/x.sock", rev, run=desktop(), request=achat,
                       sleep=lambda s: None, log=lambda *_: None)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def sidecar(self, pid, tty, sock=True, mtime=None):
        path = os.path.join(self.dir, f"{pid}.json")
        with open(path, "w") as f:
            json.dump({"pid": pid, "tty": tty, "wezterm_pane": "7"}, f)
        if sock:
            open(os.path.join(self.dir, f"{pid}.sock"), "w").close()
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return os.path.join(self.dir, f"{pid}.sock")

    def test_matches_on_tty_not_inherited_pane(self):
        self.sidecar(100, "/dev/pts/18")
        want = self.sidecar(200, "/dev/pts/4")
        self.assertEqual(find_session_socket("/dev/pts/4", self.dir, lambda pid: True), want)

    def test_dead_wrapper_is_skipped(self):
        self.sidecar(100, "/dev/pts/4")
        self.assertIsNone(find_session_socket("/dev/pts/4", self.dir, lambda pid: False))

    def test_sidecar_without_socket_is_skipped(self):
        self.sidecar(100, "/dev/pts/4", sock=False)
        self.assertIsNone(find_session_socket("/dev/pts/4", self.dir, lambda pid: True))

    def test_newest_live_sidecar_wins(self):
        self.sidecar(100, "/dev/pts/4", mtime=1000)
        want = self.sidecar(200, "/dev/pts/4", mtime=2000)
        self.assertEqual(find_session_socket("/dev/pts/4", self.dir, lambda pid: True), want)

    def test_garbage_sidecar_is_ignored(self):
        with open(os.path.join(self.dir, "9.json"), "w") as f:
            f.write("{nope")
        self.assertIsNone(find_session_socket("/dev/pts/4", self.dir, lambda pid: True))

    def test_no_tty_is_none(self):
        self.assertIsNone(find_session_socket(None, self.dir, lambda pid: True))

    def test_input_dir_follows_achat_rules(self):
        self.assertEqual(achat_input_dir({"ACHAT_INPUT_DIR": "/a", "XDG_RUNTIME_DIR": "/r"}), "/a")
        self.assertEqual(achat_input_dir({"XDG_RUNTIME_DIR": "/r"}), "/r/achat/input")
        self.assertTrue(achat_input_dir({}).endswith(f"achat-{os.getuid()}/input"))

    def test_pane_tty(self):
        self.assertEqual(wezterm_pane_tty(7, desktop()), "/dev/pts/4")
        self.assertIsNone(wezterm_pane_tty(8, desktop()))

    def test_detect_reads_rev_from_state(self):
        sock = self.sidecar(100, "/dev/pts/4")
        achat = FakeAchat({"ok": True, "rev": 5, "known": True, "text": "", "cursor": 0})
        t = AchatTarget.detect(desktop(), input_dir=self.dir, request=achat,
                               pid_alive=lambda pid: True)
        self.assertEqual((t.pane_id, t.window_id, t.sock_path, t.rev), (7, "42", sock, 5))

    def test_detect_without_session_is_none(self):
        self.assertIsNone(AchatTarget.detect(desktop(), input_dir=self.dir, request=FakeAchat()))

    def test_detect_unreachable_socket_is_none(self):
        self.sidecar(100, "/dev/pts/4")
        achat = FakeAchat(ConnectionRefusedError())
        self.assertIsNone(AchatTarget.detect(desktop(), input_dir=self.dir, request=achat,
                                             pid_alive=lambda pid: True))


class SendTests(unittest.TestCase):
    def test_edit_is_compare_and_swap_and_tracks_rev(self):
        achat = FakeAchat({"ok": True, "rev": 13}, {"ok": True, "rev": 17})
        t = target(achat)
        self.assertTrue(t.send(Edit(0, "abc")))
        self.assertTrue(t.send(Edit(2, "xy")))
        self.assertEqual(achat.requests, [
            {"op": "edit", "expect_rev": 10, "backspace": 0, "insert": "abc"},
            {"op": "edit", "expect_rev": 13, "backspace": 2, "insert": "xy"},
        ])
        self.assertEqual(t.rev, 17)

    def test_empty_edit_sends_nothing(self):
        achat = FakeAchat()
        self.assertTrue(target(achat).send(Edit(0, "")))
        self.assertEqual(achat.requests, [])

    def test_append_after_operator_typed_retries_on_fresh_rev(self):
        achat = FakeAchat(
            {"ok": False, "error": "conflict", "rev": 12},
            {"ok": True, "rev": 12, "known": True, "text": "hi", "cursor": 2},
            {"ok": True, "rev": 16},
        )
        t = target(achat)
        self.assertFalse(t.send(Edit(0, " two")))  # typed, but the window must be committed
        self.assertEqual(achat.requests[2]["expect_rev"], 12)
        self.assertEqual(t.rev, 16)

    def test_correction_never_backspaces_over_operator_text(self):
        achat = FakeAchat({"ok": False, "error": "conflict", "rev": 12})
        t = target(achat)
        self.assertFalse(t.send(Edit(3, "abc")))
        self.assertEqual(len(achat.requests), 1)
        self.assertEqual(t.rev, 12)

    def test_append_not_retried_when_cursor_moved(self):
        achat = FakeAchat(
            {"ok": False, "error": "cursor_not_at_end", "rev": 11},
            {"ok": True, "rev": 11, "known": True, "text": "hello", "cursor": 1},
        )
        self.assertFalse(target(achat).send(Edit(0, "x")))
        self.assertEqual(len(achat.requests), 2)

    def test_append_not_retried_on_unknown_line(self):
        achat = FakeAchat(
            {"ok": False, "error": "unknown_line"},
            {"ok": True, "rev": 11, "known": False, "text": "", "cursor": 0},
        )
        self.assertFalse(target(achat).send(Edit(0, "x")))

    def test_busy_is_retried(self):
        achat = FakeAchat({"ok": False, "error": "busy", "rev": 10}, {"ok": True, "rev": 11})
        self.assertTrue(target(achat).send(Edit(0, "x")))

    def test_busy_forever_gives_up(self):
        achat = FakeAchat(*[{"ok": False, "error": "busy", "rev": 10}] * 8)
        self.assertFalse(target(achat).send(Edit(0, "x")))
        self.assertEqual(len(achat.requests), 8)

    def test_control_bytes_become_spaces(self):
        achat = FakeAchat({"ok": True, "rev": 13})
        target(achat).send(Edit(0, "a\nb\tc"))
        self.assertEqual(achat.requests[0]["insert"], "a b c")

    def test_socket_gone_marks_target_dead(self):
        t = target(FakeAchat(ConnectionRefusedError()))
        self.assertFalse(t.send(Edit(0, "x")))
        self.assertFalse(t.still_focused())

    def test_still_focused_needs_same_pane_and_window(self):
        t = target(FakeAchat())
        self.assertTrue(t.still_focused())
        t.window_id = "99"
        self.assertFalse(t.still_focused())

    def test_rejections_are_recoverable(self):
        self.assertTrue(AchatTarget.recoverable_rejections)


class ChooseTargetTests(unittest.TestCase):
    WEZ = 'WM_CLASS(STRING) = "org.wezfurlong.wezterm", "org.wezfurlong.wezterm"'

    def test_achat_session_is_preferred_in_wezterm(self):
        sentinel = object()
        self.assertIs(choose_target(self.WEZ, desktop(), achat_detect=lambda run: sentinel), sentinel)

    def test_plain_wezterm_pane_without_session(self):
        t = choose_target(self.WEZ, desktop(), achat_detect=lambda run: None)
        self.assertIsInstance(t, WezTermTarget)

    def test_achat_not_consulted_outside_wezterm(self):
        def boom(run):
            raise AssertionError("achat detect called")
        self.assertIsNotNone(choose_target('WM_CLASS(STRING) = "firefox"', desktop(), achat_detect=boom))


class SocketRequestTests(unittest.TestCase):
    def test_round_trip_over_a_real_unix_socket(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        path = os.path.join(d, "s.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)
        self.addCleanup(server.close)
        seen = []

        def serve():
            conn, _ = server.accept()
            with conn, conn.makefile("rb") as f:
                seen.append(json.loads(f.readline()))
                conn.sendall(b'{"ok":true,"rev":3}\n')

        thread = threading.Thread(target=serve)
        thread.start()
        self.assertEqual(socket_request(path, {"op": "state"}), {"ok": True, "rev": 3})
        thread.join(2)
        self.assertEqual(seen, [{"op": "state"}])

    def test_missing_socket_raises_oserror(self):
        with self.assertRaises(OSError):
            socket_request("/nonexistent/achat.sock", {"op": "state"})


if __name__ == "__main__":
    unittest.main()
```

Add this test to `tests/test_live_controller.py`, before `test_long_pause_after_sentence_commits`:

```diff
--- tests/test_live_controller.py	2026-09-24 11:14:46.644330161 +0200
+++ /tmp/claude-1000/-mnt-nvme2tb-DEV-mine-whisper-dictate/fd332ca5-84d6-4da2-b286-3ca3c57a9f1c/scratchpad/p2/tests/test_live_controller.py	2026-09-24 17:02:46.033252160 +0200
@@ -142,6 +142,14 @@
         ctl.correct()
         self.assertEqual(correct.prompts, [])
 
+    def test_recoverable_rejection_commits_but_keeps_corrections(self):
+        target = FakeTarget(ok=False)
+        target.recoverable_rejections = True
+        ctl = self.make(Script("a"), Script(), target=target)
+        ctl.process(chunk(0.0, 1.0))
+        self.assertTrue(ctl.corrections_enabled)
+        self.assertEqual(ctl.session.chunk_count, 0)
+
     def test_long_pause_after_sentence_commits(self):
         ctl = self.make(Script("Done."))
         ctl.process(chunk(0.0, 1.0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m unittest tests.test_live_achat tests.test_live_controller`
Expected: an ImportError for `live_achat`, and `test_recoverable_rejection_commits_but_keeps_corrections` fails.

- [ ] **Step 3: Implement**

Create `live_achat.py`:

```python
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
        return cls(pane, window, sock, state.get("rev"), run, request, **kwargs)

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
        self.rev = state.get("rev")
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
            self.rev = resp.get("rev")
            return True
        self._log(f"[live] achat edit rejected: {resp.get('error')}")
        if "rev" in resp:
            self.rev = resp["rev"]
        if resp.get("error") in _LINE_MOVED and edit.backspace == 0 and self._refresh():
            # Appending after the operator's text is fine (backspacing over it
            # never is). Still report False: the window now ends in our text
            # but a correction must not reach back past theirs.
            resp = self._edit(0, insert)
            if resp is not None and resp.get("ok"):
                self.rev = resp.get("rev")
        return False

    def still_focused(self):
        if self.dead:
            return False
        if xdotool_active_window(self._run) != self.window_id:
            return False
        return wezterm_focused_pane(self._run) == self.pane_id
```

Change `live_controller.py` `_send`:

```diff
--- live_controller.py	2026-09-24 11:15:09.122294909 +0200
+++ /tmp/claude-1000/-mnt-nvme2tb-DEV-mine-whisper-dictate/fd332ca5-84d6-4da2-b286-3ca3c57a9f1c/scratchpad/p2/live_controller.py	2026-09-24 17:02:07.485300611 +0200
@@ -162,9 +162,12 @@
                 return True
         except Exception as e:
             self.log(f"[live] target.send raised: {e}")
-        self.log("[live] output failed; committing window and disabling corrections")
         self.session.commit()
-        self.corrections_enabled = False
+        if getattr(self.target, "recoverable_rejections", False):
+            self.log("[live] edit rejected; committing window")
+        else:
+            self.log("[live] output failed; committing window and disabling corrections")
+            self.corrections_enabled = False
         return False
 
     def _report(self, message):
```

Change `live_output.py` `choose_target` (lazy import: `live_achat` imports from `live_output`):

```diff
--- live_output.py	2026-09-24 11:11:20.134664208 +0200
+++ /tmp/claude-1000/-mnt-nvme2tb-DEV-mine-whisper-dictate/fd332ca5-84d6-4da2-b286-3ca3c57a9f1c/scratchpad/p2/live_output.py	2026-09-24 17:02:07.485447577 +0200
@@ -120,9 +120,19 @@
         return xdotool_active_window(self._run) == self.window_id
 
 
-def choose_target(wm_class, run=subprocess.run):
-    """WezTerm pane target when a WezTerm window is focused, else xdotool."""
+def choose_target(wm_class, run=subprocess.run, achat_detect=None):
+    """Best target for the focused window.
+
+    In WezTerm: the pane's `achat run` input socket when there is one, else
+    the pane itself. Anything else: xdotool.
+    """
     if "wezterm" in (wm_class or "").lower():
+        if achat_detect is None:
+            from live_achat import AchatTarget
+            achat_detect = AchatTarget.detect
+        target = achat_detect(run)
+        if target is not None:
+            return target
         target = WezTermTarget.detect(run)
         if target is not None:
             return target
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m unittest discover -s tests`
Expected: `Ran 129 tests` and `OK` (100 before, plus 28 in `test_live_achat`, plus 1 controller test).

- [ ] **Step 5: README**

In `README.md` "## Live dictation", replace the line `- WezTerm: typed into the pane focused at start via \`wezterm cli send-text\`.` with:

```markdown
- WezTerm pane running `achat run`: typed into the agent's prompt line through achat's
  input-control socket. Every edit is checked against the line's revision, so typing by
  hand or an incoming nudge is never overwritten: dictation just continues after it.
  Needs an `achat` build with the input socket; restart old `achat run` sessions.
- Other WezTerm panes: typed into the pane focused at start via `wezterm cli send-text`.
```

and change the start of the "Corrections work by blindly backspacing" bullet to `- Outside achat, corrections work by blindly backspacing`.

- [ ] **Step 6: Commit**

```bash
git add live_achat.py tests/test_live_achat.py live_output.py live_controller.py tests/test_live_controller.py README.md
git commit -m "Type live dictation into achat sessions through the input socket"
```

- [ ] **Step 7: Manual check (human)**

1. Restart one agent under the new `achat run` directly in a WezTerm pane (not in tmux). Start live dictation there: the status reads `⚡ Live → achat`.
2. Speak two sentences. Text appears in the agent prompt and "brunch"-style errors are corrected in place.
3. While speaking, type a few characters by hand. They survive, and dictation continues after them.
4. Get a nudge delivered during dictation. It is not garbled, and dictation resumes.
5. Quit the agent mid-session. Dictation continues in the pane via plain WezTerm typing.
