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

    def test_detect_returns_none_when_state_has_no_rev(self):
        sock = self.sidecar(100, "/dev/pts/4")
        achat = FakeAchat({"ok": True, "known": True, "text": "", "cursor": 0})
        self.assertIsNone(AchatTarget.detect(desktop(), input_dir=self.dir, request=achat,
                                             pid_alive=lambda pid: True))

    def test_detect_returns_none_when_rev_is_null(self):
        sock = self.sidecar(100, "/dev/pts/4")
        achat = FakeAchat({"ok": True, "rev": None, "known": True, "text": "", "cursor": 0})
        self.assertIsNone(AchatTarget.detect(desktop(), input_dir=self.dir, request=achat,
                                             pid_alive=lambda pid: True))

    def test_detect_returns_none_when_rev_is_bool(self):
        sock = self.sidecar(100, "/dev/pts/4")
        achat = FakeAchat({"ok": True, "rev": True, "known": True, "text": "", "cursor": 0})
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

    def test_success_reply_without_rev_returns_false_and_marks_dead(self):
        achat = FakeAchat({"ok": True})  # Missing rev
        t = target(achat)
        self.assertFalse(t.send(Edit(0, "abc")))
        self.assertTrue(t.dead)
        self.assertFalse(t.still_focused())

    def test_success_reply_with_null_rev_returns_false_and_marks_dead(self):
        achat = FakeAchat({"ok": True, "rev": None})
        t = target(achat)
        self.assertFalse(t.send(Edit(0, "abc")))
        self.assertTrue(t.dead)

    def test_success_reply_with_bool_rev_returns_false_and_marks_dead(self):
        achat = FakeAchat({"ok": True, "rev": True})
        t = target(achat)
        self.assertFalse(t.send(Edit(0, "abc")))
        self.assertTrue(t.dead)

    def test_conflict_reply_with_null_rev_leaves_rev_unchanged(self):
        achat = FakeAchat({"ok": False, "error": "conflict", "rev": None})
        t = target(achat)
        t.send(Edit(3, "abc"))
        self.assertEqual(t.rev, 10)  # unchanged

    def test_append_retry_without_rev_marks_dead(self):
        achat = FakeAchat(
            {"ok": False, "error": "conflict", "rev": 12},
            {"ok": True, "rev": 12, "known": True, "text": "hi", "cursor": 2},
            {"ok": True},  # Missing rev in append retry
        )
        t = target(achat)
        self.assertFalse(t.send(Edit(0, " two")))
        self.assertTrue(t.dead)


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
