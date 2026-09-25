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
                        type_into_line, wezterm_pane_tty)
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
    def test_line_context_reads_known_line(self):
        achat = FakeAchat({"ok": True, "rev": 4, "known": True, "text": "fix the", "cursor": 7})
        self.assertEqual(target(achat).line_context(), ("fix the", ""))

    def test_line_context_splits_at_the_cursor(self):
        achat = FakeAchat({"ok": True, "rev": 4, "known": True,
                           "text": "His is a test.", "cursor": 9})
        self.assertEqual(target(achat).line_context(), ("His is a ", "test."))

    def test_line_context_adopts_current_rev(self):
        # After a submitted prompt the first append must not hit a stale rev.
        achat = FakeAchat({"ok": True, "rev": 40, "known": True, "text": "", "cursor": 0})
        t = target(achat)
        t.line_context()
        self.assertEqual(t.rev, 40)

    def test_line_context_ignores_bad_reply(self):
        t = target(FakeAchat(["not", "a", "dict"]))
        self.assertIsNone(t.line_context())

    def test_line_context_none_when_unknown(self):
        achat = FakeAchat({"ok": True, "rev": 4, "known": False, "text": "", "cursor": 0})
        self.assertIsNone(target(achat).line_context())

    def test_line_context_none_when_socket_gone(self):
        self.assertIsNone(target(FakeAchat(ConnectionRefusedError())).line_context())

    def test_dropped_flag(self):
        achat = FakeAchat(
            {"ok": True, "rev": 11},
            {"ok": False, "error": "conflict", "rev": 12},
            {"ok": False, "error": "cursor_not_at_end", "rev": 13},
            {"ok": False, "error": "conflict", "rev": 13},
            {"ok": True, "rev": 13, "known": True, "text": "hi", "cursor": 2},
            {"ok": True, "rev": 15},
        )
        t = target(achat)
        t.send(Edit(0, "a"))
        self.assertFalse(t.last_send_dropped)
        t.send(Edit(1, "b"))  # refused correction: the fast text is still there
        self.assertFalse(t.last_send_dropped)
        t.send(Edit(0, "c"))  # append refused by an achat without at_cursor: words lost
        self.assertTrue(t.last_send_dropped)
        t.send(Edit(0, "d"))  # append retried after operator text: typed
        self.assertFalse(t.last_send_dropped)


    def test_edit_is_compare_and_swap_and_tracks_rev(self):
        achat = FakeAchat({"ok": True, "rev": 13}, {"ok": True, "rev": 17})
        t = target(achat)
        self.assertTrue(t.send(Edit(0, "abc")))
        self.assertTrue(t.send(Edit(2, "xy")))
        self.assertEqual(achat.requests, [
            {"op": "edit", "expect_rev": 10, "backspace": 0, "insert": "abc", "at_cursor": True},
            {"op": "edit", "expect_rev": 13, "backspace": 2, "insert": "xy", "at_cursor": True},
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

    def test_append_retry_drops_its_space_after_operator_space(self):
        achat = FakeAchat(
            {"ok": False, "error": "conflict", "rev": 12},
            {"ok": True, "rev": 12, "known": True, "text": "hi ok ", "cursor": 6},
            {"ok": True, "rev": 15},
        )
        target(achat).send(Edit(0, " two"))
        self.assertEqual(achat.requests[2]["insert"], "two")

    def test_append_retry_works_mid_line(self):
        achat = FakeAchat(
            {"ok": False, "error": "conflict", "rev": 12},
            {"ok": True, "rev": 12, "known": True, "text": "His is a test", "cursor": 9},
            {"ok": True, "rev": 15},
        )
        target(achat).send(Edit(0, " quick "))
        self.assertEqual(achat.requests[2]["insert"], "quick ")

    def test_retry_respaces_for_the_new_cursor_position(self):
        # Built for "a |test" but the cursor moved to "widget|, x".
        achat = FakeAchat(
            {"ok": False, "error": "conflict", "rev": 12},
            {"ok": True, "rev": 12, "known": True, "text": "widget, x", "cursor": 6},
            {"ok": True, "rev": 15},
        )
        target(achat).send(Edit(0, "quick "))
        self.assertEqual(achat.requests[2]["insert"], " quick")

    def test_retry_adds_space_before_a_following_word(self):
        achat = FakeAchat(
            {"ok": False, "error": "conflict", "rev": 12},
            {"ok": True, "rev": 12, "known": True, "text": "a bar", "cursor": 2},
            {"ok": True, "rev": 15},
        )
        target(achat).send(Edit(0, "quick"))
        self.assertEqual(achat.requests[2]["insert"], "quick ")


        achat = FakeAchat({"ok": False, "error": "conflict", "rev": 12})
        t = target(achat)
        self.assertFalse(t.send(Edit(3, "abc")))
        self.assertEqual(len(achat.requests), 1)
        self.assertEqual(t.rev, 12)

    def test_append_follows_a_moved_cursor(self):
        # The operator moved the cursor mid-line: the words go where it is now.
        achat = FakeAchat(
            {"ok": False, "error": "conflict", "rev": 11},
            {"ok": True, "rev": 11, "known": True, "text": "hello", "cursor": 1},
            {"ok": True, "rev": 12},
        )
        self.assertFalse(target(achat).send(Edit(0, "x")))
        self.assertEqual(achat.requests[2]["expect_rev"], 11)

    def test_append_on_an_unknown_draft_goes_as_keys(self):
        t = target(FakeAchat({"ok": False, "error": "unknown_line"}))
        self.assertTrue(t.send(Edit(0, "x")))
        self.assertFalse(t.exact_line)

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


def state(text, cursor=None, rev=20, known=True):
    return {"ok": True, "rev": rev, "known": known, "text": text,
            "cursor": len(text) if cursor is None else cursor}


class TypeIntoLineTests(unittest.TestCase):
    """One-shot dictation into an achat prompt: shaped by the text around the cursor."""

    def test_continuing_a_sentence_lowercases_and_spaces(self):
        achat = FakeAchat(state("This is"), {"ok": True, "rev": 38})
        self.assertTrue(type_into_line(target(achat), "The full sentence"))
        self.assertEqual(achat.requests[1]["insert"], " the full sentence")
        self.assertEqual(achat.requests[1]["expect_rev"], 20)

    def test_after_a_full_stop_keeps_the_capital(self):
        achat = FakeAchat(state("Done. "), {"ok": True, "rev": 30})
        type_into_line(target(achat), "Next one")
        self.assertEqual(achat.requests[1]["insert"], "Next one")

    def test_mid_line_insert(self):
        achat = FakeAchat(state("His is a test.", cursor=9), {"ok": True, "rev": 30})
        type_into_line(target(achat), "Quick.")
        self.assertEqual(achat.requests[1]["insert"], "quick ")

    def test_empty_line_keeps_text_as_is(self):
        achat = FakeAchat(state(""), {"ok": True, "rev": 30})
        type_into_line(target(achat), "Hello there")
        self.assertEqual(achat.requests[1]["insert"], "Hello there")

    def test_unknown_line_is_not_handled(self):
        achat = FakeAchat(state("", known=False))
        self.assertFalse(type_into_line(target(achat), "Hello"))
        self.assertEqual(len(achat.requests), 1)

    def test_typed_after_operator_retry_counts_as_handled(self):
        achat = FakeAchat(state("This is"), {"ok": False, "error": "conflict", "rev": 21},
                          state("This is more"), {"ok": True, "rev": 40})
        self.assertTrue(type_into_line(target(achat), "The end"))

    def test_dropped_text_is_not_handled(self):
        achat = FakeAchat(state("This is"), {"ok": False, "error": "cursor_not_at_end", "rev": 21})
        self.assertFalse(type_into_line(target(achat), "The end"))

    def test_nothing_to_type_is_handled(self):
        achat = FakeAchat(state("This is"))
        self.assertTrue(type_into_line(target(achat), " ... "))
        self.assertEqual(len(achat.requests), 1)


class UnknownDraftFallbackTests(unittest.TestCase):
    """A multi-line draft is Unknown to achat: use the pane like a plain WezTerm pane."""

    def make(self, *responses):
        self.run = FakeRun({
            LIST_CLIENTS: json.dumps([{"focused_pane_id": 7, "idle_time": {"secs": 0, "nanos": 0}}]),
            LIST_PANES + ("--format",): json.dumps([{"pane_id": 7, "tty_name": "/dev/pts/4",
                                                    "cursor_x": 11, "cursor_y": 3}]),
            ("wezterm", "cli", "get-text"): "> line one\n",
            ACTIVE_WINDOW: "42\n",
        })
        self.sent = []
        run = self.run

        def recording_run(cmd, **kwargs):
            if cmd[:3] == ["wezterm", "cli", "send-text"]:
                self.sent.append(kwargs["input"])
            return run(cmd, **kwargs)

        return AchatTarget(7, "42", "/x.sock", 10, run=recording_run, request=FakeAchat(*responses),
                           sleep=lambda s: None, log=lambda *_: None)

    def unknown(self):
        return {"ok": True, "rev": 12, "known": False, "text": "", "cursor": 0}

    def test_typing_into_an_unknown_draft_uses_keystrokes(self):
        t = self.make({"ok": False, "error": "unknown_line"}, self.unknown())
        self.assertTrue(t.send(Edit(0, " more")))
        self.assertEqual(self.sent, [b" more"])
        self.assertFalse(t.exact_line)

    def test_line_context_of_an_unknown_draft_comes_from_the_screen(self):
        t = self.make(self.unknown())
        self.assertEqual(t.line_context(), ("line one", ""))
        self.assertFalse(t.exact_line)

    def test_known_again_is_exact_again(self):
        t = self.make(self.unknown(), {"ok": True, "rev": 20, "known": True, "text": "", "cursor": 0})
        t.line_context()
        self.assertEqual(t.line_context(), ("", ""))
        self.assertTrue(t.exact_line)

    def test_cursor_move_on_an_unknown_draft_uses_readline_keys(self):
        from live_commands import CursorMove
        t = self.make(self.unknown())
        self.assertTrue(t.move_cursor(CursorMove("word", -1, 2)))
        self.assertEqual(self.sent, [b"\x1bb\x1bb"])

    def test_clear_on_an_unknown_draft_uses_readline_keys(self):
        t = self.make(self.unknown())
        self.assertTrue(t.clear_line())
        self.assertEqual(self.sent, [b"\x05\x15"])


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
