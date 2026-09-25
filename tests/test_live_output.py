import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_output import (WezTermTarget, XdotoolTarget, choose_target, wezterm_focused_pane,
                         wezterm_line_context)
from live_session import Edit

LIST_CLIENTS = ("wezterm", "cli", "list-clients")
ACTIVE_WINDOW = ("xdotool", "getactivewindow")


class FakeRun:
    """Stands in for subprocess.run: records calls, answers stdout by command prefix."""

    def __init__(self, answers=None, returncode=0):
        self.calls = []
        self.answers = answers or {}
        self.returncode = returncode

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        stdout = ""
        for prefix, out in self.answers.items():
            if tuple(cmd[:len(prefix)]) == prefix:
                stdout = out
        return SimpleNamespace(returncode=self.returncode, stdout=stdout)


def clients(*entries):
    return json.dumps([
        {"focused_pane_id": pane, "idle_time": {"secs": secs, "nanos": 0}}
        for pane, secs in entries
    ])


class WezTermTests(unittest.TestCase):
    def test_focused_pane_picks_most_recently_active_client(self):
        run = FakeRun({LIST_CLIENTS: clients((3, 50), (7, 1))})
        self.assertEqual(wezterm_focused_pane(run), 7)

    def test_focused_pane_bad_json_is_none(self):
        self.assertIsNone(wezterm_focused_pane(FakeRun({LIST_CLIENTS: "nope"})))

    def test_focused_pane_command_failure_is_none(self):
        self.assertIsNone(wezterm_focused_pane(FakeRun({LIST_CLIENTS: clients((7, 0))}, returncode=1)))

    def test_detect_pins_pane_and_window(self):
        run = FakeRun({LIST_CLIENTS: clients((7, 0)), ACTIVE_WINDOW: "42\n"})
        target = WezTermTarget.detect(run)
        self.assertEqual((target.pane_id, target.window_id), (7, "42"))

    def test_send_backspaces_and_insert_via_stdin(self):
        run = FakeRun()
        self.assertTrue(WezTermTarget(7, "42", run).send(Edit(3, "abc")))
        cmd, kwargs = run.calls[0]
        self.assertEqual(cmd, ["wezterm", "cli", "send-text", "--pane-id", "7", "--no-paste"])
        self.assertEqual(kwargs["input"], b"\x7f\x7f\x7fabc")

    def test_send_failure_returns_false(self):
        self.assertFalse(WezTermTarget(7, "42", FakeRun(returncode=1)).send(Edit(0, "x")))

    def test_empty_edit_runs_nothing(self):
        run = FakeRun()
        self.assertTrue(WezTermTarget(7, "42", run).send(Edit(0, "")))
        self.assertEqual(run.calls, [])

    def test_still_focused_needs_same_pane_and_window(self):
        run = FakeRun({LIST_CLIENTS: clients((7, 0)), ACTIVE_WINDOW: "42\n"})
        self.assertTrue(WezTermTarget(7, "42", run).still_focused())
        self.assertFalse(WezTermTarget(8, "42", run).still_focused())

    def test_switching_to_another_x_window_is_focus_loss(self):
        run = FakeRun({LIST_CLIENTS: clients((7, 0)), ACTIVE_WINDOW: "99\n"})
        self.assertFalse(WezTermTarget(7, "42", run).still_focused())


LIST_PANES = ("wezterm", "cli", "list", "--format")
GET_TEXT = ("wezterm", "cli", "get-text")


def screen(row, cursor_x, pane=7, cursor_y=39):
    return FakeRun({
        LIST_PANES: json.dumps([{"pane_id": pane, "cursor_x": cursor_x, "cursor_y": cursor_y}]),
        GET_TEXT: row + "\n",
        LIST_CLIENTS: clients((pane, 0)),
        ACTIVE_WINDOW: "42\n",
    })


class WezTermLineContextTests(unittest.TestCase):
    def test_empty_prompt_has_no_context(self):
        self.assertEqual(wezterm_line_context(7, screen("\u276f\u00a0", 2)), ("", ""))

    def test_prompt_symbol_is_dropped(self):
        self.assertEqual(wezterm_line_context(7, screen("\u276f This is", 9)), ("This is", ""))

    def test_shell_prompt_is_dropped(self):
        run = screen("sander@pcRyzen:~/src$ git commit", 32)
        self.assertEqual(wezterm_line_context(7, run), ("git commit", ""))

    def test_empty_shell_prompt_has_no_context(self):
        self.assertEqual(wezterm_line_context(7, screen("user@host:~$ ", 13)), ("", ""))

    def test_mid_line_splits_at_the_cursor(self):
        run = screen("$ His is a test sentence.            ", 11)
        self.assertEqual(wezterm_line_context(7, run), ("His is a ", "test sentence."))

    def test_wide_characters_take_two_cells(self):
        # "> " is 2 cells, "\u4f60\u597d" 4 more, " x" 2 more: cell 8 is right after "x".
        self.assertEqual(wezterm_line_context(7, screen("> \u4f60\u597d x rest", 8)),
                         ("\u4f60\u597d x", " rest"))

    def test_reads_the_cursor_row(self):
        run = screen("> hi", 4, cursor_y=12)
        wezterm_line_context(7, run)
        cmd = [c for c, _ in run.calls if tuple(c[:3]) == GET_TEXT and "--start-line" in c][0]
        self.assertEqual(cmd[-4:], ["--start-line", "12", "--end-line", "12"])

    def test_unknown_pane_is_none(self):
        self.assertIsNone(wezterm_line_context(8, screen("> hi", 4)))

    def test_target_exposes_line_context(self):
        self.assertEqual(WezTermTarget(7, "42", screen("\u276f fix the", 9)).line_context(),
                         ("fix the", ""))


class XdotoolTests(unittest.TestCase):
    def test_send_backspace_then_type(self):
        run = FakeRun()
        self.assertTrue(XdotoolTarget("42", run).send(Edit(2, "hi")))
        self.assertEqual(run.calls[0][0], ["xdotool", "key", "--clearmodifiers", "--delay", "8",
                                           "--repeat", "2", "BackSpace"])
        self.assertEqual(run.calls[1][0], ["xdotool", "type", "--clearmodifiers", "--delay", "8",
                                           "--", "hi"])

    def test_append_only_skips_backspace(self):
        run = FakeRun()
        XdotoolTarget("42", run).send(Edit(0, "hi"))
        self.assertEqual(len(run.calls), 1)

    def test_failed_backspace_stops_before_typing(self):
        run = FakeRun(returncode=1)
        self.assertFalse(XdotoolTarget("42", run).send(Edit(2, "hi")))
        self.assertEqual(len(run.calls), 1)

    def test_still_focused_compares_window(self):
        run = FakeRun({ACTIVE_WINDOW: "42\n"})
        self.assertTrue(XdotoolTarget("42", run).still_focused())
        self.assertFalse(XdotoolTarget("43", run).still_focused())


class ChooseTargetTests(unittest.TestCase):
    def test_wezterm_class_gives_wezterm_target(self):
        run = FakeRun({LIST_CLIENTS: clients((5, 0)), ACTIVE_WINDOW: "42\n"})
        target = choose_target('WM_CLASS(STRING) = "org.wezfurlong.wezterm", "org.wezfurlong.wezterm"', run)
        self.assertIsInstance(target, WezTermTarget)
        self.assertEqual(target.pane_id, 5)

    def test_other_class_gives_xdotool(self):
        run = FakeRun({ACTIVE_WINDOW: "42\n"})
        self.assertIsInstance(choose_target('WM_CLASS(STRING) = "firefox", "firefox"', run), XdotoolTarget)

    def test_wezterm_unreachable_falls_back_to_xdotool(self):
        run = FakeRun({ACTIVE_WINDOW: "42\n"})
        self.assertIsInstance(choose_target("wezterm", run), XdotoolTarget)

    def test_nothing_focused_is_none(self):
        self.assertIsNone(choose_target(None, FakeRun()))


if __name__ == "__main__":
    unittest.main()
