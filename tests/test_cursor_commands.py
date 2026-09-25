import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import CursorMove, parse_cursor, word_target


class ParseCursorTests(unittest.TestCase):
    def test_back_and_forward_one_word(self):
        self.assertEqual(parse_cursor("Cursor back."), CursorMove("word", -1, 1))
        self.assertEqual(parse_cursor("cursor forward"), CursorMove("word", 1, 1))

    def test_counts_as_digits_or_words(self):
        self.assertEqual(parse_cursor("Cursor back 3 words."), CursorMove("word", -1, 3))
        self.assertEqual(parse_cursor("Cursor back three words."), CursorMove("word", -1, 3))
        self.assertEqual(parse_cursor("cursor forward, twelve words"), CursorMove("word", 1, 12))
        self.assertEqual(parse_cursor("Cursor forward one word."), CursorMove("word", 1, 1))
        self.assertEqual(parse_cursor("Cursor back two."), CursorMove("word", -1, 2))

    def test_asr_spelling_of_two(self):
        self.assertEqual(parse_cursor("Cursor back to words."), CursorMove("word", -1, 2))

    def test_start_and_end(self):
        self.assertEqual(parse_cursor("Cursor to start."), CursorMove("start"))
        self.assertEqual(parse_cursor("Cursor to the start of the line"), CursorMove("start"))
        self.assertEqual(parse_cursor("Cursor to the beginning."), CursorMove("start"))
        self.assertEqual(parse_cursor("Cursor to end."), CursorMove("end"))
        self.assertEqual(parse_cursor("cursor to the end"), CursorMove("end"))

    def test_only_as_the_whole_utterance(self):
        for text in ("Move the cursor back three words.", "The cursor back button", "cursor", "Cursor to Paris."):
            self.assertIsNone(parse_cursor(text), text)


class WordTargetTests(unittest.TestCase):
    TEXT = "fix the brown fox now"

    def test_back_words(self):
        self.assertEqual(word_target(self.TEXT, 21, CursorMove("word", -1, 1)), 18)  # |now
        self.assertEqual(word_target(self.TEXT, 21, CursorMove("word", -1, 3)), 8)   # |brown
        self.assertEqual(word_target(self.TEXT, 21, CursorMove("word", -1, 9)), 0)

    def test_back_from_inside_a_word_goes_to_its_start(self):
        self.assertEqual(word_target(self.TEXT, 10, CursorMove("word", -1, 1)), 8)

    def test_forward_words(self):
        self.assertEqual(word_target(self.TEXT, 0, CursorMove("word", 1, 1)), 3)    # fix|
        self.assertEqual(word_target(self.TEXT, 3, CursorMove("word", 1, 2)), 13)   # brown|
        self.assertEqual(word_target(self.TEXT, 0, CursorMove("word", 1, 99)), 21)

    def test_start_and_end(self):
        self.assertEqual(word_target(self.TEXT, 9, CursorMove("start")), 0)
        self.assertEqual(word_target(self.TEXT, 9, CursorMove("end")), 21)


class Run:
    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        out = next((v for k, v in self.answers.items() if tuple(cmd[:len(k)]) == k), "")
        return SimpleNamespace(returncode=0, stdout=out)


class TargetMoveTests(unittest.TestCase):
    def test_achat_moves_with_arrows_it_can_follow(self):
        from live_achat import AchatTarget
        run = Run()
        state = {"ok": True, "rev": 4, "known": True, "text": "fix the brown fox now", "cursor": 21}
        t = AchatTarget(7, "42", "/x.sock", 4, run=run, request=lambda s, p: state,
                        sleep=lambda s: None, log=lambda *_: None)
        self.assertTrue(t.move_cursor(CursorMove("word", -1, 3)))
        cmd, kwargs = run.calls[-1]
        self.assertEqual(cmd[:6], ["wezterm", "cli", "send-text", "--pane-id", "7", "--no-paste"])
        self.assertEqual(kwargs["input"], b"\x1b[D" * 13)

    def test_achat_unknown_draft_moves_with_readline_keys(self):
        from live_achat import AchatTarget
        run = Run()
        t = AchatTarget(7, "42", "/x.sock", 4, run=run,
                        request=lambda s, p: {"ok": True, "rev": 4, "known": False, "text": "", "cursor": 0},
                        sleep=lambda s: None, log=lambda *_: None)
        # An Unknown (multi-line) draft: readline keys on the pane, like plain WezTerm.
        self.assertTrue(t.move_cursor(CursorMove("end")))
        self.assertEqual(run.calls[-1][1]["input"], b"\x05")

    def test_wezterm_uses_readline_keys(self):
        from live_output import WezTermTarget
        run = Run()
        WezTermTarget(7, "42", run).move_cursor(CursorMove("word", 1, 2))
        self.assertEqual(run.calls[-1][1]["input"], b"\x1bf\x1bf")
        WezTermTarget(7, "42", run).move_cursor(CursorMove("start"))
        self.assertEqual(run.calls[-1][1]["input"], b"\x01")

    def test_xdotool_uses_editor_keys(self):
        from live_output import XdotoolTarget
        run = Run()
        XdotoolTarget("42", run).move_cursor(CursorMove("word", -1, 3))
        self.assertEqual(run.calls[-1][0], ["xdotool", "key", "--clearmodifiers", "--repeat", "3", "ctrl+Left"])
        XdotoolTarget("42", run).move_cursor(CursorMove("end"))
        self.assertEqual(run.calls[-1][0], ["xdotool", "key", "--clearmodifiers", "End"])


class ClearLineTests(unittest.TestCase):
    def test_achat_moves_to_the_end_then_deletes_exactly_the_known_line(self):
        from live_achat import AchatTarget
        run = Run()
        states = [
            {"ok": True, "rev": 4, "known": True, "text": "fix the bug", "cursor": 4},
            {"ok": True, "rev": 11, "known": True, "text": "fix the bug", "cursor": 11},
        ]
        requests = []

        def request(sock, payload):
            requests.append(payload)
            if payload["op"] == "state":
                return states.pop(0)
            return {"ok": True, "rev": 22}

        t = AchatTarget(7, "42", "/x.sock", 4, run=run, request=request,
                        sleep=lambda s: None, log=lambda *_: None)
        self.assertTrue(t.clear_line())
        self.assertEqual(run.calls[-1][1]["input"], b"\x1b[C" * 7)
        self.assertEqual(requests[-1], {"op": "edit", "expect_rev": 11, "backspace": 11,
                                        "insert": "", "at_cursor": True})

    def test_achat_unknown_draft_clears_with_readline_keys(self):
        from live_achat import AchatTarget
        run = Run()
        t = AchatTarget(7, "42", "/x.sock", 4, run=run,
                        request=lambda s, p: {"ok": True, "rev": 4, "known": False, "text": "", "cursor": 0},
                        sleep=lambda s: None, log=lambda *_: None)
        self.assertTrue(t.clear_line())
        self.assertEqual(run.calls[-1][1]["input"], b"\x05\x15")

    def test_wezterm_end_then_kill_to_start(self):
        from live_output import WezTermTarget
        run = Run()
        WezTermTarget(7, "42", run).clear_line()
        self.assertEqual(run.calls[-1][1]["input"], b"\x05\x15")

    def test_xdotool_has_no_clear(self):
        from live_output import XdotoolTarget
        self.assertFalse(hasattr(XdotoolTarget("42", Run()), "clear_line"))


if __name__ == "__main__":
    unittest.main()
