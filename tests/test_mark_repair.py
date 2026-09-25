import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import Command
from live_controller import LiveController
from live_output import WezTermTarget
from live_segmenter import ChunkReady
from live_session import Edit, LiveSession


def chunk(t0, t1):
    return ChunkReady(np.zeros(int(16000 * (t1 - t0)), dtype=np.float32), t0, t1)


class ReplaceMarkTests(unittest.TestCase):
    def repair(self, typed, command):
        s = LiveSession()
        s.add_chunk(np.zeros(16000, dtype=np.float32), 0.0, 1.0)
        s.add_fast_result(1.0, typed)
        return s.apply_command(command)

    def test_question_mark_replaces_the_full_stop(self):
        self.assertEqual(self.repair("Can you fix that.", Command(end="?")), Edit(1, "?"))

    def test_question_mark_replaces_a_comma_or_colon(self):
        self.assertEqual(self.repair("Can you fix that,", Command(end="?")), Edit(1, "?"))
        self.assertEqual(self.repair("Can you fix that:", Command(end="?")), Edit(1, "?"))

    def test_question_mark_added_when_there_is_none(self):
        self.assertEqual(self.repair("can you fix that", Command(end="?")), Edit(0, "?"))

    def test_comma_replaces_a_semicolon(self):
        self.assertEqual(self.repair("first part;", Command(comma=True)), Edit(1, ","))


class ScreenTarget:
    """A WezTerm-like target whose cursor row is not the input row."""
    name = "wezterm"
    exact_line = False

    def __init__(self, screen):
        self.screen = screen
        self.edits = []

    def send(self, edit):
        self.edits.append(edit)
        return True

    def still_focused(self):
        return True

    def line_context(self):
        return ("Press up to edit queued messages", "")

    def shows_line_end(self, text, tail_chars=24):
        return WezTermTarget.line_end_in(self.screen, text, tail_chars=tail_chars)


class ScreenFallbackTests(unittest.TestCase):
    def run_chunks(self, screen, *results):
        results = list(results)
        target = ScreenTarget(screen)
        ctl = LiveController(LiveSession(), target, lambda audio, prompt: results.pop(0),
                             log=lambda *_: None)
        for i in range(len(results)):
            ctl.process(chunk(float(i), i + 1.0))
        return target.edits

    def test_mark_repair_accepted_when_the_screen_shows_the_text(self):
        edits = self.run_chunks("> can you fix that.\n\n  status bar\n", "Can you fix that.", "Question mark.")
        self.assertEqual(edits[-1], Edit(1, "?"))

    def test_mark_repair_refused_when_the_text_is_gone(self):
        edits = self.run_chunks("> \n\n  status bar\n", "Can you fix that.", "Question mark.")
        self.assertEqual(len(edits), 1)

    def test_scratch_allowed_when_the_screen_shows_the_text(self):
        # Claude Code redraws while an agent works: the cursor row is often another row.
        edits = self.run_chunks("> can you fix that.\n", "Can you fix that.", "Scratch that.")
        self.assertEqual(edits[-1], Edit(18, ""))

    def test_scratch_needs_a_long_enough_tail_at_a_row_end(self):
        typed = "We will first insert a space when not at the beginning of the sentence"
        edits = self.run_chunks("> " + typed[-30:] + "\n", typed, "Scratch that.")
        self.assertEqual(len(edits), 1)

    def test_failed_undo_never_types_the_command(self):
        target = ScreenTarget("")
        results = ["Command undo.", "Command undo."]
        ctl = LiveController(LiveSession(), target, lambda audio, prompt: results.pop(0),
                             lambda audio, prompt: results.pop(0), log=lambda *_: None)
        ctl.process(chunk(0.0, 1.0))
        ctl.correct()
        self.assertEqual(target.edits, [])


class PasteSpaceTests(unittest.TestCase):
    def run_paste(self, *typed):
        target = ScreenTarget("")
        target.clipboard = lambda action: target.edits.append(action) or True
        results = [*typed, "Command paste."]
        ctl = LiveController(LiveSession(), target, lambda audio, prompt: results.pop(0),
                             log=lambda *_: None)
        for i in range(len(typed) + 1):
            ctl.process(chunk(float(i), i + 1.0))
        return target.edits

    def test_space_before_a_paste_after_text(self):
        self.assertEqual(self.run_paste("See this.")[-2:], [Edit(0, " "), "paste"])

    def test_no_space_after_a_space(self):
        target = ScreenTarget("")
        target.line_context = lambda: ("See this ", "")
        target.clipboard = lambda action: target.edits.append(action) or True
        ctl = LiveController(LiveSession(), target, lambda audio, prompt: "Command paste.",
                             log=lambda *_: None)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(target.edits, ["paste"])


class LineEndInTests(unittest.TestCase):
    def test_text_at_the_end_of_a_row(self):
        self.assertTrue(WezTermTarget.line_end_in("x\n❯ When I say can you fix that   \nfoot", "fix that"))

    def test_text_mid_row_does_not_count(self):
        self.assertFalse(WezTermTarget.line_end_in("❯ can you fix that thing\n", "can you fix that"))

    def test_tiny_text_does_not_count(self):
        self.assertFalse(WezTermTarget.line_end_in("❯ ok\n", "ok"))


class SlashCommandTests(unittest.TestCase):
    def test_no_leading_space_before_a_slash_command(self):
        from live_session import normalise
        self.assertEqual(normalise("/clear.", "Press up to edit queued messages"), "/clear")
        self.assertEqual(normalise("/compact", "some text"), "/compact")

    def test_slash_command_closes_the_correction_window(self):
        target = ScreenTarget("")
        ctl = LiveController(LiveSession(), target, lambda audio, prompt: "/clear",
                             log=lambda *_: None)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(ctl.session.chunk_count, 0)
        self.assertEqual(target.edits, [Edit(0, "/clear")])


if __name__ == "__main__":
    unittest.main()
