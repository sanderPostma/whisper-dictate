import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_output import claude_input_before, wezterm_line_context

RULE = "\x1b[38:2::136:136:136m" + "─" * 60 + "\x1b[39m"
GRAY = "\x1b[38:2::153:153:153m"


def box(*rows, above="⏺ Done, pushed.", below="  \x1b[33mOpus\x1b[39m status"):
    return "\n".join([above, "", RULE, *rows, RULE, below, ""])


class ClaudeInputBeforeTests(unittest.TestCase):
    def test_empty_box(self):
        self.assertEqual(claude_input_before(box("\x1b[39m❯ "), 0, 0), "")

    def test_placeholder_is_not_text(self):
        screen = box("❯ " + GRAY + "Press up to edit queued messages\x1b[39m")
        self.assertEqual(claude_input_before(screen, 0, 0), "")

    def test_typed_text_cursor_elsewhere(self):
        self.assertEqual(claude_input_before(box("❯ Fix the login"), 0, 0), "Fix the login")

    def test_wrapped_rows_are_joined(self):
        screen = box("❯ The starting space at the", "  beginning of each line")
        self.assertEqual(claude_input_before(screen, 0, 0), "The starting space at the beginning of each line")

    def test_cursor_inside_the_box_splits_there(self):
        screen = box("❯ His is a test")
        # rows: 0 above, 1 blank, 2 rule, 3 input; "❯ His is a " is 11 cells
        self.assertEqual(claude_input_before(screen, 11, 3), "His is a ")

    def test_past_prompts_in_the_transcript_are_ignored(self):
        screen = "❯ earlier message\n\n" + box("❯ ")
        self.assertEqual(claude_input_before(screen, 0, 0), "")

    def test_no_box(self):
        self.assertIsNone(claude_input_before("$ ls\nfile\n$ ", 2, 2))


class LineContextUsesTheBoxTests(unittest.TestCase):
    def test_box_wins_over_a_wrong_cursor_row(self):
        screen = box("❯ ")
        pane = json.dumps([{"pane_id": 7, "cursor_x": 5, "cursor_y": 0}])

        def run(cmd, **kwargs):
            class R:
                returncode = 0
                stdout = pane if cmd[2] == "list" else screen
            return R()

        self.assertEqual(wezterm_line_context(7, run), ("", ""))


class EchoWordBoundaryTests(unittest.TestCase):
    def test_partial_word_is_not_an_echo(self):
        from live_session import strip_echo
        self.assertEqual(strip_echo("Auto capitalization off.", "retry auto capitalization of"),
                         "Auto capitalization off.")
        self.assertEqual(strip_echo("auto capitalization of more", "retry auto capitalization of"), "more")


class FreshStartTests(unittest.TestCase):
    def test_long_pause_rereads_the_line(self):
        import numpy as np
        from live_controller import LiveController
        from live_segmenter import ChunkReady, LongPause
        from live_session import Edit, LiveSession

        lines = iter([("", ""), ("", "")])  # Enter pressed by hand during the pause

        class Target:
            name = "t"
            edits = []

            def send(self, edit):
                self.edits.append(edit)
                return True

            def still_focused(self):
                return True

            def line_context(self):
                return next(lines)

        results = ["Fix the login", "Next message"]
        target = Target()
        ctl = LiveController(LiveSession(), target, lambda a, p: results.pop(0), log=lambda *_: None)
        audio = np.zeros(16000, dtype=np.float32)
        ctl.process(ChunkReady(audio, 0.0, 1.0))
        ctl.process(LongPause(3.0))
        ctl.process(ChunkReady(audio, 3.0, 4.0))
        self.assertEqual(target.edits[-1], Edit(0, "Next message"))

    def test_new_window_without_a_readable_line_starts_fresh(self):
        import numpy as np
        from live_controller import LiveController
        from live_segmenter import ChunkReady
        from live_session import Edit, LiveSession

        class Target:
            name = "t"

            def __init__(self):
                self.edits, self.focused = [], True

            def send(self, edit):
                self.edits.append(edit)
                return True

            def still_focused(self):
                return self.focused

        old, new = Target(), Target()
        results = ["Hello there", "Next thing"]
        ctl = LiveController(LiveSession(), old, lambda a, p: results.pop(0), make_target=lambda: new,
                             log=lambda *_: None)
        audio = np.zeros(16000, dtype=np.float32)
        ctl.process(ChunkReady(audio, 0.0, 1.0))
        old.focused = False
        ctl.process(ChunkReady(audio, 1.0, 2.0))
        self.assertEqual(new.edits, [Edit(0, "Next thing")])


if __name__ == "__main__":
    unittest.main()
