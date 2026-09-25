import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_fixes import DEFAULT_SHELL_ALIASES, DEFAULT_SHELL_COMMANDS, fix_shell_command


def fix(text):
    return fix_shell_command(text, DEFAULT_SHELL_COMMANDS, DEFAULT_SHELL_ALIASES)


class ShellCommandTests(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(fix("Alas."), "ls")
        self.assertEqual(fix("L s minus L."), "ls -l")
        self.assertEqual(fix("Git branch."), "git branch")
        self.assertEqual(fix("Sudo."), "sudo")

    def test_spelled_and_model_forms(self):
        self.assertEqual(fix("LS."), "ls")
        self.assertEqual(fix("L.S. -L"), "ls -l")
        self.assertEqual(fix("C D, docs."), "cd docs")
        self.assertEqual(fix("Pseudo apt update."), "sudo apt update")

    def test_flags(self):
        self.assertEqual(fix("Git push dash dash force."), "git push --force")
        self.assertEqual(fix("Git push double dash force"), "git push --force")
        self.assertEqual(fix("ls minus L A"), "ls -l a")
        self.assertEqual(fix("Git log - - oneline."), "git log --oneline")

    def test_ordinary_sentences_untouched(self):
        self.assertIsNone(fix("Alas, it failed."))
        self.assertIsNone(fix("Let's check git branch."))
        self.assertIsNone(fix("Sudoku is fun."))
        self.assertIsNone(fix(""))

    def test_alias_needs_to_be_the_whole_word(self):
        self.assertIsNone(fix("Alaska is big."))


class ApplyReplacementsTests(unittest.TestCase):
    def test_shell_command_skips_punctuation_steps(self):
        import whisper_dictate
        a = object.__new__(whisper_dictate.WhisperDictate)
        a.config = {"auto_punctuation": False}
        a.load_replacements = lambda: {}
        with mock.patch("builtins.print"):
            self.assertEqual(a.apply_replacements("Git branch period"), "git branch period")
            self.assertEqual(a.apply_replacements("L s minus L."), "ls -l")


class LiveVerbatimTests(unittest.TestCase):
    def test_command_line_closes_the_correction_window(self):
        import numpy as np
        from live_controller import LiveController
        from live_segmenter import ChunkReady
        from live_session import LiveSession

        class Target:
            name = "t"

            def send(self, edit):
                return True

            def still_focused(self):
                return True

        ctl = LiveController(LiveSession(), Target(), lambda audio, prompt: "git branch",
                             log=lambda *_: None, is_verbatim=lambda text: fix(text) is not None)
        ctl.process(ChunkReady(np.zeros(16000, dtype=np.float32), 0.0, 1.0))
        self.assertEqual(ctl.session.chunk_count, 0)


if __name__ == "__main__":
    unittest.main()
