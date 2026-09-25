import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import parse_clear, parse_undo
from text_fixes import DEFAULT_SLASH_COMMANDS, fix_slash_command


class SlashCommandFixTests(unittest.TestCase):
    def fix(self, text):
        return fix_slash_command(text, DEFAULT_SLASH_COMMANDS)

    def test_misheard_slash_before_a_known_command(self):
        self.assertEqual(self.fix("Flash clear."), "/clear")
        self.assertEqual(self.fix("Less clear."), "/clear")
        self.assertEqual(self.fix("Splash compact"), "/compact")
        self.assertEqual(self.fix("Slash new."), "/new")

    def test_slash_word_at_the_start_before_any_word(self):
        self.assertEqual(self.fix("Dash compact."), "/compact")
        self.assertEqual(self.fix("Flash review the branch."), "/review the branch.")
        self.assertEqual(self.fix("Dash, runbook."), "/runbook")
        self.assertEqual(self.fix("Slash Rumbuk."), "/rumbuk")

    def test_split_known_command_joined(self):
        self.assertEqual(self.fix("Slash run book."), "/runbook")
        self.assertEqual(self.fix("Dash run book for the deploy"), "/runbook for the deploy")
        self.assertEqual(self.fix("Slash run the tests"), "/run the tests")

    def test_model_dash_before_a_known_command(self):
        self.assertEqual(self.fix("-compact."), "/compact")
        self.assertEqual(self.fix("- clear"), "/clear")
        self.assertEqual(self.fix("- item one"), "- item one")

    def test_other_text_untouched(self):
        self.assertEqual(self.fix("Less clear than before."), "Less clear than before.")
        self.assertEqual(self.fix("a splash command"), "a splash command")
        self.assertEqual(self.fix("VDX dash 283"), "VDX dash 283")
        self.assertEqual(self.fix("Dash."), "Dash.")


class MisheardCommandWordTests(unittest.TestCase):
    def test_undo_variants(self):
        for text in ("Comment. Undo.", "Comment undo", "Undo.", "Commend, undo."):
            self.assertTrue(parse_undo(text), text)
        self.assertFalse(parse_undo("Undo the last change."))

    def test_clear_variants(self):
        self.assertTrue(parse_clear("Comment clear line."))
        self.assertTrue(parse_clear("Comment. Clear."))


class VocabularyPromptTests(unittest.TestCase):
    def app(self, **config):
        import whisper_dictate
        a = object.__new__(whisper_dictate.WhisperDictate)
        a.config = {"context_pack": "none", **config}
        return a

    def test_command_words_always_in_the_prompt(self):
        prompt = self.app().get_asr_context()
        self.assertIn("slash", prompt)
        self.assertIn("command undo", prompt)

    def test_configured_words_and_ticket_keys_added(self):
        prompt = self.app(vocabulary=["Sphereon"], ticket_keys=["VDX"]).get_asr_context()
        self.assertIn("Sphereon", prompt)
        self.assertIn("VDX", prompt)

    def test_an_echoed_vocabulary_line_is_dropped(self):
        a = self.app()
        a.load_replacements = lambda: {}
        with mock.patch("builtins.print"):
            self.assertEqual(a.apply_replacements("Vocabulary: slash, command, command undo."), "")


if __name__ == "__main__":
    unittest.main()
