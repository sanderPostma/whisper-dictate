import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import whisper_dictate
from whisper_dictate import WhisperDictate


class FakeLine:
    def __init__(self, context):
        self.context = context

    def line_context(self):
        return self.context


def app(**config):
    a = object.__new__(WhisperDictate)
    a.config = {"output_mode": "type", **config}
    a.get_asr_context = lambda: "Vocabulary: kubectl"
    a.update_status = lambda status: None
    a.get_focused_window_class = lambda: "org.wezfurlong.wezterm"
    return a


class OneShotPromptTests(unittest.TestCase):
    def test_text_before_cursor_joins_the_asr_context(self):
        prompt = app()._oneshot_prompt(FakeLine(("This is", " rest")))
        self.assertEqual(prompt, "Vocabulary: kubectl\nThis is")

    def test_no_achat_session_keeps_asr_context(self):
        self.assertEqual(app()._oneshot_prompt(None), "Vocabulary: kubectl")

    def test_unknown_line_keeps_asr_context(self):
        self.assertEqual(app()._oneshot_prompt(FakeLine(None)), "Vocabulary: kubectl")


class OneShotOutputTests(unittest.TestCase):
    def test_achat_line_is_used_instead_of_keystrokes(self):
        with mock.patch.object(whisper_dictate, "type_into_line", return_value=True) as typed, \
                mock.patch.object(whisper_dictate.subprocess, "run") as run:
            app().output_text("The end", achat_target=object())
        typed.assert_called_once()
        run.assert_not_called()

    def test_falls_back_to_keystrokes_when_line_unusable(self):
        with mock.patch.object(whisper_dictate, "type_into_line", return_value=False), \
                mock.patch.object(whisper_dictate.subprocess, "run") as run, \
                mock.patch.object(whisper_dictate.time, "sleep"):
            app().output_text("The end", achat_target=object())
        self.assertEqual(run.call_args[0][0][:2], ["xdotool", "type"])

    def test_single_word_lowercasing_left_to_the_line(self):
        a = app()
        a.load_replacements = lambda: {"zzqx": "q"}
        self.assertEqual(a.apply_replacements("API", lower_single_word=False), "API")
        self.assertEqual(a.apply_replacements("API"), "api")


if __name__ == "__main__":
    unittest.main()
