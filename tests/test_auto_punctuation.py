import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import Command, parse_auto_punctuation, parse_command
from live_session import normalise
from text_fixes import manual_punctuation


class ParseAutoPunctuationTests(unittest.TestCase):
    def test_on_and_off(self):
        self.assertIs(parse_auto_punctuation("auto punctuation on"), True)
        self.assertIs(parse_auto_punctuation("Auto punctuation off."), False)

    def test_model_spellings(self):
        self.assertIs(parse_auto_punctuation("Auto-punctuation, off."), False)
        self.assertIs(parse_auto_punctuation("autopunctuation on"), True)
        self.assertIs(parse_auto_punctuation("Automatic punctuation off!"), False)
        self.assertIs(parse_auto_punctuation("auto punctuation of"), False)

    def test_only_as_the_whole_utterance(self):
        self.assertIsNone(parse_auto_punctuation("turn auto punctuation off in the settings"))
        self.assertIsNone(parse_auto_punctuation("auto punctuation"))


class ManualPunctuationTests(unittest.TestCase):
    def test_model_punctuation_removed(self):
        self.assertEqual(manual_punctuation("Okay, I got to go. Then we test."), "Okay I got to go then we test")

    def test_spoken_marks_become_punctuation(self):
        self.assertEqual(manual_punctuation("Okay comma I got to go period then we test"),
                         "Okay, I got to go. Then we test")
        self.assertEqual(manual_punctuation("Is it done question mark"), "Is it done?")
        self.assertEqual(manual_punctuation("Wow exclamation point"), "Wow!")
        self.assertEqual(manual_punctuation("note colon read it semicolon done full stop"),
                         "note: read it; done.")

    def test_model_punctuated_command_words(self):
        self.assertEqual(manual_punctuation("I got to go. Period."), "I got to go.")
        self.assertEqual(manual_punctuation("Comma, then more."), ", then more")

    def test_marks_inside_words_kept(self):
        self.assertEqual(manual_punctuation("Version 3.14 of e.g. VDX-283."), "Version 3.14 of e.g VDX-283")
        self.assertEqual(manual_punctuation("see https://x.io/a?b=1."), "see https://x.io/a?b=1")

    def test_i_and_acronyms_keep_case(self):
        self.assertEqual(manual_punctuation("Done. I think. API works."), "Done I think API works")

    def test_lone_mark(self):
        self.assertEqual(manual_punctuation("Period."), ".")
        self.assertEqual(manual_punctuation("comma"), ",")


class LoneMarkTests(unittest.TestCase):
    def test_lone_end_mark_is_a_join(self):
        self.assertEqual(parse_command("."), Command(end="."))
        self.assertEqual(parse_command("?"), Command(end="?"))

    def test_lone_comma_is_a_comma_repair(self):
        self.assertEqual(parse_command(","), Command(comma=True))

    def test_normalise_glues_a_leading_mark(self):
        self.assertEqual(normalise(". Then more", "I got to go"), ". Then more")
        self.assertEqual(normalise(":", "note", keep_lone_mark=True), ":")

    def test_normalise_still_drops_noise(self):
        self.assertEqual(normalise("...", "note"), "")
        self.assertEqual(normalise(".", "note"), "")  # auto punctuation on: model noise


class AppToggleTests(unittest.TestCase):
    def app(self, **config):
        import whisper_dictate
        a = object.__new__(whisper_dictate.WhisperDictate)
        a.config = {"output_mode": "type", **config}
        a.load_replacements = lambda: {}
        a.save_config = mock.Mock()
        a.notify = mock.Mock()
        return a

    def test_off_strips_in_apply_replacements(self):
        a = self.app(auto_punctuation=False)
        with mock.patch("builtins.print"):
            self.assertEqual(a.apply_replacements("Hello, world. period", strip_trailing_period=False),
                             "Hello world.")

    def test_on_by_default(self):
        a = self.app()
        with mock.patch("builtins.print"):
            self.assertEqual(a.apply_replacements("Hello, world.", strip_trailing_period=False),
                             "Hello, world.")

    def test_off_keeps_a_spoken_final_period_in_one_shot(self):
        a = self.app(auto_punctuation=False)
        with mock.patch("builtins.print"):
            self.assertEqual(a.apply_replacements("Done period"), "Done.")

    def test_the_toggle_itself_is_not_rewritten(self):
        a = self.app(auto_punctuation=False)
        with mock.patch("builtins.print"):
            self.assertIs(parse_auto_punctuation(a.apply_replacements("Auto punctuation on.")), True)

    def test_set_auto_punctuation_saves_and_notifies(self):
        a = self.app()
        with mock.patch("builtins.print"):
            a.set_auto_punctuation(False)
        self.assertFalse(a.config["auto_punctuation"])
        a.save_config.assert_called_once()
        a.notify.assert_called_once_with("Auto punctuation off")


class LiveToggleTests(unittest.TestCase):
    def test_spoken_toggle_is_not_typed(self):
        import numpy as np
        from live_controller import LiveController
        from live_segmenter import ChunkReady
        from live_session import LiveSession

        edits, toggles = [], []

        class Target:
            name = "t"

            def send(self, edit):
                edits.append(edit)
                return True

            def still_focused(self):
                return True

        results = ["Hello there.", "Auto punctuation off."]
        ctl = LiveController(LiveSession(), Target(), lambda audio, prompt: results.pop(0),
                             log=lambda *_: None, on_setting=lambda key, on: toggles.append(on))
        audio = np.zeros(16000, dtype=np.float32)
        ctl.process(ChunkReady(audio, 0.0, 1.0))
        ctl.process(ChunkReady(audio, 1.0, 2.0))
        self.assertEqual(toggles, [False])
        self.assertEqual(len(edits), 1)
        self.assertEqual(ctl.session.chunk_count, 0)  # committed


if __name__ == "__main__":
    unittest.main()
