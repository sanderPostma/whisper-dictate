import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import parse_auto_setting
from text_fixes import lower_case


class ParseAutoSettingTests(unittest.TestCase):
    def test_both_settings(self):
        self.assertEqual(parse_auto_setting("Auto punctuation off."), ("auto_punctuation", False))
        self.assertEqual(parse_auto_setting("Auto capitalization off."), ("auto_capitalization", False))
        self.assertEqual(parse_auto_setting("auto capitalisation on"), ("auto_capitalization", True))
        self.assertEqual(parse_auto_setting("Auto-capitalization of"), ("auto_capitalization", False))
        self.assertEqual(parse_auto_setting("auto caps off"), ("auto_capitalization", False))

    def test_not_a_setting(self):
        self.assertIsNone(parse_auto_setting("turn auto capitalization off please"))


class LowerCaseTests(unittest.TestCase):
    def test_everything_lowered(self):
        self.assertEqual(lower_case("The build failed. I think API and Kotlin work"),
                         "the build failed. i think api and kotlin work")

    def test_jira_keys_kept(self):
        self.assertEqual(lower_case("Look at VDX-283 and EDK-12 First"), "look at VDX-283 and EDK-12 first")
        self.assertEqual(lower_case("Look at videx two eight three"), "look at videx two eight three")


class AppTests(unittest.TestCase):
    def app(self, **config):
        import whisper_dictate
        a = object.__new__(whisper_dictate.WhisperDictate)
        a.config = dict(config)
        a.load_replacements = lambda: {}
        a.save_config = mock.Mock()
        a.notify = mock.Mock()
        return a

    def test_off_lowers_the_first_word(self):
        a = self.app(auto_capitalization=False)
        with mock.patch("builtins.print"):
            self.assertEqual(a.apply_replacements("The starting space. Is gone", strip_trailing_period=False),
                             "the starting space. is gone")

    def test_off_keeps_fixed_ticket_keys(self):
        a = self.app(auto_capitalization=False, ticket_keys=["VDX"])
        with mock.patch("builtins.print"):
            self.assertEqual(a.apply_replacements("Check V D X dash two eight three", strip_trailing_period=False),
                             "check VDX-283")

    def test_off_with_spoken_period(self):
        a = self.app(auto_capitalization=False, auto_punctuation=False)
        with mock.patch("builtins.print"):
            self.assertEqual(a.apply_replacements("Done period Next one", strip_trailing_period=False),
                             "done. next one")

    def test_on_by_default(self):
        with mock.patch("builtins.print"):
            self.assertEqual(self.app().apply_replacements("The build.", strip_trailing_period=False), "The build.")

    def test_toggle_command_itself_untouched(self):
        a = self.app(auto_capitalization=False)
        with mock.patch("builtins.print"):
            self.assertEqual(parse_auto_setting(a.apply_replacements("Auto capitalization on.")),
                             ("auto_capitalization", True))

    def test_set_setting_saves_and_notifies(self):
        a = self.app()
        with mock.patch("builtins.print"):
            a.set_setting("auto_capitalization", False)
        self.assertFalse(a.config["auto_capitalization"])
        a.notify.assert_called_once_with("Auto capitalization off")


class LiveTests(unittest.TestCase):
    def test_spoken_setting_reaches_the_app_and_is_not_typed(self):
        from live_controller import LiveController
        from live_segmenter import ChunkReady
        from live_session import LiveSession

        settings, edits = [], []

        class Target:
            name = "t"

            def send(self, edit):
                edits.append(edit)
                return True

            def still_focused(self):
                return True

        ctl = LiveController(LiveSession(), Target(), lambda a, p: "Auto capitalization off.",
                             log=lambda *_: None, on_setting=lambda key, on: settings.append((key, on)))
        ctl.process(ChunkReady(np.zeros(16000, dtype=np.float32), 0.0, 1.0))
        self.assertEqual((settings, edits), ([("auto_capitalization", False)], []))


if __name__ == "__main__":
    unittest.main()
