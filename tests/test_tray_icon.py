import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import whisper_dictate
from whisper_dictate import WhisperDictate


class TrayIconTests(unittest.TestCase):
    def app(self):
        a = object.__new__(WhisperDictate)
        a.icons = []
        a._set_icon = a.icons.append
        return a

    def test_live_dictation_is_steady_red(self):
        a = self.app()
        with mock.patch.object(whisper_dictate.GLib, "timeout_add") as timer:
            a.update_icon(True, steady=True)
        self.assertEqual(a.icons, ["mic-recording"])
        timer.assert_not_called()

    def test_one_shot_blinks(self):
        a = self.app()
        with mock.patch.object(whisper_dictate.GLib, "timeout_add", return_value=7) as timer:
            a.update_icon(True)
        timer.assert_called_once()

    def test_switch_from_blinking_to_steady_stops_the_blink(self):
        a = self.app()
        a._blink_timer_id = 7
        with mock.patch.object(whisper_dictate.GLib, "source_remove") as remove:
            a.update_icon(True, steady=True)
        remove.assert_called_once_with(7)
        self.assertEqual(a.icons[-1], "mic-recording")


if __name__ == "__main__":
    unittest.main()
