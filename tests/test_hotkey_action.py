import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_dictate import hotkey_action


class HotkeyActionTests(unittest.TestCase):
    def test_idle_press_starts_recording(self):
        self.assertEqual(hotkey_action(False, False, False), "start_recording")

    def test_press_while_recording_stops_it(self):
        self.assertEqual(hotkey_action(False, False, True), "stop_recording")

    def test_press_during_live_stops_it(self):
        self.assertEqual(hotkey_action(True, False, False), "stop_live")

    def test_press_while_live_is_finishing_waits(self):
        self.assertEqual(hotkey_action(False, True, False), "busy")


if __name__ == "__main__":
    unittest.main()
