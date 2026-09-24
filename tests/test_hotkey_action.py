import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_dictate import hotkey_action


class HotkeyActionTests(unittest.TestCase):
    def act(self, live_active=False, live_busy=False, recording=False, since_start=None):
        return hotkey_action(live_active, live_busy, recording, since_start, double_press_s=1.0)

    def test_idle_press_starts_recording(self):
        self.assertEqual(self.act(), "start_recording")

    def test_second_press_within_a_second_switches_to_live(self):
        self.assertEqual(self.act(recording=True, since_start=0.4), "to_live")

    def test_later_press_stops_the_recording(self):
        self.assertEqual(self.act(recording=True, since_start=3.0), "stop_recording")

    def test_press_during_live_stops_it(self):
        self.assertEqual(self.act(live_active=True), "stop_live")

    def test_press_while_live_is_finishing_waits(self):
        self.assertEqual(self.act(live_busy=True), "busy")


if __name__ == "__main__":
    unittest.main()
