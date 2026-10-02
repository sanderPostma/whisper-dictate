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


class ActionMarkTests(unittest.TestCase):
    """A 'v' on the hot action, because the tray icon often stays stale."""

    def labels(self, **flags):
        a = object.__new__(WhisperDictate)
        a.recording = flags.get("recording", False)
        a.transcribe_active = flags.get("transcribe_active", False)
        a.live_active = flags.get("live_active", False)
        a.live_busy = flags.get("live_busy", False)
        return a.action_menu_labels()

    def test_idle_has_no_mark(self):
        labels = self.labels()
        self.assertEqual(labels["record"], "🎤 Record/Stop")
        self.assertEqual(labels["transcribe"], "🎙️ Transcribe...")
        self.assertEqual(labels["live"], "⚡ Live dictation")
        self.assertFalse(any(text.startswith("v ") for text in labels.values()))

    def test_recording_marks_record(self):
        labels = self.labels(recording=True)
        self.assertEqual(labels["record"], "v 🎤 Record/Stop")
        self.assertFalse(labels["live"].startswith("v "))
        self.assertFalse(labels["transcribe"].startswith("v "))

    def test_live_and_finishing_mark_live(self):
        self.assertEqual(self.labels(live_active=True)["live"], "v ⚡ Live dictation")
        # Icon stays red until the controller finishes; the mark stays with it.
        labels = self.labels(live_busy=True)
        self.assertEqual(labels["live"], "v ⚡ Live dictation")
        self.assertFalse(labels["record"].startswith("v "))

    def test_transcribe_session_marks_its_stop_label(self):
        labels = self.labels(transcribe_active=True)
        self.assertEqual(labels["transcribe"], "v 🛑 Stop Transcribe")
        self.assertFalse(labels["record"].startswith("v "))
        self.assertFalse(labels["live"].startswith("v "))


if __name__ == "__main__":
    unittest.main()
