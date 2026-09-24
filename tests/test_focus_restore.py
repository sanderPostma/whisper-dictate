import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import whisper_dictate
from whisper_dictate import WhisperDictate


class RestoreFocusTests(unittest.TestCase):
    def app(self, active):
        app = object.__new__(WhisperDictate)
        app.get_focused_window = lambda: active
        return app

    def test_window_that_still_has_focus_is_left_alone(self):
        # Re-activating it while the hotkey's Alt is held makes Electron
        # apps (Slack, VS Code) open their menu bar.
        with mock.patch.object(whisper_dictate.subprocess, "run") as run:
            self.app("42").restore_focus("42")
        run.assert_not_called()

    def test_focus_that_moved_away_is_restored(self):
        with mock.patch.object(whisper_dictate.subprocess, "run") as run:
            self.app("99").restore_focus("42")
        self.assertEqual(run.call_args[0][0], ["xdotool", "windowactivate", "42"])

    def test_no_saved_window_does_nothing(self):
        with mock.patch.object(whisper_dictate.subprocess, "run") as run:
            self.app("42").restore_focus(None)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
