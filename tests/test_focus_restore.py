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
        app.desktop = mock.Mock()
        return app

    def test_window_that_still_has_focus_is_left_alone(self):
        # Re-activating it while the hotkey's Alt is held makes Electron
        # apps (Slack, VS Code) open their menu bar.
        a = self.app("42")
        a.restore_focus("42")
        a.desktop.activate.assert_not_called()

    def test_focus_that_moved_away_is_restored(self):
        a = self.app("99")
        a.restore_focus("42")
        a.desktop.activate.assert_called_once_with("42")

    def test_no_saved_window_does_nothing(self):
        a = self.app("42")
        a.restore_focus(None)
        a.desktop.activate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
