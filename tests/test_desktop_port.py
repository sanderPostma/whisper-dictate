import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import whisper_dictate
from desktop import GnomeWaylandDesktop
from whisper_dictate import WhisperDictate

MISSING = "✗ GNOME extension whisper-dictate not active — run ./install-gnome-extension.sh and log out/in"


class FakeDesktop:
    def __init__(self, available=True, active="42"):
        self._available = available
        self.active = active
        self.activated = []
        self.bound = {}
        self.keys = []
        self.events = []

    def available(self):
        return self._available

    def active_window(self):
        return self.active

    def activate(self, window_id):
        self.activated.append(window_id)
        return True

    def bind_hotkey(self, accel, callback):
        self.bound[accel] = callback
        return True

    def unbind_all(self):
        self.bound = {}

    def key(self, chord, repeat=1, delay_ms=None):
        self.events.append(("key", chord))
        return True


class FakeWayland(FakeDesktop, GnomeWaylandDesktop):
    def __init__(self, **kw):
        FakeDesktop.__init__(self, **kw)


def app(desktop):
    a = object.__new__(WhisperDictate)
    a.config = {"output_mode": "type", "hotkey": "<Alt>d", "live_hotkey": "<Alt>l"}
    a.desktop = desktop
    a.update_status = lambda s: None
    return a


class DesktopPortTests(unittest.TestCase):
    def test_restore_focus_uses_desktop_activate(self):
        d = FakeDesktop(active="99")
        app(d).restore_focus("42")
        self.assertEqual(d.activated, ["42"])
        d2 = FakeDesktop(active="42")
        app(d2).restore_focus("42")
        self.assertEqual(d2.activated, [])

    def test_hotkeys_bound_through_desktop(self):
        d = FakeDesktop()
        a = app(d)
        a.toggle_recording = mock.Mock()
        a.toggle_live = mock.Mock()
        a._bind_hotkeys()
        self.assertEqual(set(d.bound), {"<Alt>d", "<Alt>l"})
        d.bound["<Alt>d"]()
        a.toggle_recording.assert_called_once()
        d.bound["<Alt>l"]()
        a.toggle_live.assert_called_once()

    def test_wayland_enter_leaves_modifier_wait_to_the_desktop(self):
        d = FakeWayland()
        a = app(d)
        with mock.patch.object(whisper_dictate.time, "sleep",
                               side_effect=lambda s: d.events.append(("sleep", s))), \
                mock.patch("builtins.print"):
            a.output_text("", press_enter=True)
        self.assertEqual(d.events, [("key", "Return")])

    def test_missing_extension_warns_and_keeps_running(self):
        a = app(FakeDesktop(available=False))
        a.notify = mock.Mock()
        with mock.patch("builtins.print") as pr:
            a._check_desktop()
        pr.assert_any_call(MISSING)
        a.notify.assert_called_once_with(MISSING)

    def test_available_desktop_is_silent(self):
        a = app(FakeDesktop())
        a.notify = mock.Mock()
        a._check_desktop()
        a.notify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
