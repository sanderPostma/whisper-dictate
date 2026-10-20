import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import desktop
from desktop import GnomeWaylandDesktop, X11Desktop, get_desktop
from gi.repository import GLib


class FakeRun:
    def __init__(self, answers=None, returncode=0):
        self.calls = []
        self.answers = answers or {}
        self.returncode = returncode

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        out = b""
        rc = self.returncode
        for prefix, ans in self.answers.items():
            if tuple(cmd[:len(prefix)]) == prefix:
                if isinstance(ans, int):
                    rc = ans
                else:
                    out = ans
        return SimpleNamespace(returncode=rc, stdout=out)


class FakeCall:
    def __init__(self, answers=None, error=False):
        self.calls = []
        self.answers = answers or {}
        self.error = error

    def __call__(self, method, args=None):
        self.calls.append((method, args))
        if self.error:
            raise GLib.Error("no such object")
        return self.answers.get(method, (True,))


def now(fn):
    fn()


class X11Tests(unittest.TestCase):
    def test_x11_key_matches_xdotool_argv(self):
        run = FakeRun()
        self.assertTrue(X11Desktop(run).key("BackSpace", repeat=3, delay_ms=8))
        self.assertEqual(run.calls[0][0], ["xdotool", "key", "--clearmodifiers", "--delay", "8",
                                           "--repeat", "3", "BackSpace"])

    def test_x11_type_text_argv(self):
        run = FakeRun()
        X11Desktop(run).type_text("hi", delay_ms=25)
        self.assertEqual(run.calls[0][0], ["xdotool", "type", "--clearmodifiers", "--delay", "25",
                                           "--", "hi"])

    def test_x11_type_text_timeout_scales_with_text(self):
        run = FakeRun()
        X11Desktop(run).type_text("x" * 5000, delay_ms=25)
        self.assertGreater(run.calls[0][1]["timeout"], 5000 * 0.025)
        run = FakeRun()
        X11Desktop(run).type_text("hi", delay_ms=25)
        self.assertGreaterEqual(run.calls[0][1]["timeout"], 10)

    def test_x11_activate_and_clipboard_argv(self):
        run = FakeRun({("xclip", "-o"): b"clip"})
        d = X11Desktop(run)
        d.activate("42")
        d.clipboard_set(b"x")
        self.assertEqual(d.clipboard_get(), b"clip")
        cmds = [c for c, _ in run.calls]
        self.assertEqual(cmds[0], ["xdotool", "windowactivate", "42"])
        self.assertEqual(cmds[1], ["xclip", "-selection", "clipboard"])
        self.assertEqual(cmds[2], ["xclip", "-o", "-selection", "clipboard"])

    def test_x11_window_class_is_raw_xprop_line(self):
        run = FakeRun({("xdotool", "getactivewindow"): "42\n",
                       ("xprop",): 'WM_CLASS(STRING) = "kitty", "kitty"\n'})
        self.assertEqual(X11Desktop(run).window_class(), 'WM_CLASS(STRING) = "kitty", "kitty"')


class WaylandTests(unittest.TestCase):
    def make(self, call=None, run=None):
        call = call or FakeCall()
        run = run or FakeRun()
        return GnomeWaylandDesktop(call=call, run=run, sleep=lambda s: None, spawn=now), call, run

    def test_wayland_key_calls_extension(self):
        d, call, _ = self.make()
        self.assertTrue(d.key("ctrl+Left", 2))
        self.assertEqual(call.calls[0], ("PressKeys", ("ctrl+Left", 2)))

    def test_wayland_active_window_and_class(self):
        d, _, _ = self.make(FakeCall({"FocusedWindow": ("42", "kitty", "t", 7)}))
        self.assertEqual((d.active_window(), d.window_class()), ("42", "kitty"))
        d, _, _ = self.make(FakeCall({"FocusedWindow": ("", "", "", 0)}))
        self.assertEqual((d.active_window(), d.window_class()), (None, None))

    def test_wayland_type_text_restores_clipboard(self):
        d, call, run = self.make(run=FakeRun({("wl-paste",): b"old"}))
        self.assertTrue(d.type_text("héllo"))
        cmds = [c for c, _ in run.calls]
        self.assertEqual(cmds[0][0], "wl-paste")
        self.assertEqual(cmds[1], ["wl-copy"])
        self.assertEqual(run.calls[1][1]["input"], "héllo".encode())
        self.assertEqual(call.calls, [("PressKeys", ("ctrl+v", 1))])
        self.assertEqual(cmds[2], ["wl-copy"])
        self.assertEqual(run.calls[2][1]["input"], b"old")

    def test_wayland_type_text_restores_empty_clipboard(self):
        d, _, run = self.make(run=FakeRun({("wl-paste",): 1}))
        d.type_text("x")
        self.assertEqual(run.calls[-1][0], ["wl-copy", "--clear"])

    def test_wayland_type_text_terminal_uses_ctrl_shift_v(self):
        d, call, _ = self.make()
        d.type_text("x", terminal=True)
        self.assertEqual(call.calls[0], ("PressKeys", ("ctrl+shift+v", 1)))

    def test_wayland_type_text_unicode_goes_via_clipboard(self):
        d, _, run = self.make()
        d.type_text("日本 🎤")
        copies = [k["input"] for c, k in run.calls if c == ["wl-copy"]]
        self.assertEqual(copies[0], "日本 🎤".encode("utf-8"))

    def test_rapid_type_text_keeps_user_clipboard_and_restores_once(self):
        pending = []
        run = FakeRun({("wl-paste",): b"mine"})
        d = GnomeWaylandDesktop(call=FakeCall(), run=run, sleep=lambda s: None,
                                spawn=pending.append)
        d.type_text("A")
        # the clipboard now holds A's text; B must not take that for the user's
        run.answers[("wl-paste",)] = b"A"
        d.type_text("B")
        self.assertEqual(len(pending), 2)
        for fn in pending:
            fn()
        copies = [k["input"] for c, k in run.calls if c == ["wl-copy"]]
        self.assertEqual(copies, [b"A", b"B", b"mine"])
        self.assertEqual(len([c for c, _ in run.calls if c[0] == "wl-paste"]), 1)

    def test_second_paste_comes_before_any_restore(self):
        pending = []
        d, call, run = self.make(run=FakeRun({("wl-paste",): b"mine"}))
        d._spawn = pending.append
        d.type_text("A")
        d.type_text("B")
        order = [("copy", k["input"]) for c, k in run.calls if c == ["wl-copy"]]
        self.assertEqual(order, [("copy", b"A"), ("copy", b"B")])
        self.assertEqual(len(call.calls), 2)

    def test_wayland_without_extension_reports_unavailable(self):
        d, _, _ = self.make(FakeCall(error=True))
        self.assertFalse(d.available())
        self.assertIsNone(d.active_window())
        self.assertFalse(d.key("a"))

    def test_wayland_available_and_activate(self):
        d, call, _ = self.make(FakeCall({"Version": (1,)}))
        self.assertTrue(d.available())
        self.assertTrue(d.activate("7"))
        self.assertEqual(call.calls[-1], ("ActivateWindow", ("7",)))


class FactoryTests(unittest.TestCase):
    def tearDown(self):
        desktop._desktop = None

    def test_get_desktop_picks_by_session_type(self):
        desktop._desktop = None
        self.assertIsInstance(get_desktop({"XDG_SESSION_TYPE": "wayland"}), GnomeWaylandDesktop)
        self.assertIsInstance(get_desktop(), GnomeWaylandDesktop)  # cached
        desktop._desktop = None
        self.assertIsInstance(get_desktop({"XDG_SESSION_TYPE": "x11"}), X11Desktop)
        desktop._desktop = None
        self.assertIsInstance(get_desktop({}), X11Desktop)


if __name__ == "__main__":
    unittest.main()
