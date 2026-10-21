import subprocess
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

    def test_stale_restore_is_noop_when_run_during_next_paste(self):
        pending = []
        run = FakeRun({("wl-paste",): b"mine"})
        d, call, _ = self.make(run=run)
        d._spawn = pending.append
        original = d.clipboard_set
        fired = []

        def hooked(data):
            if data == b"B" and not fired:
                fired.append(True)
                pending[0]()  # A's restore, mid-way through B's paste
            return original(data)
        d.clipboard_set = hooked
        d.type_text("A")
        d.type_text("B")
        copies = [k["input"] for c, k in run.calls if c == ["wl-copy"]]
        self.assertEqual(copies, [b"A", b"B"])
        self.assertFalse([c for c, _ in run.calls if c == ["wl-copy", "--clear"]])
        self.assertEqual(len(call.calls), 2)

    def test_paste_settles_before_following_key_or_type_text(self):
        log = []

        class LogCall(FakeCall):
            def __call__(self, method, args=None):
                log.append(("dbus", method, args))
                return super().__call__(method, args)

        class LogRun(FakeRun):
            def __call__(self, cmd, **kw):
                if cmd == ["wl-copy"]:
                    log.append(("copy", kw["input"]))
                return super().__call__(cmd, **kw)
        d = GnomeWaylandDesktop(call=LogCall(), run=LogRun(), sleep=lambda s: log.append(("sleep", s)),
                                spawn=lambda fn: None)
        self.assertEqual(d.paste_settle_s, 0.12)
        d.type_text("A")
        d.key("BackSpace")
        d.type_text("B")
        i = log.index(("dbus", "PressKeys", ("ctrl+v", 1)))
        self.assertEqual(log[i + 1], ("sleep", 0.12))
        self.assertEqual(log[i + 2], ("dbus", "PressKeys", ("BackSpace", 1)))
        j = log.index(("copy", b"B"))
        self.assertGreater(j, i + 2)

    def test_key_waits_for_a_paste_in_progress(self):
        import threading
        d, call, _ = self.make()
        seen = {}
        t = []

        def sleep(s):
            if s == 0.12 and not t:
                th = threading.Thread(target=lambda: d.key("BackSpace"))
                t.append(th)
                th.start()
                th.join(0.3)
                seen["blocked"] = th.is_alive()
        d._sleep = sleep
        d.type_text("A")
        t[0].join(2)
        self.assertTrue(seen["blocked"])
        self.assertEqual([c for c in call.calls if c[0] == "PressKeys"],
                         [("PressKeys", ("ctrl+v", 1)), ("PressKeys", ("BackSpace", 1))])

    def test_hotkey_then_first_key_waits_for_modifier_release(self):
        sleeps, t = [], [100.0]
        d = GnomeWaylandDesktop(call=FakeCall({"GrabAccelerator": (5,)}), run=FakeRun(),
                                sleep=lambda s: sleeps.append(s), spawn=now, clock=lambda: t[0])
        d._sub = object()
        fired = []
        self.assertTrue(d.bind_hotkey("<Alt>d", lambda: fired.append(1)))
        d._actions[5]()
        self.assertEqual(fired, [1])
        t[0] += 0.05
        d.key("Return")
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 0.10)
        d.key("Return")  # consumed
        self.assertEqual(len(sleeps), 1)

    def test_no_modifier_wait_long_after_hotkey_or_without_one(self):
        sleeps, t = [], [100.0]
        d = GnomeWaylandDesktop(call=FakeCall({"GrabAccelerator": (5,)}), run=FakeRun(),
                                sleep=lambda s: sleeps.append(s), spawn=now, clock=lambda: t[0])
        d._sub = object()
        d.key("a")
        d.bind_hotkey("<Alt>d", lambda: None)
        d._actions[5]()
        t[0] += 5
        d.key("a")
        self.assertEqual(sleeps, [])

    def test_type_text_after_hotkey_waits_for_modifier_release(self):
        sleeps, t = [], [100.0]
        d = GnomeWaylandDesktop(call=FakeCall({"GrabAccelerator": (5,)}), run=FakeRun(),
                                sleep=lambda s: sleeps.append(s), spawn=now, clock=lambda: t[0])
        d._sub = object()
        d.bind_hotkey("<Alt>d", lambda: None)
        d._actions[5]()
        d.type_text("x")
        self.assertAlmostEqual(sleeps[0], 0.15)

    def test_clipboard_set_failure_skips_paste(self):
        d, call, run = self.make(run=FakeRun({("wl-copy",): 1}))
        self.assertFalse(d.type_text("x"))
        self.assertEqual([c for c in call.calls if c[0] == "PressKeys"], [])

    def test_clipboard_read_failure_skips_restore(self):
        def boom(cmd, **kw):
            if cmd[0] == "wl-paste":
                raise subprocess.TimeoutExpired(cmd, 1)
            return run(cmd, **kw)
        run = FakeRun()
        d = GnomeWaylandDesktop(call=FakeCall(), run=boom, sleep=lambda s: None, spawn=now)
        d.type_text("x")
        cmds = [c for c, _ in run.calls]
        self.assertEqual(cmds, [["wl-copy"]])

    def test_clipboard_read_error_exit_skips_restore(self):
        class R(FakeRun):
            def __call__(self, cmd, **kw):
                r = super().__call__(cmd, **kw)
                if cmd[0] == "wl-paste":
                    r.stderr = b"Failed to connect to a Wayland server"
                return r
        run = R({("wl-paste",): 1})
        d = GnomeWaylandDesktop(call=FakeCall(), run=run, sleep=lambda s: None, spawn=now)
        d.type_text("x")
        self.assertEqual([c for c, _ in run.calls if c == ["wl-copy", "--clear"]], [])

    def test_empty_clipboard_with_no_selection_message_is_cleared(self):
        class R(FakeRun):
            def __call__(self, cmd, **kw):
                r = super().__call__(cmd, **kw)
                if cmd[0] == "wl-paste":
                    r.stderr = b"No selection"
                return r
        run = R({("wl-paste",): 1})
        d = GnomeWaylandDesktop(call=FakeCall(), run=run, sleep=lambda s: None, spawn=now)
        d.type_text("x")
        self.assertEqual(run.calls[-1][0], ["wl-copy", "--clear"])

    def test_wl_copy_calls_have_timeout(self):
        d, _, run = self.make(run=FakeRun({("wl-paste",): b"old"}))
        d.type_text("x")
        d2, _, run2 = self.make(run=FakeRun({("wl-paste",): 1}))
        d2.type_text("x")
        for r in (run, run2):
            for c, k in r.calls:
                if c[0] == "wl-copy":
                    self.assertEqual(k.get("timeout"), 2, c)

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
        up = lambda: _FakeWayland(True)
        self.assertIsInstance(get_desktop({"XDG_SESSION_TYPE": "wayland"}, make_wayland=up), _FakeWayland)
        self.assertIsInstance(get_desktop(), _FakeWayland)  # cached
        desktop._desktop = None
        self.assertIsInstance(get_desktop({"XDG_SESSION_TYPE": "x11"}), X11Desktop)
        desktop._desktop = None
        self.assertIsInstance(get_desktop({}), X11Desktop)

    def test_wayland_without_extension_falls_back_to_xwayland(self):
        desktop._desktop = None
        d = get_desktop({"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"},
                        make_wayland=lambda: _FakeWayland(False))
        self.assertIsInstance(d, X11Desktop)
        self.assertTrue(d.xwayland_fallback)

    def test_wayland_without_extension_or_xwayland_keeps_wayland_desktop(self):
        desktop._desktop = None
        d = get_desktop({"XDG_SESSION_TYPE": "wayland"}, make_wayland=lambda: _FakeWayland(False))
        self.assertIsInstance(d, _FakeWayland)

    def test_plain_x11_is_not_a_fallback(self):
        desktop._desktop = None
        self.assertFalse(get_desktop({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"}).xwayland_fallback)


class _FakeWayland:
    def __init__(self, up):
        self._up = up

    def available(self):
        return self._up

if __name__ == "__main__":
    unittest.main()
