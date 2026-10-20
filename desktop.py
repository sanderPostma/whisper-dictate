"""Desktop backend abstraction: X11 (xdotool/xprop/xclip/Keybinder) and GNOME Wayland
(whisper-dictate Shell extension over D-Bus plus wl-clipboard)."""
import os
import subprocess
import threading
import time

from gi.repository import Gio, GLib

BUS_NAME = "org.gnome.Shell"
OBJECT_PATH = "/org/whisperdictate/Shell"
INTERFACE = "org.whisperdictate.Shell"
_SIGNATURES = {
    "ActivateWindow": "(s)",
    "PressKeys": "(su)",
    "GrabAccelerator": "(s)",
    "UngrabAccelerator": "(u)",
}


def _thread_spawn(fn):
    threading.Thread(target=fn, daemon=True).start()


class X11Desktop:
    def __init__(self, run=subprocess.run):
        self._run = run
        self._bound = []

    def available(self):
        return True

    def active_window(self):
        try:
            res = self._run(["xdotool", "getactivewindow"], capture_output=True, text=True,
                            check=True, timeout=1)
            return (res.stdout or "").strip() or None
        except Exception:
            return None

    def window_class(self):
        wid = self.active_window()
        if not wid:
            return None
        try:
            res = self._run(["xprop", "-id", wid, "WM_CLASS"], capture_output=True, text=True,
                            check=True, timeout=1)
            return res.stdout.strip()
        except Exception:
            return None

    def activate(self, window_id):
        try:
            self._run(["xdotool", "windowactivate", window_id], check=False,
                      capture_output=True, timeout=2)
            return True
        except Exception:
            return False

    def key(self, chord, repeat=1, delay_ms=None):
        cmd = ["xdotool", "key", "--clearmodifiers"]
        if delay_ms is not None:
            cmd += ["--delay", str(delay_ms)]
        if repeat > 1:
            cmd += ["--repeat", str(repeat)]
        cmd.append(chord)
        try:
            return self._run(cmd, capture_output=True, timeout=10).returncode == 0
        except Exception:
            return False

    def type_text(self, text, terminal=False, delay_ms=25):
        try:
            res = self._run(["xdotool", "type", "--clearmodifiers", "--delay", str(delay_ms),
                             "--", text], capture_output=True,
                            timeout=10 + len(text) * delay_ms / 1000 * 2)
            return res.returncode == 0
        except Exception:
            return False

    def clipboard_set(self, data):
        try:
            self._run(["xclip", "-selection", "clipboard"], input=data, check=False)
            return True
        except Exception:
            return False

    def clipboard_get(self):
        try:
            return self._run(["xclip", "-o", "-selection", "clipboard"],
                             capture_output=True, timeout=1).stdout
        except Exception:
            return None

    def bind_hotkey(self, accel, callback):
        try:
            import gi
            gi.require_version("Keybinder", "3.0")
            from gi.repository import Keybinder
            Keybinder.init()
            if Keybinder.bind(accel, lambda _keystring: callback()):
                self._bound.append(accel)
                return True
        except Exception:
            pass
        return False

    def unbind_all(self):
        try:
            from gi.repository import Keybinder
            for accel in self._bound:
                Keybinder.unbind(accel)
        except Exception:
            pass
        self._bound = []


class GnomeWaylandDesktop:
    def __init__(self, call=None, run=subprocess.run, sleep=time.sleep, spawn=_thread_spawn):
        self._call = call or self._dbus_call
        self._run = run
        self._sleep = sleep
        self._spawn = spawn
        self._actions = {}
        self._sub = None
        self._lock = threading.Lock()
        self._saved = None
        self._has_saved = False
        self._gen = 0

    @staticmethod
    def _dbus_call(method, args=None):
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        variant = GLib.Variant(_SIGNATURES[method], args) if method in _SIGNATURES else None
        res = bus.call_sync(BUS_NAME, OBJECT_PATH, INTERFACE, method, variant, None,
                            Gio.DBusCallFlags.NONE, 1000, None)
        return res.unpack()

    def _try(self, method, args=None):
        try:
            return self._call(method, args)
        except GLib.Error:
            return None

    def available(self):
        return self._try("Version") is not None

    def _focused(self):
        res = self._try("FocusedWindow")
        return res if res and res[0] else None

    def active_window(self):
        res = self._focused()
        return res[0] if res else None

    def window_class(self):
        res = self._focused()
        return res[1] if res else None

    def activate(self, window_id):
        res = self._try("ActivateWindow", (window_id,))
        return bool(res and res[0])

    def key(self, chord, repeat=1, delay_ms=None):
        res = self._try("PressKeys", (chord, repeat))
        return bool(res and res[0])

    def clipboard_set(self, data):
        try:
            return self._run(["wl-copy"], input=data, check=False).returncode == 0
        except Exception:
            return False

    def clipboard_get(self):
        try:
            res = self._run(["wl-paste", "--no-newline"], capture_output=True, timeout=1)
        except Exception:
            return None
        return res.stdout if res.returncode == 0 else None

    def type_text(self, text, terminal=False, delay_ms=25):
        # Serialised; the user's clipboard is saved once across overlapping sends and
        # restored once, 0.3s after the LAST paste.
        with self._lock:
            if not self._has_saved:
                self._saved = self.clipboard_get()
                self._has_saved = True
            self._gen += 1
            gen = self._gen
            self.clipboard_set(text.encode("utf-8"))
            self._sleep(0.05)
            ok = self.key("ctrl+shift+v" if terminal else "ctrl+v")

        def restore():
            self._sleep(0.3)
            with self._lock:
                if gen != self._gen:
                    return
                saved, self._saved, self._has_saved = self._saved, None, False
                if saved:
                    self.clipboard_set(saved)
                else:
                    try:
                        self._run(["wl-copy", "--clear"], check=False)
                    except Exception:
                        pass
        self._spawn(restore)
        return ok

    def bind_hotkey(self, accel, callback):
        res = self._try("GrabAccelerator", (accel,))
        action = res[0] if res else 0
        if not action:
            return False
        self._actions[action] = callback
        if self._sub is None:
            try:
                bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
                self._sub = bus.signal_subscribe(
                    BUS_NAME, INTERFACE, "AcceleratorActivated", OBJECT_PATH, None,
                    Gio.DBusSignalFlags.NONE, self._on_signal)
            except GLib.Error:
                return False
        return True

    def _on_signal(self, _conn, _sender, _path, _iface, _signal, params, *_):
        cb = self._actions.get(params.unpack()[0])
        if cb:
            GLib.idle_add(lambda: (cb(), False)[1])

    def unbind_all(self):
        for action in list(self._actions):
            self._try("UngrabAccelerator", (action,))
        self._actions = {}


_desktop = None


def get_desktop(env=os.environ):
    global _desktop
    if _desktop is None:
        if env.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            _desktop = GnomeWaylandDesktop()
        else:
            _desktop = X11Desktop()
    return _desktop
