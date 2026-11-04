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
SCREENSAVER = "org.gnome.ScreenSaver"
_SIGNATURES = {
    "ActivateWindow": "(s)",
    "PressKeys": "(su)",
    "GrabAccelerator": "(s)",
    "UngrabAccelerator": "(u)",
}


def _thread_spawn(fn):
    threading.Thread(target=fn, daemon=True).start()


class X11Desktop:
    def __init__(self, run=subprocess.run, xwayland_fallback=False):
        self._run = run
        # Standing in for a Wayland session whose Shell extension is not loaded
        # yet: xdotool still sees XWayland windows.
        self.xwayland_fallback = xwayland_fallback
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
    paste_settle_s = 0.12      # let the focused app consume ctrl+v before touching the clipboard
    modifier_release_s = 0.15  # hotkey modifiers may still be physically held
    modifier_window_s = 1.0

    def __init__(self, call=None, run=subprocess.run, sleep=time.sleep, spawn=_thread_spawn,
                 clock=time.monotonic, timeout=None):
        self._call = call or self._dbus_call
        self._timeout = timeout or GLib.timeout_add
        self._run = run
        self._sleep = sleep
        self._spawn = spawn
        self._clock = clock
        self._actions = {}
        self._bindings = {}  # accel -> callback, so grabs can be renewed
        self._sub = None
        # Re-entrant: type_text holds it across its own key() call.
        self._lock = threading.RLock()
        self._saved = None
        self._saved_state = None  # None (nothing saved) | "ok" | "empty" | "failed"
        self._gen = 0
        self._hotkey_fired_at = None

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

    def _wait_modifier_release(self):
        fired, self._hotkey_fired_at = self._hotkey_fired_at, None
        if fired is None:
            return
        elapsed = self._clock() - fired
        if 0 <= elapsed < self.modifier_window_s and elapsed < self.modifier_release_s:
            self._sleep(self.modifier_release_s - elapsed)

    def key(self, chord, repeat=1, delay_ms=None):
        with self._lock:
            self._wait_modifier_release()
            res = self._try("PressKeys", (chord, repeat))
        return bool(res and res[0])

    def clipboard_set(self, data):
        try:
            return self._run(["wl-copy"], input=data, check=False, timeout=2).returncode == 0
        except Exception:
            return False

    def _read_clipboard(self):
        """(state, data): state is "ok", "empty" (nothing selected) or "failed"."""
        try:
            res = self._run(["wl-paste", "--no-newline"], capture_output=True, timeout=1)
        except Exception:
            return "failed", None
        if res.returncode == 0:
            return "ok", res.stdout
        err = (getattr(res, "stderr", b"") or b"")
        if isinstance(err, str):
            err = err.encode()
        err = err.lower()
        if not err.strip() or b"no selection" in err or b"nothing is copied" in err:
            return "empty", None
        return "failed", None

    def clipboard_get(self):
        state, data = self._read_clipboard()
        return data if state == "ok" else None

    def type_text(self, text, terminal=False, delay_ms=25):
        # Serialised; the user's clipboard is saved once across overlapping sends and
        # restored once, 0.3s after the LAST paste.
        with self._lock:
            self._wait_modifier_release()
            if self._saved_state is None:
                self._saved_state, self._saved = self._read_clipboard()
            self._gen += 1
            gen = self._gen
            if self.clipboard_set(text.encode("utf-8")):
                self._sleep(0.05)
                ok = self.key("ctrl+shift+v" if terminal else "ctrl+v")
                self._sleep(self.paste_settle_s)
            else:
                ok = False

        def restore():
            self._sleep(0.3)
            with self._lock:
                if gen != self._gen:
                    return
                state, saved = self._saved_state, self._saved
                self._saved, self._saved_state = None, None
                if state == "failed":
                    return
                if state == "ok" and saved:
                    self.clipboard_set(saved)
                else:
                    try:
                        self._run(["wl-copy", "--clear"], check=False, timeout=2)
                    except Exception:
                        pass
        self._spawn(restore)
        return ok

    # The Shell disables extensions while the screen is locked, which drops every
    # grab; after unlock the extension is back but holds nothing. Grabs are
    # renewed when the extension announces Ready, and on unlock (for an
    # extension loaded before Ready existed; it may come back a moment later).
    rebind_interval_ms = 500
    rebind_attempts = 20

    def bind_hotkey(self, accel, callback):
        self._bindings[accel] = callback
        if not self._grab(accel, callback):
            return False
        if self._sub is None:
            try:
                bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
                self._sub = [
                    bus.signal_subscribe(
                        BUS_NAME, INTERFACE, None, OBJECT_PATH, None,
                        Gio.DBusSignalFlags.NONE, self._on_signal),
                    bus.signal_subscribe(
                        None, SCREENSAVER, "ActiveChanged", "/org/gnome/ScreenSaver", None,
                        Gio.DBusSignalFlags.NONE, self._on_signal),
                ]
            except GLib.Error:
                return False
        return True

    def _grab(self, accel, callback):
        res = self._try("GrabAccelerator", (accel,))
        action = res[0] if res else 0
        if not action:
            return False
        def dispatch(cb=callback):
            self._hotkey_fired_at = self._clock()
            return cb()
        dispatch.accel = accel
        for old in [a for a, cb in self._actions.items() if cb.accel == accel]:
            del self._actions[old]
        self._actions[action] = dispatch
        return True

    def rebind(self):
        """Grab every bound hotkey again; True when all succeeded."""
        self._actions = {}
        ok = True
        for accel, callback in list(self._bindings.items()):
            ok = self._grab(accel, callback) and ok
        return ok

    def _rebind_until_ok(self):
        left = [self.rebind_attempts]
        def tick():
            left[0] -= 1
            return not self.rebind() and left[0] > 0
        self._timeout(self.rebind_interval_ms, tick)

    def _on_signal(self, _conn, _sender, _path, iface, signal, params, *_):
        if iface == INTERFACE and signal == "AcceleratorActivated":
            cb = self._actions.get(params.unpack()[0])
            if cb:
                GLib.idle_add(lambda: (cb(), False)[1])
        elif iface == INTERFACE and signal == "Ready":
            if self._bindings:
                print("[whisper-dictate] Shell extension re-enabled, renewing hotkeys")
                self.rebind()
        elif iface == SCREENSAVER and signal == "ActiveChanged":
            if not params.unpack()[0] and self._bindings:
                print("[whisper-dictate] Screen unlocked, renewing hotkeys")
                self._rebind_until_ok()

    def unbind_all(self):
        for action in list(self._actions):
            self._try("UngrabAccelerator", (action,))
        self._actions = {}
        self._bindings = {}


_desktop = None


def get_desktop(env=os.environ, make_wayland=None):
    global _desktop
    if _desktop is None:
        if env.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            _desktop = (make_wayland or GnomeWaylandDesktop)()
            if not _desktop.available() and env.get("DISPLAY"):
                _desktop = X11Desktop(xwayland_fallback=True)
        else:
            _desktop = X11Desktop()
    return _desktop
