# GNOME Wayland Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every feature of whisper-dictate (hotkeys, typing, live dictation, focus restore, terminal/kitty/WezTerm/achat targeting, clipboard commands) works on Ubuntu 26 / GNOME 50 Wayland, with no feature degraded relative to X11.

**Architecture:** A small GNOME Shell extension exports a D-Bus object inside gnome-shell that reports the focused window (id, WM class, title, pid), activates windows, presses key chords via a Clutter virtual keyboard, and grabs global accelerators (emitting a signal when pressed). A new Python module `desktop.py` hides the session type behind one interface: `X11Desktop` (today's xdotool/xprop/xclip/Keybinder calls, unchanged behaviour) and `GnomeWaylandDesktop` (the extension over D-Bus + wl-copy/wl-paste). All callers in `whisper_dictate.py`, `live_output.py`, `live_achat.py` go through `desktop.py`; nothing else calls xdotool/xprop/xclip/Keybinder directly.

**Tech Stack:** Python 3.14, PyGObject (Gio D-Bus, GLib), GJS ESM extension for GNOME Shell 50, wl-clipboard.

**Spec:** this conversation's decisions (no separate spec):
- Wayland is the primary platform from now on; never degrade features on Wayland.
- Text insertion on Wayland: clipboard (wl-copy) + paste key chord, clipboard restored afterwards.
- Focused-window info comes from a GNOME Shell extension.
- Global hotkeys: user picked the GlobalShortcuts portal. **Deviation proposed:** since the extension already exists, it grabs the accelerators itself (`global.display.grab_accelerator`) — no portal confirm dialog, same config strings (`<Alt>d`, `<Alt>l`). Likewise key presses go through the extension's virtual keyboard instead of the RemoteDesktop portal (no permission prompt, no daemon). Confirm at plan review.

## Global Constraints

- Session selection: `XDG_SESSION_TYPE == "wayland"` → `GnomeWaylandDesktop`; anything else → `X11Desktop`.
- D-Bus: bus name `org.gnome.Shell`, object path `/org/whisperdictate/Shell`, interface `org.whisperdictate.Shell`.
- Extension UUID `whisper-dictate@whisper-dictate.local`, `shell-version: ["50"]`, lives in `gnome-extension/whisper-dictate@whisper-dictate.local/`.
- Key chord strings keep the existing xdotool syntax everywhere (`ctrl+shift+v`, `BackSpace`, `Return`, `ctrl+Left`, `Home`, `End`); the extension parses that syntax.
- Hotkey strings keep the existing Keybinder/GTK accelerator syntax from config (`<Alt>d`).
- X11 behaviour must stay byte-for-byte the same commands as today (existing tests that assert xdotool argv keep passing, possibly moved to `tests/test_desktop.py`).
- New runtime deps: `wl-clipboard` (apt). No new pip deps.
- Full test suite (`venv/bin/python -m pytest -q`, 472 passing at start) stays green after every task.

## Review Focus

1. Extension not installed/enabled on Wayland → app must start, print one clear line (`✗ GNOME extension whisper-dictate not active — run ./install-gnome-extension.sh and log out/in`) and show a notification, not crash or silently do nothing. Test in Task 2 (`test_wayland_without_extension_reports_unavailable`).
2. Clipboard restore after paste: user's previous clipboard (including empty clipboard) must come back; an empty clipboard must not become the literal text. Test in Task 2 (`test_wayland_type_text_restores_clipboard`, `..._empty_clipboard`).
3. Hotkey held modifiers: pressing a chord while the user still holds Alt from the hotkey must not produce Alt+V. The extension releases/ignores physically held modifiers the way `xdotool --clearmodifiers` does (Task 1: virtual device presses explicit modifiers only; Python waits `0.15s` after hotkey before first key on Wayland — Task 4 test `test_wayland_hotkey_output_waits_for_modifier_release`).
4. Window ids: on Wayland ids are Mutter window ids as decimal strings; equality comparisons in `still_focused` must work unchanged. Test in Task 3 (`test_xdotool_target_still_focused_uses_desktop`).
5. Non-ASCII text (é, emoji, CJK) must arrive intact on Wayland — guaranteed by paste path; test in Task 2 (`test_wayland_type_text_unicode_goes_via_clipboard`).

---

### Task 1: GNOME Shell extension + installer

**Files:**
- Create: `gnome-extension/whisper-dictate@whisper-dictate.local/metadata.json`
- Create: `gnome-extension/whisper-dictate@whisper-dictate.local/extension.js`
- Create: `install-gnome-extension.sh`

**Interfaces (D-Bus, `org.whisperdictate.Shell` at `/org/whisperdictate/Shell` on `org.gnome.Shell`):**
- `FocusedWindow() -> (s id, s wm_class, s title, u pid)`; empty id / class when nothing is focused. `id` = `String(metaWindow.get_id())`.
- `ActivateWindow(s id) -> b` — finds the window among `global.get_window_actors()`, `Main.activateWindow(win)`.
- `PressKeys(s chord, u repeat) -> b` — chord in xdotool syntax; modifiers `ctrl|shift|alt|super`, key name resolved with `Clutter.KEY_<name>` (fallback: single char → `Clutter.unicode_to_keysym`). Uses one `seat.create_virtual_device(Clutter.InputDeviceType.KEYBOARD_DEVICE)` created in `enable()`; for each repeat: press modifiers, press/release key, release modifiers in reverse, timestamps from `GLib.get_monotonic_time()`, ~8 ms between repeats. Returns false on unknown key name.
- `GrabAccelerator(s accel) -> u action` — `global.display.grab_accelerator(accel, Meta.KeyBindingFlags.NONE)`; on non-zero action `Main.wm.allowKeybinding(Meta.external_binding_name_for_action(action), Shell.ActionMode.ALL)`. Returns 0 on failure.
- `UngrabAccelerator(u action) -> b`.
- Signal `AcceleratorActivated(u action)` — emitted from `global.display` `accelerator-activated` for actions this extension grabbed.
- `Version() -> u` returns `1`.
- `disable()` ungrabs everything, unexports, destroys the virtual device.

- [ ] **Step 1: Write metadata.json** (uuid, name "Whisper Dictate helper", description, `shell-version: ["50"]`).
- [ ] **Step 2: Write extension.js** as an ESM `Extension` subclass implementing the interface above via `Gio.DBusExportedObject.wrapJSObject(xml, impl)` exported on `Gio.DBus.session`.
- [ ] **Step 3: Write `install-gnome-extension.sh`**: copy the folder to `~/.local/share/gnome-shell/extensions/`, `gnome-extensions enable <uuid>`, print "Log out and back in to load it (Wayland cannot restart gnome-shell)".
- [ ] **Step 4: Verify in a nested shell.** Run `dbus-run-session -- bash -c 'gnome-shell --headless --virtual-monitor 1280x720 --wayland & sleep 8; gnome-extensions enable <uuid>; ...'` with `XDG_DATA_HOME` pointing at a temp dir holding the extension; then `gdbus call --session --dest org.gnome.Shell --object-path /org/whisperdictate/Shell --method org.whisperdictate.Shell.Version`.
  Expected: `(uint32 1,)`, and `FocusedWindow` returns a 4-tuple. If headless gnome-shell can't run in this environment, record that and fall back to `gjs` syntax check (`gjs -c 'import("file://.../extension.js").catch(e=>{print(e); })'` — import failure on `resource:///org/gnome/shell` is expected; a SyntaxError is not) and leave live verification for the manual step in Task 5.
- [ ] **Step 5: Commit** `feat: GNOME Shell helper extension for Wayland focus, keys and hotkeys`.

### Task 2: `desktop.py` backend abstraction

**Files:**
- Create: `desktop.py`
- Test: `tests/test_desktop.py`

**Interfaces:**
- Produces:
  - `class X11Desktop(run=subprocess.run)` and `class GnomeWaylandDesktop(call=None, run=subprocess.run, sleep=time.sleep)` where `call(method: str, args: GLib.Variant|None) -> tuple` defaults to a Gio session-bus call to the extension (sync, timeout 1000 ms).
  - Common methods: `available() -> bool`, `active_window() -> str|None`, `window_class() -> str|None` (lowercase-insensitive string; X11 returns today's raw `xprop` WM_CLASS line, Wayland returns the class string), `activate(window_id: str) -> bool`, `key(chord: str, repeat: int = 1, delay_ms: int | None = None) -> bool`, `type_text(text: str, terminal: bool = False, delay_ms: int = 25) -> bool`, `clipboard_set(data: bytes) -> bool`, `clipboard_get() -> bytes|None`, `bind_hotkey(accel: str, callback: Callable[[], None]) -> bool`, `unbind_all() -> None`.
  - `get_desktop(env=os.environ) -> X11Desktop | GnomeWaylandDesktop` (module-level singleton cache `_desktop`; `get_desktop()` returns the cached one).
- X11 commands (must equal today's): `key` → `["xdotool","key","--clearmodifiers", (*"--delay",str(d)), (*"--repeat",str(n) if n>1), chord]`; `type_text` → `["xdotool","type","--clearmodifiers","--delay",str(delay_ms),"--",text]`; clipboard via `xclip -selection clipboard` / `xclip -o -selection clipboard`; `activate` → `xdotool windowactivate <id>`; hotkeys via `Keybinder.init()/bind/unbind` (import lazily).
- Wayland: `key` → `call("PressKeys", (chord, repeat))`; `type_text` → save `clipboard_get()`, `clipboard_set(text)`, sleep 0.05, `key("ctrl+shift+v" if terminal else "ctrl+v")`, then restore in a daemon thread after 0.3 s (empty/None saved → `wl-copy --clear`); clipboard via `wl-copy` (stdin) / `wl-paste --no-newline` (returncode≠0 → `None`); `bind_hotkey` → `GrabAccelerator`, subscribe once to `AcceleratorActivated` on the session bus, dispatch to callback via `GLib.idle_add`; `available()` → `Version` call succeeds.

- [ ] **Step 1: Write failing tests** in `tests/test_desktop.py` using a fake `run` recorder and fake `call`:
  - `test_x11_key_matches_xdotool_argv` — `key("BackSpace", repeat=3, delay_ms=8)` → `["xdotool","key","--clearmodifiers","--delay","8","--repeat","3","BackSpace"]`.
  - `test_x11_type_text_argv` — delay 25 argv as above.
  - `test_wayland_key_calls_extension` — fake call receives `("PressKeys", ("ctrl+Left", 2))`, returns `(True,)` → `True`.
  - `test_wayland_active_window_and_class` — `FocusedWindow` → `("42","kitty","t",7)` gives `"42"` / `"kitty"`; `("","","",0)` gives `None` / `None`.
  - `test_wayland_type_text_restores_clipboard` — saved `b"old"`; sequence: wl-paste, wl-copy `"héllo"`, PressKeys `ctrl+v`, wl-copy `b"old"` (run the restore synchronously by injecting a `spawn` that calls immediately).
  - `test_wayland_type_text_restores_empty_clipboard` — wl-paste fails → restore is `wl-copy --clear`.
  - `test_wayland_type_text_terminal_uses_ctrl_shift_v`.
  - `test_wayland_type_text_unicode_goes_via_clipboard` — `"日本 🎤"` bytes reach wl-copy stdin as UTF-8.
  - `test_wayland_without_extension_reports_unavailable` — call raises `GLib.Error` → `available()` False, `active_window()` None, `key()` False.
  - `test_get_desktop_picks_by_session_type`.
- [ ] **Step 2:** `venv/bin/python -m pytest tests/test_desktop.py -q` → FAIL (module missing).
- [ ] **Step 3:** Implement `desktop.py` (add `spawn=` injectable for the restore thread).
- [ ] **Step 4:** Run the file's tests → PASS; full suite → PASS.
- [ ] **Step 5: Commit** `feat: desktop backend abstraction for X11 and GNOME Wayland`.

### Task 3: Port live output targets to `desktop`

**Files:**
- Modify: `live_output.py` (`xdotool_active_window`, `press_key`, `WezTermTarget`, `XdotoolTarget`, `choose_target`)
- Modify: `live_achat.py` (`AchatTarget.detect`, `still_focused`, imports)
- Test: `tests/test_live_output.py`, `tests/test_live_achat.py`

**Interfaces:**
- Consumes: `desktop.get_desktop()`, `Desktop.active_window/key/type_text` from Task 2.
- Produces: `active_window(desktop=None) -> str|None` replaces `xdotool_active_window(run)` (keep `xdotool_active_window` as an alias only if callers remain — prefer removing). `press_key(key, desktop=None) -> bool`. `XdotoolTarget` renamed `KeyboardTarget(window_id, desktop=None, terminal=False)`, `name = "keyboard"`; `send` uses `desktop.key("BackSpace", repeat=n, delay_ms=8)` then `desktop.type_text(insert, terminal=self.terminal, delay_ms=8)`. `WezTermTarget`/`AchatTarget` take `desktop=None` alongside `run` (run stays for `wezterm cli`). `choose_target(wm_class, run=..., achat_detect=None, desktop=None)`.

- [ ] **Step 1:** Update tests to inject a fake desktop (record `key`/`type_text` calls, scripted `active_window`) instead of asserting xdotool argv; add `test_xdotool_target_still_focused_uses_desktop` (renamed `test_keyboard_target_still_focused_uses_desktop`): window `"42"` → True, `"43"` → False.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement; grep `live_output.py live_achat.py` for `xdotool` → no hits except docstrings.
- [ ] **Step 4:** Full suite → PASS.
- [ ] **Step 5: Commit** `refactor: live targets use the desktop backend`.

### Task 4: Port `whisper_dictate.py` to `desktop`

**Files:**
- Modify: `whisper_dictate.py` — `get_focused_window` (~l.576), `restore_focus` (~l.587), `get_focused_window_class` (~l.1281), `_paste_via_clipboard_preserving` (~l.1632), `output_text` Enter press (~l.1672), `_output_text` clipboard/typing (~l.1776–1810), hotkey setup/teardown (~l.2576–2623), Keybinder imports (l.67, l.75), live/oneshot target creation (~l.1149, 1185).
- Test: `tests/test_focus_restore.py`, `tests/test_hotkey_action.py`, `tests/test_clipboard_commands.py` and any test patching `subprocess.run` for xdotool.

**Interfaces:**
- Consumes: `get_desktop()` and its methods (Task 2); `choose_target(..., desktop=)`, `press_key(..., desktop=)` (Task 3).
- `self.desktop = get_desktop()` set in `__init__`; all X calls go through it. Keybinder `gi.require_version` moves into `X11Desktop` (lazy), so the app no longer needs Keybinder at import on Wayland.
- At startup on Wayland, if `not self.desktop.available()`: print `✗ GNOME extension whisper-dictate not active — run ./install-gnome-extension.sh and log out/in` and `notify-send` the same; keep running (tray + menu still work).
- Wayland hotkey: before output after a hotkey-triggered stop, `GnomeWaylandDesktop` callers wait 0.15 s (existing `time.sleep(0.3)` before typing already covers one-shot; add the same guard to the Enter press path).

- [ ] **Step 1:** Adapt/add tests: `test_restore_focus_uses_desktop_activate` (only when focus moved), `test_hotkeys_bound_through_desktop` (both `<Alt>d` and live hotkey bound, callback toggles), `test_wayland_hotkey_output_waits_for_modifier_release`, `test_missing_extension_warns_and_keeps_running`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Afterwards `grep -n "xdotool\|xclip\|xprop\|Keybinder" whisper_dictate.py` → only comments.
- [ ] **Step 4:** Full suite → PASS. Smoke: `timeout 20 ./whisper-dictate.sh > $SCRATCH/wd.log 2>&1` on this Wayland session → log shows either `✓ Hotkey <Alt>d registered` (extension active) or the extension warning; no traceback.
- [ ] **Step 5: Commit** `feat: run on GNOME Wayland through the desktop backend`.

### Task 5: Docs, install, manual verification

**Files:**
- Modify: `README.md` (prerequisites: add `wl-clipboard`; new "GNOME Wayland" section: run `./install-gnome-extension.sh`, log out/in; note X11 still supported)
- Modify: `whisper-dictate.service` only if it hardcodes `DISPLAY`/X11 env (check; on Wayland it needs `WAYLAND_DISPLAY` + `DBUS_SESSION_BUS_ADDRESS` from the user session — `systemctl --user import-environment` note).

- [ ] **Step 1:** Update docs.
- [ ] **Step 2:** Run `./install-gnome-extension.sh`; tell the user to log out/in.
- [ ] **Step 3 (manual, user):** checklist — Alt+D in gedit/Text Editor (native Wayland), in Chrome, in kitty, in WezTerm (achat pane too); Alt+L live dictation with corrections (backspaces) in Text Editor; "command paste/copy"; clipboard content restored after typing; focus returns after tray-menu use.
- [ ] **Step 4: Commit + push** `docs: GNOME Wayland setup`.
