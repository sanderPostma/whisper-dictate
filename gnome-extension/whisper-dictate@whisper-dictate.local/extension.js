import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Meta from 'gi://Meta';
import Shell from 'gi://Shell';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const IFACE_XML = `
<node>
  <interface name="org.whisperdictate.Shell">
    <method name="FocusedWindow">
      <arg type="s" direction="out" name="id"/>
      <arg type="s" direction="out" name="wm_class"/>
      <arg type="s" direction="out" name="title"/>
      <arg type="u" direction="out" name="pid"/>
    </method>
    <method name="ActivateWindow">
      <arg type="s" direction="in" name="id"/>
      <arg type="b" direction="out" name="ok"/>
    </method>
    <method name="PressKeys">
      <arg type="s" direction="in" name="chord"/>
      <arg type="u" direction="in" name="repeat"/>
      <arg type="b" direction="out" name="ok"/>
    </method>
    <method name="GrabAccelerator">
      <arg type="s" direction="in" name="accel"/>
      <arg type="u" direction="out" name="action"/>
    </method>
    <method name="UngrabAccelerator">
      <arg type="u" direction="in" name="action"/>
      <arg type="b" direction="out" name="ok"/>
    </method>
    <method name="Version">
      <arg type="u" direction="out" name="version"/>
    </method>
    <signal name="AcceleratorActivated">
      <arg type="u" name="action"/>
    </signal>
    <signal name="Ready"/>
  </interface>
</node>`;

const MODIFIERS = {
    ctrl: Clutter.KEY_Control_L,
    control: Clutter.KEY_Control_L,
    shift: Clutter.KEY_Shift_L,
    alt: Clutter.KEY_Alt_L,
    super: Clutter.KEY_Super_L,
};

const REPEAT_GAP_MS = 8;

function resolveKeyval(name) {
    const sym = Clutter[`KEY_${name}`];
    if (typeof sym === 'number')
        return sym;
    if ([...name].length === 1) {
        const v = Clutter.unicode_to_keysym(name.codePointAt(0));
        if (v)
            return v;
    }
    return null;
}

function parseChord(chord) {
    const parts = chord.split('+').filter(p => p.length > 0);
    if (parts.length === 0)
        return null;
    const keyName = parts.pop();
    const mods = [];
    for (const p of parts) {
        const m = MODIFIERS[p.toLowerCase()];
        if (m === undefined)
            return null;
        mods.push(m);
    }
    let key = resolveKeyval(keyName);
    if (key === null && keyName.length > 1)
        key = resolveKeyval(keyName.toLowerCase());
    if (key === null)
        return null;
    return {mods, key};
}

export default class WhisperDictateExtension extends Extension {
    enable() {
        // action id -> {sender, accel}; watchers: sender -> bus name watch id
        this._grabs = new Map();
        this._watchers = new Map();
        this._lastTs = 0;
        this._device = Clutter.get_default_backend()
            .get_default_seat()
            .create_virtual_device(Clutter.InputDeviceType.KEYBOARD_DEVICE);

        this._acceleratorId = global.display.connect(
            'accelerator-activated', (_display, action) => {
                if (this._grabs.has(action)) {
                    this._dbus?.emit_signal('AcceleratorActivated',
                        new GLib.Variant('(u)', [action]));
                }
            });

        this._dbus = Gio.DBusExportedObject.wrapJSObject(IFACE_XML, this);
        this._dbus.export(Gio.DBus.session, '/org/whisperdictate/Shell');
        // The Shell disables extensions on screen lock, which drops every grab.
        // Tell clients we are back so they grab their accelerators again.
        this._dbus.emit_signal('Ready', null);
    }

    disable() {
        if (this._acceleratorId) {
            global.display.disconnect(this._acceleratorId);
            this._acceleratorId = 0;
        }
        for (const id of this._watchers?.values() ?? [])
            Gio.bus_unwatch_name(id);
        this._watchers?.clear();
        this._watchers = null;
        for (const action of this._grabs?.keys() ?? []) {
            try {
                global.display.ungrab_accelerator(action);
            } catch (e) {
                console.error(`whisper-dictate: ungrab failed: ${e}`);
            }
        }
        this._grabs?.clear();
        this._grabs = null;
        if (this._dbus) {
            this._dbus.unexport();
            this._dbus = null;
        }
        // Virtual devices have no explicit close API; dropping the reference
        // lets the backend remove it.
        this._device = null;
    }

    FocusedWindow() {
        const win = global.display.focus_window;
        if (!win)
            return ['', '', '', 0];
        return [
            String(win.get_id()),
            win.get_wm_class() ?? '',
            win.get_title() ?? '',
            Math.max(0, win.get_pid()),
        ];
    }

    ActivateWindow(id) {
        for (const actor of global.get_window_actors()) {
            const win = actor.meta_window;
            if (String(win.get_id()) === id) {
                Main.activateWindow(win);
                return true;
            }
        }
        return false;
    }

    PressKeys(chord, repeat) {
        const parsed = parseChord(chord);
        if (!parsed || !this._device)
            return false;
        const {mods, key} = parsed;
        const count = Math.max(1, repeat);
        const dev = this._device;
        const PRESSED = Clutter.KeyState.PRESSED;
        const RELEASED = Clutter.KeyState.RELEASED;
        // All repeats are sent synchronously, in call order, with strictly
        // increasing timestamps (REPEAT_GAP_MS apart), so consecutive
        // PressKeys calls can never interleave.
        let t = Math.max(GLib.get_monotonic_time(), this._lastTs + 1);
        const tick = () => t++;
        for (let i = 0; i < count; i++) {
            for (const m of mods)
                dev.notify_keyval(tick(), m, PRESSED);
            dev.notify_keyval(tick(), key, PRESSED);
            dev.notify_keyval(tick(), key, RELEASED);
            for (const m of [...mods].reverse())
                dev.notify_keyval(tick(), m, RELEASED);
            t += REPEAT_GAP_MS * 1000;
        }
        this._lastTs = t;
        return true;
    }

    _ungrabAction(action) {
        this._grabs.delete(action);
        try {
            return global.display.ungrab_accelerator(action);
        } catch (e) {
            console.error(`whisper-dictate: ungrab failed: ${e}`);
            return false;
        }
    }

    _releaseSender(sender) {
        const id = this._watchers?.get(sender);
        if (id !== undefined) {
            this._watchers.delete(sender);
            Gio.bus_unwatch_name(id);
        }
        if (!this._grabs)
            return;
        for (const [action, grab] of [...this._grabs]) {
            if (grab.sender === sender)
                this._ungrabAction(action);
        }
    }

    // Grabs are owned by the D-Bus client that made them: when it vanishes
    // (crash, kill -9) its grabs are released, so a restarted client can grab
    // the same accelerator again. Mutter refuses an accelerator that is
    // already grabbed, so a repeat request drops the old grab first.
    GrabAcceleratorAsync(params, invocation) {
        const [accel] = params;
        const sender = invocation.get_sender();
        let action = 0;
        try {
            for (const [old, grab] of [...this._grabs]) {
                if (grab.accel === accel)
                    this._ungrabAction(old);
            }
            action = global.display.grab_accelerator(
                accel, Meta.KeyBindingFlags.NONE);
            if (action) {
                Main.wm.allowKeybinding(
                    Meta.external_binding_name_for_action(action), Shell.ActionMode.ALL);
                this._grabs.set(action, {sender, accel});
                if (!this._watchers.has(sender)) {
                    this._watchers.set(sender, Gio.bus_watch_name_on_connection(
                        Gio.DBus.session, sender, Gio.BusNameWatcherFlags.NONE,
                        null, () => this._releaseSender(sender)));
                }
            }
        } catch (e) {
            console.error(`whisper-dictate: grab failed: ${e}`);
            action = 0;
        }
        invocation.return_value(new GLib.Variant('(u)', [action]));
    }

    UngrabAccelerator(action) {
        if (!this._grabs.has(action))
            return false;
        return this._ungrabAction(action);
    }

    Version() {
        return 2;
    }
}
