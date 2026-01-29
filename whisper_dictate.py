#!/usr/bin/env python3
"""
Whisper Dictate - Voice-to-text with system tray integration
Click tray icon or use hotkey to toggle recording.

Usage:
    whisper_dictate.py [--mode MODE] [--hotkey HOTKEY]
    
Modes:
    type      - Type directly into active window (default)
    clipboard - Copy to clipboard only
    both      - Type and copy to clipboard
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper
import yaml

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Keybinder', '3.0')
try:
    gi.require_version('AppIndicator3', '0.1')
    from gi.repository import AppIndicator3 as appindicator
    HAS_APPINDICATOR = True
except:
    HAS_APPINDICATOR = False

from gi.repository import Gtk, GLib, Keybinder, GdkPixbuf

# Config
CONFIG_DIR = Path.home() / ".config" / "whisper-dictate"
CONFIG_PATH = CONFIG_DIR / "config.json"
REPLACEMENTS_PATH = CONFIG_DIR / "replacements.yml"
ICON_DIR = Path(__file__).parent / "icons"
DEFAULT_CONFIG = {
    "hotkey": "<Alt>d",
    "model": "base",
    "language": "en",
    "sample_rate": 16000,
    "output_mode": "type",  # type, clipboard, or both
}


class WhisperDictate:
    def __init__(self):
        self.config = self.load_config()
        self.model = None
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.indicator = None
        self.status_item = None
        self.create_icons()
        
    def load_config(self):
        """Load config from file or create default."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = DEFAULT_CONFIG.copy()
                config.update(json.load(f))
                return config
        else:
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    
    def save_config(self, config):
        """Save config to file."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    
    def load_replacements(self):
        """Load text replacements from YAML file."""
        if not REPLACEMENTS_PATH.exists():
            return {}
        try:
            with open(REPLACEMENTS_PATH) as f:
                data = yaml.safe_load(f)
                return data.get("replacements", {}) if data else {}
        except Exception as e:
            print(f"[whisper-dictate] Error loading replacements: {e}")
            return {}
    
    def apply_replacements(self, text):
        """Apply text replacements (case-insensitive matching)."""
        replacements = self.load_replacements()
        print(f"[post-process] IN:  |{text}|")
        
        if not replacements:
            print(f"[post-process] No replacements loaded")
            return text
        
        for pattern, replacement in replacements.items():
            # Determine if we should trim spaces around the replacement
            is_single_char = len(replacement) == 1
            is_escape_seq = replacement in ('\n', '\t', '\r', '\\')
            is_dot_prefix = replacement.startswith('.')
            should_trim_spaces = is_single_char or is_escape_seq or is_dot_prefix
            
            if should_trim_spaces:
                # Match pattern with optional surrounding spaces
                regex = re.compile(r'\s*' + re.escape(pattern.strip()) + r'\s*', re.IGNORECASE)
            else:
                regex = re.compile(re.escape(pattern), re.IGNORECASE)
            
            new_text = regex.sub(lambda m: replacement, text)
            if new_text != text:
                print(f"[post-process] Matched |{pattern}| -> |{replacement}| (trim={should_trim_spaces})")
            text = new_text
        
        # Remove trailing period (but keep periods between sentences)
        if text.endswith('.'):
            text = text[:-1]
            print(f"[post-process] Removed trailing period")
        
        # Lowercase single words (no spaces)
        if ' ' not in text.strip():
            text = text.lower()
            print(f"[post-process] Lowercased single word")
        
        print(f"[post-process] OUT: |{text}|")
        return text
    
    def create_icons(self):
        """Create icon files for the indicator."""
        ICON_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create simple SVG icons
        icon_idle = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="22" height="22" viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg">
  <circle cx="11" cy="8" r="4" fill="#ffffff" stroke="#888888" stroke-width="1"/>
  <rect x="9" y="11" width="4" height="4" fill="#ffffff" stroke="#888888" stroke-width="1"/>
  <path d="M 6 14 Q 6 18 11 18 Q 16 18 16 14" fill="none" stroke="#888888" stroke-width="1.5"/>
  <line x1="11" y1="18" x2="11" y2="21" stroke="#888888" stroke-width="1.5"/>
  <line x1="7" y1="21" x2="15" y2="21" stroke="#888888" stroke-width="1.5"/>
</svg>'''
        
        icon_recording = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="22" height="22" viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg">
  <circle cx="11" cy="8" r="4" fill="#ff4444" stroke="#cc0000" stroke-width="1"/>
  <rect x="9" y="11" width="4" height="4" fill="#ff4444" stroke="#cc0000" stroke-width="1"/>
  <path d="M 6 14 Q 6 18 11 18 Q 16 18 16 14" fill="none" stroke="#cc0000" stroke-width="1.5"/>
  <line x1="11" y1="18" x2="11" y2="21" stroke="#cc0000" stroke-width="1.5"/>
  <line x1="7" y1="21" x2="15" y2="21" stroke="#cc0000" stroke-width="1.5"/>
</svg>'''
        
        (ICON_DIR / "mic-idle.svg").write_text(icon_idle)
        (ICON_DIR / "mic-recording.svg").write_text(icon_recording)
    
    def load_model(self):
        """Load Whisper model (lazy loading)."""
        if self.model is None:
            print("[whisper-dictate] Loading Whisper model...")
            self.model = whisper.load_model(self.config["model"])
            print("[whisper-dictate] Model loaded")
    
    def get_focused_window(self):
        """Get currently focused window ID."""
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return None
    
    def restore_focus(self, window_id):
        """Restore focus to a window."""
        if window_id:
            try:
                subprocess.run(
                    ["xdotool", "windowactivate", window_id],
                    check=False, capture_output=True
                )
            except:
                pass
    
    def toggle_recording(self, *args):
        """Toggle recording on/off."""
        # Save focused window before any UI changes
        self.saved_window = self.get_focused_window()
        # Run in main thread via GLib
        GLib.idle_add(self._toggle_recording_impl)
    
    def _toggle_recording_impl(self):
        """Actual toggle implementation (runs in main thread)."""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
        return False
    
    def start_recording(self):
        """Start recording audio."""
        if self.recording:
            return
        
        self.recording = True
        self.audio_data = []
        self.update_icon(True)
        self.update_status("🔴 Recording...")
        
        def audio_callback(indata, frames, time_info, status):
            if self.recording:
                self.audio_data.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.config["sample_rate"],
            channels=1,
            dtype=np.float32,
            callback=audio_callback
        )
        self.stream.start()
        self.beep_start()
        # Restore focus after icon change
        GLib.timeout_add(50, lambda: self.restore_focus(getattr(self, 'saved_window', None)) or False)
    
    def stop_recording(self):
        """Stop recording and transcribe."""
        if not self.recording:
            return
        
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.beep_stop()
        self.update_icon(False)
        self.update_status("Processing...")
        # Restore focus after icon change
        GLib.timeout_add(50, lambda: self.restore_focus(getattr(self, 'saved_window', None)) or False)
        
        if not self.audio_data:
            self.update_status("Ready")
            return
        
        # Concatenate audio
        audio = np.concatenate(self.audio_data, axis=0).flatten()
        
        # Transcribe in background
        threading.Thread(target=self.transcribe_and_paste, args=(audio,), daemon=True).start()
    
    def transcribe_and_paste(self, audio):
        """Transcribe audio and paste result."""
        GLib.idle_add(lambda: self.update_status("Transcribing..."))
        
        # Ensure model is loaded
        self.load_model()
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            language=self.config["language"],
            fp16=False  # CPU mode
        )
        
        text = result["text"].strip()
        
        if not text:
            GLib.idle_add(lambda: self.update_status("Ready"))
            return
        
        # Apply text replacements
        text = self.apply_replacements(text)
        
        # Output based on mode
        GLib.idle_add(lambda: self.output_text(text))
    
    def output_text(self, text):
        """Output text based on mode (type/clipboard/both)."""
        mode = self.config.get("output_mode", "type")
        
        if mode in ("clipboard", "both"):
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode(),
                check=False
            )
        
        if mode in ("type", "both"):
            # Small delay to let user focus target window
            time.sleep(0.3)
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text],
                check=False
            )
        
        self.update_status("Ready")
        print(f"Transcribed: {text}")
    
    def beep(self, frequency=800, duration=0.1):
        """Play a short beep sound."""
        try:
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(frequency * 2 * np.pi * t) * 0.3
            # Fade in/out to avoid clicks
            fade_len = int(sample_rate * 0.01)
            tone[:fade_len] *= np.linspace(0, 1, fade_len)
            tone[-fade_len:] *= np.linspace(1, 0, fade_len)
            sd.play(tone.astype(np.float32), sample_rate, blocking=False)
        except Exception as e:
            print(f"[whisper-dictate] Beep failed: {e}")
    
    def beep_start(self):
        """Beep for recording start (higher tone)."""
        self.beep(frequency=1200, duration=0.08)
    
    def beep_stop(self):
        """Beep for recording stop (lower tone)."""
        self.beep(frequency=800, duration=0.08)
    
    def notify(self, message):
        """Show notification."""
        try:
            subprocess.run(
                ["notify-send", "-t", "2000", "-a", "Whisper Dictate", 
                 "-h", "string:x-canonical-private-synchronous:whisper-dictate",
                 "🎤 Whisper Dictate", message],
                check=False,
                capture_output=True
            )
        except Exception:
            pass
        print(f"[whisper-dictate] {message}")
    
    def update_icon(self, recording):
        """Update tray icon."""
        if self.indicator:
            icon_name = "mic-recording" if recording else "mic-idle"
            self.indicator.set_icon_full(str(ICON_DIR / f"{icon_name}.svg"), "Whisper Dictate")
    
    def update_status(self, status):
        """Update status in menu."""
        if self.status_item:
            self.status_item.set_label(f"Status: {status}")
    
    def create_menu(self):
        """Create indicator menu."""
        menu = Gtk.Menu()
        
        # Record/Stop button
        record_item = Gtk.MenuItem(label="🎤 Record/Stop")
        record_item.connect("activate", self.toggle_recording)
        menu.append(record_item)
        
        menu.append(Gtk.SeparatorMenuItem())
        
        # Status
        self.status_item = Gtk.MenuItem(label="Status: Ready")
        self.status_item.set_sensitive(False)
        menu.append(self.status_item)
        
        # Mode toggles
        mode = self.config.get("output_mode", "type")
        
        mode_type = Gtk.CheckMenuItem(label="Mode: Type")
        mode_type.set_active(mode in ("type", "both"))
        mode_type.connect("toggled", self.on_mode_type_toggled)
        menu.append(mode_type)
        self.mode_type_item = mode_type
        
        mode_clip = Gtk.CheckMenuItem(label="Mode: Clipboard")
        mode_clip.set_active(mode in ("clipboard", "both"))
        mode_clip.connect("toggled", self.on_mode_clip_toggled)
        menu.append(mode_clip)
        self.mode_clip_item = mode_clip
        
        menu.append(Gtk.SeparatorMenuItem())
        
        # Info items
        hotkey = self.config.get("hotkey", "<Alt>d")
        
        info1 = Gtk.MenuItem(label=f"Hotkey: {hotkey}")
        info1.set_sensitive(False)
        menu.append(info1)
        
        # Model submenu with radio items
        model_item = Gtk.MenuItem(label=f"Model: {self.config['model']}")
        model_submenu = Gtk.Menu()
        
        current_model = self.config.get("model", "base")
        models = ["tiny", "base", "small", "medium", "large"]
        group = None
        self.model_items = {}
        
        for model_name in models:
            if group is None:
                radio = Gtk.RadioMenuItem(label=model_name)
                group = radio
            else:
                radio = Gtk.RadioMenuItem(label=model_name, group=group)
            
            radio.set_active(model_name == current_model)
            radio.connect("toggled", self.on_model_changed, model_name)
            model_submenu.append(radio)
            self.model_items[model_name] = radio
        
        model_item.set_submenu(model_submenu)
        menu.append(model_item)
        self.model_menu_item = model_item
        
        menu.append(Gtk.SeparatorMenuItem())
        
        # Settings
        settings_item = Gtk.MenuItem(label="Settings")
        settings_item.connect("activate", self.open_settings)
        menu.append(settings_item)
        
        # Quit
        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", self.quit)
        menu.append(quit_item)
        
        menu.show_all()
        return menu
    
    def update_mode(self):
        """Update output mode based on checkbox states."""
        type_on = self.mode_type_item.get_active()
        clip_on = self.mode_clip_item.get_active()
        
        if type_on and clip_on:
            mode = "both"
        elif clip_on:
            mode = "clipboard"
        else:
            mode = "type"
        
        self.config["output_mode"] = mode
        self.save_config(self.config)
        print(f"[whisper-dictate] Mode changed to: {mode}")
    
    def on_mode_type_toggled(self, item):
        """Handle type mode toggle."""
        self.update_mode()
    
    def on_mode_clip_toggled(self, item):
        """Handle clipboard mode toggle."""
        self.update_mode()
    
    def on_model_changed(self, item, model_name):
        """Handle model selection change."""
        if item.get_active():
            old_model = self.config.get("model", "base")
            if model_name != old_model:
                self.config["model"] = model_name
                self.save_config(self.config)
                self.model = None  # Force reload
                self.model_menu_item.set_label(f"Model: {model_name}")
                print(f"[whisper-dictate] Model changed to: {model_name}")
                # Preload new model in background
                threading.Thread(target=self.load_model, daemon=True).start()
    
    def open_settings(self, *args):
        """Open config file in editor."""
        subprocess.run(["xdg-open", str(CONFIG_PATH)], check=False)
    
    def quit(self, *args):
        """Quit application."""
        Gtk.main_quit()
    
    def on_hotkey(self, keystring):
        """Handle global hotkey press."""
        print(f"Hotkey pressed: {keystring}")
        self.toggle_recording()
    
    def run(self):
        """Run the application."""
        mode = self.config.get("output_mode", "type")
        hotkey = self.config.get("hotkey", "<Alt>d")
        
        print(f"Whisper Dictate starting...")
        print(f"Config: {CONFIG_PATH}")
        print(f"Mode: {mode} | Model: {self.config['model']} | Language: {self.config['language']}")
        print(f"Hotkey: {hotkey}")
        print(f"Click tray icon or press hotkey to record")
        
        # Preload model in background
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Initialize keybinder
        Keybinder.init()
        if Keybinder.bind(hotkey, self.on_hotkey):
            print(f"✓ Hotkey {hotkey} registered")
        else:
            print(f"✗ Failed to register hotkey {hotkey}")
        
        # Create indicator
        if HAS_APPINDICATOR:
            self.indicator = appindicator.Indicator.new(
                "whisper-dictate",
                str(ICON_DIR / "mic-idle.svg"),
                appindicator.IndicatorCategory.APPLICATION_STATUS
            )
            self.indicator.set_status(appindicator.IndicatorStatus.ACTIVE)
            self.indicator.set_menu(self.create_menu())
        else:
            # Fallback: just use a status icon (deprecated but works)
            print("AppIndicator not available, using StatusIcon")
            self.indicator = Gtk.StatusIcon()
            self.indicator.set_from_file(str(ICON_DIR / "mic-idle.svg"))
            self.indicator.set_tooltip_text("Whisper Dictate")
            self.indicator.connect("activate", self.toggle_recording)
            self.indicator.connect("popup-menu", lambda icon, button, time: 
                self.create_menu().popup(None, None, None, None, button, time))
        
        print(f"✓ Started")
        
        # Run GTK main loop
        Gtk.main()
        
        # Cleanup
        Keybinder.unbind(hotkey)


def main():
    parser = argparse.ArgumentParser(
        description="Whisper Dictate - Voice to text with system tray"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["type", "clipboard", "both"],
        default=None,
        help="Output mode: type (into active window), clipboard, or both"
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper model size"
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g., en, nl, de)"
    )
    parser.add_argument(
        "--hotkey", "-k",
        default=None,
        help="Global hotkey (e.g., '<Alt>d')"
    )
    
    args = parser.parse_args()
    
    app = WhisperDictate()
    
    # Override config with command line args
    if args.mode:
        app.config["output_mode"] = args.mode
    if args.model:
        app.config["model"] = args.model
    if args.language:
        app.config["language"] = args.language
    if args.hotkey:
        app.config["hotkey"] = args.hotkey
    
    app.run()


if __name__ == "__main__":
    main()
