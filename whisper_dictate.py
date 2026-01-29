#!/usr/bin/env python3
"""
Whisper Dictate - Voice-to-text with system tray integration
Press hotkey to record, release to transcribe and paste.
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pystray
import sounddevice as sd
import whisper
from PIL import Image, ImageDraw
from pynput import keyboard

# Config
CONFIG_PATH = Path.home() / ".config" / "whisper-dictate" / "config.json"
DEFAULT_CONFIG = {
    "hotkey": "<ctrl>+<shift>+d",
    "model": "base",
    "language": "en",
    "sample_rate": 16000,
    "paste_method": "xdotool",  # xdotool or xclip
}


class WhisperDictate:
    def __init__(self):
        self.config = self.load_config()
        self.model = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.audio_data = []
        self.icon = None
        self.hotkey_listener = None
        self.current_keys = set()
        self.hotkey_keys = self.parse_hotkey(self.config["hotkey"])
        
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
    
    def parse_hotkey(self, hotkey_str):
        """Parse hotkey string into set of keys."""
        keys = set()
        parts = hotkey_str.lower().replace(">+<", "> <").replace("<", "").replace(">", "").split()
        for part in parts:
            if part == "ctrl":
                keys.add(keyboard.Key.ctrl_l)
            elif part == "shift":
                keys.add(keyboard.Key.shift_l)
            elif part == "alt":
                keys.add(keyboard.Key.alt_l)
            elif part == "super" or part == "cmd":
                keys.add(keyboard.Key.cmd)
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
            else:
                # Try as function key
                try:
                    keys.add(getattr(keyboard.Key, part))
                except AttributeError:
                    keys.add(keyboard.KeyCode.from_char(part[0]))
        return keys
    
    def create_icon(self, color="white"):
        """Create a simple microphone icon."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Microphone body
        if color == "red":
            fill = (255, 80, 80, 255)
        else:
            fill = (255, 255, 255, 255)
        
        # Mic head (oval)
        draw.ellipse([20, 8, 44, 36], fill=fill)
        # Mic body (rectangle)
        draw.rectangle([24, 28, 40, 42], fill=fill)
        # Stand arc
        draw.arc([16, 28, 48, 52], 0, 180, fill=fill, width=3)
        # Stand line
        draw.line([32, 52, 32, 58], fill=fill, width=3)
        # Base
        draw.line([22, 58, 42, 58], fill=fill, width=3)
        
        return img
    
    def load_model(self):
        """Load Whisper model (lazy loading)."""
        if self.model is None:
            self.notify("Loading Whisper model...")
            self.model = whisper.load_model(self.config["model"])
            self.notify("Ready!")
    
    def start_recording(self):
        """Start recording audio."""
        if self.recording:
            return
        
        self.recording = True
        self.audio_data = []
        self.icon.icon = self.create_icon("red")
        
        def audio_callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.config["sample_rate"],
            channels=1,
            dtype=np.float32,
            callback=audio_callback
        )
        self.stream.start()
        self.notify("Recording...")
    
    def stop_recording(self):
        """Stop recording and transcribe."""
        if not self.recording:
            return
        
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.icon.icon = self.create_icon("white")
        
        if not self.audio_data:
            self.notify("No audio recorded")
            return
        
        # Concatenate audio
        audio = np.concatenate(self.audio_data, axis=0).flatten()
        
        # Transcribe in background
        threading.Thread(target=self.transcribe_and_paste, args=(audio,), daemon=True).start()
    
    def transcribe_and_paste(self, audio):
        """Transcribe audio and paste result."""
        self.notify("Transcribing...")
        
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
            self.notify("No speech detected")
            return
        
        # Paste
        self.paste_text(text)
        self.notify(f"✓ {text[:50]}{'...' if len(text) > 50 else ''}")
    
    def paste_text(self, text):
        """Paste text to active window."""
        if self.config["paste_method"] == "xdotool":
            # Use xdotool to type (handles special chars better)
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--", text],
                check=False
            )
        else:
            # Use xclip + paste
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode(),
                check=False
            )
            # Small delay then paste
            time.sleep(0.1)
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
                check=False
            )
    
    def notify(self, message):
        """Show notification."""
        try:
            subprocess.run(
                ["notify-send", "-t", "2000", "-a", "Whisper Dictate", "🎤 Whisper Dictate", message],
                check=False,
                capture_output=True
            )
        except Exception:
            pass
        print(f"[whisper-dictate] {message}")
    
    def on_key_press(self, key):
        """Handle key press."""
        # Normalize key
        if hasattr(key, 'char') and key.char:
            self.current_keys.add(keyboard.KeyCode.from_char(key.char.lower()))
        else:
            self.current_keys.add(key)
            # Also add generic versions for ctrl/shift/alt
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self.current_keys.add(keyboard.Key.ctrl_l)
            elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
                self.current_keys.add(keyboard.Key.shift_l)
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                self.current_keys.add(keyboard.Key.alt_l)
        
        # Check if hotkey pressed
        if self.hotkey_keys.issubset(self.current_keys):
            if not self.recording:
                self.start_recording()
    
    def on_key_release(self, key):
        """Handle key release."""
        # If recording and any hotkey key released, stop
        if self.recording:
            released_key = key
            if hasattr(key, 'char') and key.char:
                released_key = keyboard.KeyCode.from_char(key.char.lower())
            
            # Check if released key is part of hotkey
            is_hotkey_part = False
            if released_key in self.hotkey_keys:
                is_hotkey_part = True
            elif key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r) and keyboard.Key.ctrl_l in self.hotkey_keys:
                is_hotkey_part = True
            elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r) and keyboard.Key.shift_l in self.hotkey_keys:
                is_hotkey_part = True
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r) and keyboard.Key.alt_l in self.hotkey_keys:
                is_hotkey_part = True
            
            if is_hotkey_part:
                self.stop_recording()
        
        # Clear released key
        self.current_keys.discard(key)
        if hasattr(key, 'char') and key.char:
            self.current_keys.discard(keyboard.KeyCode.from_char(key.char.lower()))
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.current_keys.discard(keyboard.Key.ctrl_l)
            self.current_keys.discard(keyboard.Key.ctrl_r)
        elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.current_keys.discard(keyboard.Key.shift_l)
            self.current_keys.discard(keyboard.Key.shift_r)
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self.current_keys.discard(keyboard.Key.alt_l)
            self.current_keys.discard(keyboard.Key.alt_r)
    
    def create_menu(self):
        """Create tray menu."""
        return pystray.Menu(
            pystray.MenuItem(
                f"Hotkey: {self.config['hotkey']}",
                None,
                enabled=False
            ),
            pystray.MenuItem(
                f"Model: {self.config['model']}",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", self.open_settings),
            pystray.MenuItem("Quit", self.quit),
        )
    
    def open_settings(self):
        """Open config file in editor."""
        subprocess.run(["xdg-open", str(CONFIG_PATH)], check=False)
    
    def quit(self):
        """Quit application."""
        self.icon.stop()
    
    def run(self):
        """Run the application."""
        print(f"Whisper Dictate starting...")
        print(f"Config: {CONFIG_PATH}")
        print(f"Hotkey: {self.config['hotkey']} (hold to record)")
        
        # Start keyboard listener
        self.hotkey_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.hotkey_listener.start()
        
        # Create and run tray icon
        self.icon = pystray.Icon(
            "whisper-dictate",
            self.create_icon(),
            "Whisper Dictate",
            self.create_menu()
        )
        
        self.notify("Started! Hold hotkey to record.")
        self.icon.run()


def main():
    app = WhisperDictate()
    app.run()


if __name__ == "__main__":
    main()
