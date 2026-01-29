#!/usr/bin/env python3
"""
Whisper Dictate - Voice-to-text with system tray integration
Click tray icon or use hotkey to toggle recording.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pystray
import sounddevice as sd
import whisper
from PIL import Image, ImageDraw

# Config
CONFIG_PATH = Path.home() / ".config" / "whisper-dictate" / "config.json"
SOCKET_PATH = Path("/tmp/whisper-dictate.sock")
DEFAULT_CONFIG = {
    "hotkey": "ctrl+shift+d",
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
        self.audio_data = []
        self.stream = None
        self.icon = None
        
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
    
    def create_icon(self, color="white"):
        """Create a simple microphone icon."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
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
            self.notify("Ready! Click icon to record.")
    
    def toggle_recording(self, icon=None, item=None):
        """Toggle recording on/off."""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start recording audio."""
        if self.recording:
            return
        
        self.recording = True
        self.audio_data = []
        if self.icon:
            self.icon.icon = self.create_icon("red")
        
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
        self.notify("🔴 Recording... Click to stop")
    
    def stop_recording(self):
        """Stop recording and transcribe."""
        if not self.recording:
            return
        
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.icon:
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
        
        # Copy to clipboard
        self.paste_text(text)
        print(f"Transcribed: {text}")
    
    def paste_text(self, text):
        """Copy text to clipboard."""
        # Always copy to clipboard - user can paste where they want
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode(),
            check=False
        )
        self.notify(f"📋 Copied! Ctrl+V to paste")
    
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
    
    def create_menu(self):
        """Create tray menu."""
        return pystray.Menu(
            pystray.MenuItem(
                "🎤 Record/Stop",
                self.toggle_recording,
                default=True  # This makes it the default click action
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                f"Model: {self.config['model']}",
                None,
                enabled=False
            ),
            pystray.MenuItem(
                f"Language: {self.config['language']}",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", self.open_settings),
            pystray.MenuItem("Setup Hotkey", self.setup_hotkey),
            pystray.MenuItem("Quit", self.quit),
        )
    
    def open_settings(self):
        """Open config file in editor."""
        subprocess.run(["xdg-open", str(CONFIG_PATH)], check=False)
    
    def setup_hotkey(self):
        """Show instructions for hotkey setup."""
        instructions = f"""To set up global hotkey:

1. Create ~/.xbindkeysrc with:
   "{Path(__file__).parent}/toggle-recording.sh"
     Control+Shift+d

2. Run: xbindkeys

3. Add 'xbindkeys' to startup apps

Current config hotkey: {self.config['hotkey']}"""
        
        subprocess.run(
            ["notify-send", "-t", "10000", "🎤 Hotkey Setup", instructions],
            check=False
        )
        # Also create the toggle script
        self.create_toggle_script()
    
    def create_toggle_script(self):
        """Create the toggle script for xbindkeys."""
        script_path = Path(__file__).parent / "toggle-recording.sh"
        script_content = '''#!/bin/bash
# Toggle Whisper Dictate recording
# Sends signal to running instance

PIDFILE="/tmp/whisper-dictate.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill -USR1 "$PID"
        exit 0
    fi
fi

notify-send "Whisper Dictate" "Not running! Start it first."
'''
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        print(f"Created {script_path}")
    
    def quit(self):
        """Quit application."""
        # Remove PID file
        pidfile = Path("/tmp/whisper-dictate.pid")
        if pidfile.exists():
            pidfile.unlink()
        self.icon.stop()
    
    def handle_signal(self, signum, frame):
        """Handle USR1 signal to toggle recording."""
        print(f"Received signal {signum}, toggling recording")
        self.toggle_recording()
    
    def run(self):
        """Run the application."""
        import signal
        
        print(f"Whisper Dictate starting...")
        print(f"Config: {CONFIG_PATH}")
        print(f"Click tray icon to record/stop")
        
        # Write PID file for signal-based toggling
        pidfile = Path("/tmp/whisper-dictate.pid")
        pidfile.write_text(str(os.getpid()))
        
        # Set up signal handler for hotkey
        signal.signal(signal.SIGUSR1, self.handle_signal)
        
        # Create toggle script
        self.create_toggle_script()
        
        # Create and run tray icon
        self.icon = pystray.Icon(
            "whisper-dictate",
            self.create_icon(),
            "Whisper Dictate - Click to record",
            self.create_menu()
        )
        
        self.notify("Started! Click icon to record.")
        self.icon.run()


def main():
    app = WhisperDictate()
    app.run()


if __name__ == "__main__":
    main()
