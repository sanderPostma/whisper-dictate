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
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper
import yaml

# Optional: transformers for distil-whisper models
try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Keybinder', '3.0')
try:
    gi.require_version('AppIndicator3', '0.1')
    from gi.repository import AppIndicator3 as appindicator
    HAS_APPINDICATOR = True
except:
    HAS_APPINDICATOR = False

from gi.repository import Gtk, Gdk, GLib, Keybinder, GdkPixbuf

import tempfile
import shutil

# Config
CONFIG_DIR = Path.home() / ".config" / "whisper-dictate"
CONFIG_PATH = CONFIG_DIR / "config.json"
REPLACEMENTS_PATH = CONFIG_DIR / "replacements.yml"
ICON_DIR = Path(__file__).parent / "icons"
DEFAULT_CONFIG = {
    "hotkey": "<Alt>d",
    "model": "base",
    "cpu_fallback_model": "base.en",
    "language": "en",
    "sample_rate": 16000,
    "silence_threshold": 0.01,
    "silence_gate_delay_ms": 500,
    "output_mode": "type",  # type, clipboard, or both
    "record_timeout_ms": 60000,  # auto-stop after this; hold Alt to extend
    "transcribe_chunk_seconds": 25,
    "remote_server": {
        "enabled": False,
        "host": "192.168.1.100",
        "port": 9876,
        "model": "medium.en",
        "reconnect_check_interval_sec": 120
    }
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
        self.model_menu_item = None
        self.lang_menu_item = None
        self.remote_item = None
        self.remote_available = None
        self.remote_state_lock = threading.Lock()
        self.remote_monitor_stop = threading.Event()
        self.remote_monitor_thread = None
        self.create_icons()
        model_changes = self.normalize_models_for_language()
        if model_changes:
            self.save_config(self.config)
            print(
                "[whisper-dictate] Adjusted non-English model config: "
                + ", ".join(model_changes)
            )

    def merge_config(self, loaded_config):
        """Merge loaded config with defaults, including nested dicts."""
        merged = {}
        for key, value in DEFAULT_CONFIG.items():
            if isinstance(value, dict):
                merged[key] = value.copy()
            else:
                merged[key] = value
        for key, value in loaded_config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = merged[key].copy()
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged
        
    def load_config(self):
        """Load config from file or create default."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                return self.merge_config(json.load(f))
        else:
            default_config = self.merge_config({})
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config):
        """Save config to file."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

    def get_remote_config(self):
        """Return remote config merged with defaults."""
        remote_defaults = DEFAULT_CONFIG["remote_server"].copy()
        remote_config = self.config.get("remote_server", {})
        if isinstance(remote_config, dict):
            remote_defaults.update(remote_config)
        return remote_defaults

    def is_english_only_model(self, model_name):
        """Return True for model names that should only be used for English."""
        return bool(model_name) and (
            model_name.endswith(".en") or self.is_distil_model(model_name)
        )

    def get_multilingual_model(self, model_name):
        """Return a multilingual equivalent for an English-only model."""
        if not model_name:
            return model_name
        if model_name.endswith(".en"):
            return model_name[: -len(".en")]
        if self.is_distil_model(model_name):
            return "large"
        return model_name

    def normalize_models_for_language(self):
        """Keep configured models compatible with the selected language."""
        if self.config.get("language", "en") == "en":
            return []

        changes = []
        model = self.config.get("model", "base")
        normalized_model = self.get_multilingual_model(model)
        if normalized_model != model:
            self.config["model"] = normalized_model
            changes.append(f"model {model} -> {normalized_model}")

        fallback = self.config.get("cpu_fallback_model", "base.en")
        normalized_fallback = self.get_multilingual_model(fallback)
        if normalized_fallback != fallback:
            self.config["cpu_fallback_model"] = normalized_fallback
            changes.append(f"fallback {fallback} -> {normalized_fallback}")

        remote_config = self.get_remote_config()
        remote_model = remote_config.get("model")
        normalized_remote_model = self.get_multilingual_model(remote_model)
        if normalized_remote_model != remote_model:
            remote_config["model"] = normalized_remote_model
            self.config["remote_server"] = remote_config
            changes.append(f"remote {remote_model} -> {normalized_remote_model}")

        return changes

    def get_remote_model(self):
        """Return the effective remote model for the selected language."""
        remote_config = self.get_remote_config()
        model = remote_config.get("model", self.config.get("model", "base"))
        if self.config.get("language", "en") != "en":
            return self.get_multilingual_model(model)
        return model

    def get_cpu_fallback_model(self):
        """Get local fallback model used when remote is down.

        Strips the `.en` suffix when the configured language is not English,
        so Dutch audio doesn't get transcribed by an English-only model.
        """
        model = self.config.get("cpu_fallback_model", "base.en")
        language = self.config.get("language", "en")
        if language != "en" and model.endswith(".en"):
            return model[: -len(".en")]
        return model
    
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
    
    def is_distil_model(self, model_name=None):
        """Check if model is a distil-whisper model."""
        if model_name is None:
            model_name = self.config.get("model", "base")
        return model_name.startswith("distil-")
    
    def load_model(self, model_name=None):
        """Load Whisper model (lazy loading)."""
        if model_name is None:
            model_name = self.config.get("model", "base")
        
        # Check if we need to reload (model changed)
        current_model_name = getattr(self, '_loaded_model_name', None)
        if self.model is not None and current_model_name == model_name:
            return
        
        print(f"[whisper-dictate] Loading model: {model_name}...")
        
        if self.is_distil_model(model_name):
            # Use transformers pipeline for distil models
            if not HAS_TRANSFORMERS:
                print("[whisper-dictate] ERROR: transformers not installed for distil models")
                return
            hf_model = f"distil-whisper/{model_name}"
            self.model = hf_pipeline(
                "automatic-speech-recognition",
                model=hf_model,
                chunk_length_s=30,
                device="cpu"
            )
            self._model_backend = "transformers"
        else:
            # Use openai-whisper for standard models
            self.model = whisper.load_model(model_name)
            self._model_backend = "whisper"
        
        self._loaded_model_name = model_name
        print(f"[whisper-dictate] Model loaded (backend: {self._model_backend})")
    
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
        silence_run_samples = 0
        gate_delay_ms = max(0, int(self.config.get("silence_gate_delay_ms", 500)))
        gate_delay_samples = int(self.config["sample_rate"] * gate_delay_ms / 1000)
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal silence_run_samples
            if self.recording:
                threshold = float(self.config.get("silence_threshold", 0.0))
                if threshold > 0:
                    rms = float(np.sqrt(np.mean(indata ** 2)))
                    if rms < threshold:
                        silence_run_samples += frames
                        if silence_run_samples >= gate_delay_samples:
                            return
                    else:
                        silence_run_samples = 0
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

        # Auto-stop timeout — reschedules while Alt is held
        self._record_timeout_id = None
        timeout_ms = int(self.config.get("record_timeout_ms", 60000))
        if timeout_ms > 0:
            self._record_timeout_id = GLib.timeout_add(
                timeout_ms, self._record_timeout_check
            )

    def _alt_is_held(self):
        """Return True if the Alt modifier is currently pressed."""
        try:
            display = Gdk.Display.get_default()
            seat = display.get_default_seat()
            pointer = seat.get_pointer()
            screen = display.get_default_screen()
            root = screen.get_root_window()
            pos = root.get_device_position(pointer)
            mask = pos[3] if len(pos) >= 4 else 0
            return bool(mask & Gdk.ModifierType.MOD1_MASK)
        except Exception:
            return False

    def _record_timeout_check(self):
        """Auto-stop recording unless Alt is held."""
        if not self.recording:
            return False
        if self._alt_is_held():
            print("[whisper-dictate] Record timeout reached, Alt held — extending")
            return True  # reschedule
        print("[whisper-dictate] Record timeout reached — stopping")
        self._record_timeout_id = None
        self.stop_recording()
        return False

    def stop_recording(self):
        """Stop recording and transcribe."""
        if not self.recording:
            return
        
        self.recording = False
        if getattr(self, "_record_timeout_id", None):
            try:
                GLib.source_remove(self._record_timeout_id)
            except Exception:
                pass
            self._record_timeout_id = None
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

    def _recv_exact(self, sock, size):
        """Receive exactly size bytes from a socket."""
        chunks = []
        received = 0
        while received < size:
            chunk = sock.recv(size - received)
            if not chunk:
                raise ConnectionError("Socket closed before full payload was received")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def set_remote_available(self, available, reason=None):
        """Update remote availability state and log transitions."""
        with self.remote_state_lock:
            previous = self.remote_available
            self.remote_available = bool(available)

        if previous == self.remote_available:
            return

        if self.remote_available:
            print("[whisper-dictate] Remote server is reachable again")
        else:
            msg = f"[whisper-dictate] Remote server unavailable"
            if reason:
                msg += f": {reason}"
            print(msg)
        GLib.idle_add(self.update_remote_menu_label)

    def get_remote_menu_label(self):
        """Return remote menu label with reachability indicator."""
        if not self.get_remote_config().get("enabled", False):
            icon = "⚪"
        else:
            with self.remote_state_lock:
                available = self.remote_available
            if available is True:
                icon = "🟢"
            elif available is False:
                icon = "🔴"
            else:
                icon = "⚪"
        return f"{icon} Remote Server"

    def update_remote_menu_label(self):
        """Refresh remote menu item label from current reachability state."""
        if self.remote_item:
            self.remote_item.set_label(self.get_remote_menu_label())
        return False

    def probe_remote_service(self, timeout=3):
        """Lightweight remote health check using protocol ping."""
        remote_config = self.get_remote_config()
        host = remote_config.get("host", "127.0.0.1")
        port = int(remote_config.get("port", 9876))
        header = {
            "ping": True,
            "language": self.config.get("language", "en"),
            "model": self.get_remote_model(),
            "sample_rate": self.config.get("sample_rate", 16000),
            "audio_size": 0,
        }
        header_bytes = json.dumps(header).encode("utf-8")

        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(struct.pack(">I", len(header_bytes)))
                sock.sendall(header_bytes)
                response_len = struct.unpack(">I", self._recv_exact(sock, 4))[0]
                response_data = self._recv_exact(sock, response_len)
            response = json.loads(response_data.decode("utf-8"))
            return not response.get("error")
        except Exception:
            return False

    def check_remote_service_once(self):
        """Run one remote availability check and update state."""
        self.set_remote_available(self.probe_remote_service())

    def start_remote_monitor(self):
        """Start background monitor that retries remote when it is down."""
        if self.remote_monitor_thread and self.remote_monitor_thread.is_alive():
            return
        self.remote_monitor_stop.clear()
        self.remote_monitor_thread = threading.Thread(
            target=self.remote_monitor_loop,
            daemon=True
        )
        self.remote_monitor_thread.start()

    def stop_remote_monitor(self):
        """Stop background remote monitor."""
        self.remote_monitor_stop.set()

    def remote_monitor_loop(self):
        """Check remote service periodically while remote mode is enabled."""
        while not self.remote_monitor_stop.is_set():
            remote_config = self.get_remote_config()
            if not remote_config.get("enabled", False):
                return

            with self.remote_state_lock:
                should_probe = (self.remote_available is False)

            if should_probe and self.probe_remote_service():
                self.set_remote_available(True)

            interval = max(1, int(remote_config.get("reconnect_check_interval_sec", 120)))
            if self.remote_monitor_stop.wait(interval):
                return

    def transcribe_remote(self, audio, timeout=60):
        """Transcribe audio by sending it to a remote Whisper server."""
        remote_config = self.get_remote_config()

        host = remote_config.get("host", "127.0.0.1")
        port = int(remote_config.get("port", 9876))
        audio_bytes = audio.astype(np.float32).tobytes()
        header = {
            "language": self.config.get("language", "en"),
            "model": self.get_remote_model(),
            "sample_rate": self.config.get("sample_rate", 16000),
            "audio_size": len(audio_bytes),
        }
        header_bytes = json.dumps(header).encode("utf-8")

        with socket.create_connection((host, port), timeout=5) as sock:
            sock.settimeout(timeout)
            sock.sendall(struct.pack(">I", len(header_bytes)))
            sock.sendall(header_bytes)
            sock.sendall(audio_bytes)

            response_len = struct.unpack(">I", self._recv_exact(sock, 4))[0]
            response_data = self._recv_exact(sock, response_len)

        response = json.loads(response_data.decode("utf-8"))
        if response.get("error"):
            raise RuntimeError(response["error"])

        elapsed = response.get("elapsed")
        if elapsed is not None:
            print(f"[whisper-dictate] Remote transcription took {elapsed:.2f}s")
        return response.get("text", "").strip()

    def _transcribe_local(self, audio, model_name=None):
        """Transcribe audio using the local model."""
        if model_name is None:
            model_name = self.config.get("model", "base")
        self.load_model(model_name=model_name)
        start_time = time.time()

        if getattr(self, '_model_backend', 'whisper') == 'transformers':
            result = self.model({
                "array": audio,
                "sampling_rate": self.config["sample_rate"]
            })
            text = result.get("text", "").strip()
        else:
            result = self.model.transcribe(
                audio,
                language=self.config["language"],
                fp16=False  # CPU mode
            )
            text = result["text"].strip()

        elapsed = time.time() - start_time
        print(f"[whisper-dictate] Local transcription took {elapsed:.2f}s ({model_name})")
        return text
    
    def transcribe_and_paste(self, audio):
        """Transcribe audio and paste result."""
        GLib.idle_add(lambda: self.update_status("Transcribing..."))

        try:
            remote_config = self.get_remote_config()
            remote_enabled = remote_config.get("enabled", False)
            local_fallback_model = self.get_cpu_fallback_model()
            if remote_enabled:
                with self.remote_state_lock:
                    remote_known_down = (self.remote_available is False)

                if remote_known_down:
                    print(
                        f"[whisper-dictate] Remote unavailable, using local fallback model: "
                        f"{local_fallback_model}"
                    )
                    text = self._transcribe_local(audio, model_name=local_fallback_model)
                else:
                    try:
                        text = self.transcribe_remote(audio)
                        self.set_remote_available(True)
                    except Exception as e:
                        self.set_remote_available(False, str(e))
                        # The CPU fallback model is usually tiny (base/base.en) and
                        # produces near-gibberish for non-English audio. Refuse to
                        # fall back silently in that case; notify the user instead.
                        if self.config.get("language", "en") != "en":
                            err = str(e)[:200]
                            GLib.idle_add(lambda err=err: (
                                self.notify(f"Remote failed; skipping local fallback for non-English audio.\n{err}"),
                                self.update_status("Remote failed"),
                            ) and False)
                            return
                        GLib.idle_add(lambda: self.update_status("Remote failed, using local..."))
                        text = self._transcribe_local(audio, model_name=local_fallback_model)
            else:
                text = self._transcribe_local(audio, model_name=self.config.get("model", "base"))
        except Exception as e:
            print(f"[whisper-dictate] Transcription failed: {e}")
            GLib.idle_add(lambda: self.update_status("Ready"))
            return
        
        if not text:
            GLib.idle_add(lambda: self.update_status("Ready"))
            return
        
        # Apply text replacements
        text = self.apply_replacements(text)
        
        # Output based on mode
        GLib.idle_add(lambda: self.output_text(text))
    
    def get_focused_window_class(self):
        """Return the WM_CLASS of the focused window, or None."""
        try:
            wid = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, check=True, timeout=1
            ).stdout.strip()
            if not wid:
                return None
            result = subprocess.run(
                ["xprop", "-id", wid, "WM_CLASS"],
                capture_output=True, text=True, check=True, timeout=1
            )
            # WM_CLASS(STRING) = "kitty", "kitty"
            return result.stdout.strip()
        except Exception:
            return None

    def toggle_transcribe_session(self, *args):
        """Toggle long-form transcribe session (mic + speakers, no timeout)."""
        if getattr(self, "transcribe_active", False):
            GLib.idle_add(self.stop_transcribe_session)
        else:
            GLib.idle_add(self.start_transcribe_session)

    def _get_monitor_source(self):
        """Return the pulse/pipewire monitor source name for the default sink."""
        try:
            sink = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, check=True, timeout=2
            ).stdout.strip()
            if not sink:
                return None
            return f"{sink}.monitor"
        except Exception as e:
            print(f"[transcribe-session] pactl failed: {e}")
            return None

    def start_transcribe_session(self):
        """Start mic+speakers transcription, chunked, streamed to a tempfile."""
        if getattr(self, "transcribe_active", False) or self.recording:
            return False

        monitor = self._get_monitor_source()
        if not monitor:
            self.notify("Could not find speaker monitor source (pactl).")
            return False

        self.transcribe_tempfile_path = tempfile.NamedTemporaryFile(
            mode="w", delete=False,
            prefix=f"whisper-transcribe-{time.strftime('%Y%m%d-%H%M%S')}-",
            suffix=".txt", encoding="utf-8",
        ).name
        self.transcribe_tempfile_lock = threading.Lock()
        self.transcribe_mic_buf = []
        self.transcribe_sys_buf = []
        self.transcribe_buf_lock = threading.Lock()
        self.transcribe_active = True
        self.transcribe_pending_chunks = 0
        self.transcribe_completed_chunks = 0
        self.transcribe_failed_chunks = 0
        self.transcribe_chunk_index = 0
        self.transcribe_pending_lock = threading.Lock()
        self.transcribe_finished_event = threading.Event()
        self.transcribe_chunk_queue = queue.Queue()
        sr = int(self.config.get("sample_rate", 16000))
        self.transcribe_sr = sr
        self.transcribe_worker_thread = threading.Thread(
            target=self._transcribe_session_worker,
            daemon=True
        )
        self.transcribe_worker_thread.start()

        # Mic via sounddevice
        def mic_cb(indata, frames, time_info, status):
            if not self.transcribe_active:
                return
            with self.transcribe_buf_lock:
                self.transcribe_mic_buf.append(indata.copy())
        try:
            self.transcribe_mic_stream = sd.InputStream(
                samplerate=sr, channels=1, dtype=np.float32, callback=mic_cb
            )
            self.transcribe_mic_stream.start()
        except Exception as e:
            self.notify(f"Mic open failed: {e}")
            self.transcribe_active = False
            self.transcribe_chunk_queue.put(None)
            return False

        # Speakers via parec
        try:
            self.transcribe_parec = subprocess.Popen(
                ["parec", "-d", monitor, "--raw",
                 "--format=float32le", f"--rate={sr}", "--channels=1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            self.notify("parec not installed (pulseaudio-utils).")
            self.transcribe_mic_stream.stop()
            self.transcribe_mic_stream.close()
            self.transcribe_active = False
            self.transcribe_chunk_queue.put(None)
            return False

        def parec_reader():
            bytes_per_sec = sr * 4
            while self.transcribe_active:
                data = self.transcribe_parec.stdout.read(bytes_per_sec)
                if not data:
                    break
                arr = np.frombuffer(data, dtype=np.float32).copy()
                with self.transcribe_buf_lock:
                    self.transcribe_sys_buf.append(arr)
        threading.Thread(target=parec_reader, daemon=True).start()

        # Chunker
        def chunker():
            chunk_s = int(self.config.get("transcribe_chunk_seconds", 25))
            while self.transcribe_active:
                for _ in range(chunk_s * 10):
                    if not self.transcribe_active:
                        break
                    time.sleep(0.1)
                self._flush_transcribe_chunk()
            self._flush_transcribe_chunk()
            self.transcribe_finished_event.set()
        threading.Thread(target=chunker, daemon=True).start()

        self.transcribe_session_item.set_label("🛑 Stop Transcribe")
        GLib.idle_add(lambda: self.update_status("Transcribing (mic + speakers)..."))
        GLib.idle_add(lambda: self.update_icon(True) or False)
        self.beep_start()
        self.notify(f"Transcribe session started\n{self.transcribe_tempfile_path}")
        return False

    def _flush_transcribe_chunk(self):
        """Mix buffered mic+speaker audio and dispatch async transcription."""
        with self.transcribe_buf_lock:
            mic_parts = self.transcribe_mic_buf
            sys_parts = self.transcribe_sys_buf
            self.transcribe_mic_buf = []
            self.transcribe_sys_buf = []

        mic = np.concatenate(mic_parts).flatten() if mic_parts else np.zeros(0, dtype=np.float32)
        sysa = np.concatenate(sys_parts).flatten() if sys_parts else np.zeros(0, dtype=np.float32)
        if len(mic) == 0 and len(sysa) == 0:
            return
        n = max(len(mic), len(sysa))
        if len(mic) < n:
            mic = np.pad(mic, (0, n - len(mic)))
        if len(sysa) < n:
            sysa = np.pad(sysa, (0, n - len(sysa)))
        mixed = np.clip(mic + sysa, -1.0, 1.0).astype(np.float32)

        # Skip near-silent chunks (<0.5s or very low RMS)
        if n < self.transcribe_sr // 2:
            return
        if float(np.sqrt(np.mean(mixed ** 2))) < 0.001:
            return

        with self.transcribe_pending_lock:
            self.transcribe_pending_chunks += 1
            chunk_index = self.transcribe_chunk_index
            self.transcribe_chunk_index += 1
        self.transcribe_chunk_queue.put((chunk_index, mixed))

    def _transcribe_session_audio(self, audio):
        if self.get_remote_config().get("enabled", False):
            try:
                text = self.transcribe_remote(audio, timeout=180)
                self.set_remote_available(True)
                return text
            except Exception as e:
                self.set_remote_available(False, str(e))
                return self._transcribe_local(
                    audio, model_name=self.get_cpu_fallback_model()
                )
        return self._transcribe_local(
            audio, model_name=self.config.get("model", "base")
        )

    def _transcribe_session_worker(self):
        while True:
            item = self.transcribe_chunk_queue.get()
            try:
                if item is None:
                    return
                chunk_index, audio = item
                text = (self._transcribe_session_audio(audio) or "").strip()
                if text:
                    with self.transcribe_tempfile_lock:
                        with open(self.transcribe_tempfile_path, "a", encoding="utf-8") as f:
                            f.write(text + " ")
                with self.transcribe_pending_lock:
                    self.transcribe_completed_chunks += 1
                print(f"[transcribe-session] chunk {chunk_index} +{len(text)} chars")
            except Exception as e:
                with self.transcribe_pending_lock:
                    self.transcribe_failed_chunks += 1
                print(f"[transcribe-session] chunk failed: {e}")
            finally:
                if item is not None:
                    with self.transcribe_pending_lock:
                        self.transcribe_pending_chunks -= 1
                self.transcribe_chunk_queue.task_done()

    def stop_transcribe_session(self):
        """Stop the session, wait for pending chunks, then prompt to save."""
        if not getattr(self, "transcribe_active", False):
            return False
        self.transcribe_active = False

        try:
            self.transcribe_mic_stream.stop()
            self.transcribe_mic_stream.close()
        except Exception:
            pass
        try:
            self.transcribe_parec.terminate()
            self.transcribe_parec.wait(timeout=2)
        except Exception:
            pass

        self.beep_stop()
        self.transcribe_session_item.set_label("🎙️ Transcribe...")
        GLib.idle_add(lambda: self.update_icon(False) or False)
        GLib.idle_add(lambda: self.update_status("Finalizing transcript..."))

        def finalize():
            # Wait for chunker's final flush
            self.transcribe_finished_event.wait()
            self.transcribe_chunk_queue.put(None)
            # Wait for pending transcription work; do not offer an incomplete save.
            last_pending = None
            while True:
                with self.transcribe_pending_lock:
                    pending = self.transcribe_pending_chunks
                if pending == 0:
                    break
                if pending != last_pending:
                    last_pending = pending
                    GLib.idle_add(
                        lambda pending=pending: self.update_status(
                            f"Finalizing transcript ({pending} chunks left)..."
                        ) or False
                    )
                    print(f"[transcribe-session] finalizing: {pending} chunks left")
                time.sleep(1)
            self.transcribe_chunk_queue.join()
            with self.transcribe_pending_lock:
                completed = self.transcribe_completed_chunks
                failed = self.transcribe_failed_chunks
            print(
                f"[transcribe-session] finalized: {completed} chunks completed, "
                f"{failed} failed"
            )
            if failed:
                GLib.idle_add(
                    lambda failed=failed: self.notify(
                        f"Transcript finalized with {failed} failed chunk(s)."
                    ) or False
                )
            GLib.idle_add(self._show_save_transcribe_dialog)
        threading.Thread(target=finalize, daemon=True).start()
        return False

    def _show_save_transcribe_dialog(self):
        src = self.transcribe_tempfile_path
        dialog = Gtk.FileChooserDialog(
            title="Save transcript",
            action=Gtk.FileChooserAction.SAVE,
        )
        dialog.add_buttons(
            "Discard", Gtk.ResponseType.REJECT,
            "Save", Gtk.ResponseType.OK,
        )
        dialog.set_current_name(f"transcript-{time.strftime('%Y%m%d-%H%M%S')}.txt")
        dialog.set_do_overwrite_confirmation(True)
        response = dialog.run()
        dest = dialog.get_filename() if response == Gtk.ResponseType.OK else None
        dialog.destroy()

        if dest:
            try:
                shutil.copyfile(src, dest)
                self.notify(f"Saved: {Path(dest).name}")
            except Exception as e:
                self.notify(f"Save failed: {e}")
        elif response == Gtk.ResponseType.REJECT:
            try:
                os.remove(src)
            except Exception:
                pass
            self.notify("Transcript discarded")
        else:
            self.notify(f"Transcript kept at {src}")

        self.update_status("Ready")
        return False

    def _kitty_send_text(self, text):
        """Try kitty @ send-text via remote control. Returns True on success."""
        listen_on = os.environ.get("KITTY_LISTEN_ON") or self.config.get("kitty_listen_on")
        if not listen_on:
            return False
        try:
            result = subprocess.run(
                ["kitty", "@", "--to", listen_on, "send-text",
                 "--match=recent:0", "--stdin"],
                input=text.encode(), capture_output=True, timeout=3
            )
            if result.returncode == 0:
                print("[whisper-dictate] Kitty: sent via remote control")
                return True
            print(f"[whisper-dictate] kitty @ send-text failed: "
                  f"{result.stderr.decode(errors='replace')[:200]}")
        except FileNotFoundError:
            print("[whisper-dictate] kitty binary not found on PATH")
        except Exception as e:
            print(f"[whisper-dictate] kitty remote control error: {e}")
        return False

    def _paste_via_clipboard_preserving(self, text):
        """Save clipboard, paste text via Ctrl+Shift+V, restore clipboard in background."""
        try:
            saved = subprocess.run(
                ["xclip", "-o", "-selection", "clipboard"],
                capture_output=True, timeout=1
            ).stdout
        except Exception:
            saved = None

        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode(), check=False
        )
        time.sleep(0.05)
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
            check=False
        )

        if saved is not None:
            def restore():
                time.sleep(0.3)
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=saved, check=False
                )
            threading.Thread(target=restore, daemon=True).start()

    def output_text(self, text):
        """Output text based on mode (type/clipboard/both)."""
        mode = self.config.get("output_mode", "type")

        wclass = (self.get_focused_window_class() or "").lower()
        kitty_target = "kitty" in wclass

        if mode in ("clipboard", "both"):
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode(), check=False
            )

        if mode in ("type", "both"):
            time.sleep(0.3)
            if kitty_target:
                # Kitty drops keys from xdotool type — use remote control or paste.
                if self._kitty_send_text(text):
                    pass
                elif mode == "both":
                    # Clipboard already holds our text — just paste, no restore.
                    subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
                        check=False
                    )
                else:
                    self._paste_via_clipboard_preserving(text)
            else:
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "25", "--", text],
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
    
    def _set_icon(self, name):
        if self.indicator:
            try:
                self.indicator.set_icon_full(str(ICON_DIR / f"{name}.svg"), "Whisper Dictate")
            except AttributeError:
                # Gtk.StatusIcon fallback
                self.indicator.set_from_file(str(ICON_DIR / f"{name}.svg"))

    def _blink_tick(self):
        self._blink_state = not getattr(self, "_blink_state", False)
        self._set_icon("mic-recording" if self._blink_state else "mic-idle")
        return True

    def update_icon(self, recording):
        """Update tray icon. Blinks while recording, static idle otherwise."""
        if recording:
            if getattr(self, "_blink_timer_id", None):
                return
            self._blink_state = False
            self._blink_tick()
            self._blink_timer_id = GLib.timeout_add(500, self._blink_tick)
        else:
            if getattr(self, "_blink_timer_id", None):
                try:
                    GLib.source_remove(self._blink_timer_id)
                except Exception:
                    pass
                self._blink_timer_id = None
            self._set_icon("mic-idle")
    
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

        # Long-form transcribe session (no timeout, includes speakers)
        transcribe_session_item = Gtk.MenuItem(label="🎙️ Transcribe...")
        transcribe_session_item.connect("activate", self.toggle_transcribe_session)
        menu.append(transcribe_session_item)
        self.transcribe_session_item = transcribe_session_item

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
        models = [
            "tiny.en", "tiny",
            "base.en", "base",
            "small.en", "small",
            "medium.en", "medium",
            "large", "turbo",
            "---",  # separator
            "distil-small.en",
            "distil-medium.en",
            "distil-large-v2",
            "distil-large-v3",
        ]
        group = None
        self.model_items = {}
        
        for model_name in models:
            if model_name == "---":
                # Add separator
                model_submenu.append(Gtk.SeparatorMenuItem())
                continue
            
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

        # Language submenu
        current_lang = self.config.get("language", "en")
        lang_item = Gtk.MenuItem(label=f"Language: {current_lang}")
        lang_submenu = Gtk.Menu()
        languages = [("en", "English"), ("nl", "Dutch")]
        lang_group = None
        self.lang_items = {}
        for code, label in languages:
            if lang_group is None:
                radio = Gtk.RadioMenuItem(label=f"{label} ({code})")
                lang_group = radio
            else:
                radio = Gtk.RadioMenuItem(label=f"{label} ({code})", group=lang_group)
            radio.set_active(code == current_lang)
            radio.connect("toggled", self.on_language_changed, code)
            lang_submenu.append(radio)
            self.lang_items[code] = radio
        lang_item.set_submenu(lang_submenu)
        menu.append(lang_item)
        self.lang_menu_item = lang_item

        menu.append(Gtk.SeparatorMenuItem())

        # Remote server toggle
        remote_config = self.get_remote_config()
        remote_enabled = remote_config.get("enabled", False)
        remote_item = Gtk.CheckMenuItem(label=self.get_remote_menu_label())
        remote_item.set_active(remote_enabled)
        remote_item.connect("toggled", self.on_remote_toggled)
        menu.append(remote_item)
        self.remote_item = remote_item
        
        # Transcribe file
        transcribe_file_item = Gtk.MenuItem(label="Transcribe File...")
        transcribe_file_item.connect("activate", self.show_transcribe_file_dialog)
        menu.append(transcribe_file_item)

        # Settings
        audio_settings_item = Gtk.MenuItem(label="Audio Settings...")
        audio_settings_item.connect("activate", self.show_audio_settings)
        menu.append(audio_settings_item)

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
            selected_model = model_name
            if self.config.get("language", "en") != "en":
                selected_model = self.get_multilingual_model(model_name)
            if model_name != old_model:
                self.config["model"] = selected_model
                self.normalize_models_for_language()
                self.save_config(self.config)
                self.model = None  # Force reload
                self.model_menu_item.set_label(f"Model: {self.config['model']}")
                print(f"[whisper-dictate] Model changed to: {self.config['model']}")
                if selected_model != model_name:
                    self.notify(
                        f"{model_name} is English-only; using {selected_model} for "
                        f"{self.config.get('language')}."
                    )
                # Preload new model in background
                threading.Thread(target=self.load_model, daemon=True).start()

    def on_language_changed(self, item, lang_code):
        """Handle language selection change."""
        if not item.get_active():
            return
        if lang_code == self.config.get("language"):
            return
        self.config["language"] = lang_code
        model_changes = self.normalize_models_for_language()
        self.save_config(self.config)
        self.lang_menu_item.set_label(f"Language: {lang_code}")
        if self.model_menu_item:
            self.model_menu_item.set_label(f"Model: {self.config['model']}")
        print(f"[whisper-dictate] Language changed to: {lang_code}")
        if model_changes:
            msg = "Adjusted non-English model config: " + ", ".join(model_changes)
            print(f"[whisper-dictate] {msg}")
            self.notify(msg)

    def on_remote_toggled(self, item):
        """Enable or disable remote transcription."""
        remote_config = self.get_remote_config()
        remote_config["enabled"] = item.get_active()
        self.config["remote_server"] = remote_config
        self.save_config(self.config)
        print(f"[whisper-dictate] Remote server enabled: {remote_config['enabled']}")
        if remote_config["enabled"]:
            with self.remote_state_lock:
                self.remote_available = None
            self.update_remote_menu_label()
            self.start_remote_monitor()
            threading.Thread(target=self.check_remote_service_once, daemon=True).start()
        else:
            self.stop_remote_monitor()
            with self.remote_state_lock:
                self.remote_available = None
            self.update_remote_menu_label()

    def show_audio_settings(self, *args):
        """Show dialog for calibrating noise gate with live mic level."""
        dialog = Gtk.Dialog(
            title="Audio Settings",
            transient_for=None,
            flags=0
        )
        dialog.add_button("Close", Gtk.ResponseType.CLOSE)
        dialog.set_default_size(420, 180)

        content = dialog.get_content_area()
        content.set_spacing(8)
        content.set_border_width(10)

        level_label = Gtk.Label(label="Microphone Level")
        level_label.set_xalign(0)
        content.add(level_label)

        level_bar = Gtk.LevelBar()
        level_bar.set_min_value(0.0)
        level_bar.set_max_value(1.0)
        level_bar.set_value(0.0)
        content.add(level_bar)

        threshold_label = Gtk.Label(label="Silence Threshold")
        threshold_label.set_xalign(0)
        content.add(threshold_label)

        threshold_scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL,
            0.0,
            0.2,
            0.001
        )
        threshold_scale.set_digits(3)
        threshold_scale.set_hexpand(True)
        threshold_scale.set_value(float(self.config.get("silence_threshold", 0.01)))
        content.add(threshold_scale)

        info_label = Gtk.Label()
        info_label.set_xalign(0)
        content.add(info_label)

        def on_threshold_changed(scale):
            self.config["silence_threshold"] = float(scale.get_value())
            self.save_config(self.config)

        threshold_scale.connect("value-changed", on_threshold_changed)

        monitor_state = {"rms": 0.0}
        state_lock = threading.Lock()
        gate_delay_s = max(0.0, float(self.config.get("silence_gate_delay_ms", 500)) / 1000.0)
        below_threshold_since = {"t": None}

        def monitor_callback(indata, frames, time_info, status):
            if status:
                return
            rms = float(np.sqrt(np.mean(indata ** 2)))
            with state_lock:
                monitor_state["rms"] = rms

        monitor_stream = None
        try:
            monitor_stream = sd.InputStream(
                samplerate=self.config["sample_rate"],
                channels=1,
                dtype=np.float32,
                callback=monitor_callback
            )
            monitor_stream.start()
        except Exception as e:
            info_label.set_text(f"Audio monitor unavailable: {e}")

        def refresh_info():
            with state_lock:
                rms = monitor_state["rms"]
            threshold = float(threshold_scale.get_value())
            now = time.monotonic()
            if threshold <= 0:
                gate_state = "PASS"
                below_threshold_since["t"] = None
            elif rms >= threshold:
                gate_state = "PASS"
                below_threshold_since["t"] = None
            else:
                if below_threshold_since["t"] is None:
                    below_threshold_since["t"] = now
                gate_state = (
                    "GATE"
                    if (now - below_threshold_since["t"]) >= gate_delay_s
                    else "PASS"
                )
            level_bar.set_value(min(1.0, rms * 5.0))
            info_label.set_text(
                f"RMS: {rms:.4f} | Threshold: {threshold:.4f} | {gate_state}"
            )
            return True

        timer_id = GLib.timeout_add(50, refresh_info)
        cleaned = {"done": False}

        def cleanup():
            if cleaned["done"]:
                return
            cleaned["done"] = True
            if timer_id:
                GLib.source_remove(timer_id)
            if monitor_stream:
                try:
                    monitor_stream.stop()
                    monitor_stream.close()
                except Exception:
                    pass

        dialog.connect("response", lambda d, _r: (cleanup(), d.destroy()))
        dialog.connect("destroy", lambda _d: cleanup())
        dialog.show_all()
    
    def show_transcribe_file_dialog(self, *args):
        """Open a file chooser and launch a transient systemd service to transcribe it."""
        dialog = Gtk.FileChooserDialog(
            title="Select audio/video file to transcribe",
            parent=None,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            "Cancel", Gtk.ResponseType.CANCEL,
            "Transcribe", Gtk.ResponseType.OK,
        )

        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio/Video files")
        for pat in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.opus", "*.m4a",
                    "*.mp4", "*.mkv", "*.webm", "*.aac"):
            audio_filter.add_pattern(pat)
        dialog.add_filter(audio_filter)

        all_filter = Gtk.FileFilter()
        all_filter.set_name("All files")
        all_filter.add_pattern("*")
        dialog.add_filter(all_filter)

        response = dialog.run()
        filepath = dialog.get_filename() if response == Gtk.ResponseType.OK else None
        dialog.destroy()

        if not filepath:
            return

        self.launch_file_transcription(filepath)

    def launch_file_transcription(self, filepath):
        """Transcribe filepath. Routes to remote server if enabled, else local systemd-run."""
        if self.get_remote_config().get("enabled", False):
            threading.Thread(
                target=self._transcribe_file_remote,
                args=(filepath,),
                daemon=True,
            ).start()
            self.notify(f"Transcribing {Path(filepath).name} on remote GPU...")
            return
        self._launch_file_transcription_local(filepath)

    def _decode_audio_to_float32(self, filepath, sample_rate=16000):
        """Use ffmpeg to decode any media file to mono float32 at sample_rate."""
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", filepath,
            "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    def _transcribe_file_remote(self, filepath):
        """Decode file, send to remote server, write transcript next to source."""
        try:
            audio = self._decode_audio_to_float32(
                filepath, self.config.get("sample_rate", 16000)
            )
            duration_s = len(audio) / self.config.get("sample_rate", 16000)
            # Generous timeout: allow ~5x realtime plus model-load slack
            timeout = max(300, int(duration_s * 5) + 120)
            print(f"[whisper-dictate] Remote file transcribe: "
                  f"{Path(filepath).name} ({duration_s:.1f}s audio, timeout={timeout}s)")
            text = self.transcribe_remote(audio, timeout=timeout)
        except subprocess.CalledProcessError as e:
            err = (e.stderr.decode() if e.stderr else str(e)).strip()
            GLib.idle_add(lambda: self.notify(f"ffmpeg failed: {err[:120]}") or False)
            return
        except Exception as e:
            err = str(e)
            GLib.idle_add(lambda: self.notify(f"Remote transcription failed: {err[:120]}") or False)
            self.set_remote_available(False, err)
            return

        output_txt = Path(filepath).with_suffix(".txt")
        output_txt.write_text(text + "\n", encoding="utf-8")
        GLib.idle_add(lambda: self.notify(f"Transcript saved: {output_txt.name}") or False)
        print(f"[whisper-dictate] Wrote {output_txt}")

    def _launch_file_transcription_local(self, filepath):
        """Run local whisper CLI on filepath via systemd-run --user as a transient service."""
        whisper_bin = Path(sys.executable).parent / "whisper"
        if not whisper_bin.exists():
            whisper_bin = "whisper"

        model = self.config.get("model", "base")
        language = self.config.get("language", "en")
        output_dir = str(Path(filepath).parent)
        unit_name = f"whisper-transcribe-{time.strftime('%Y%m%d-%H%M%S')}.service"

        cmd = [
            "systemd-run",
            "--user",
            f"--unit={unit_name}",
            "--description=Whisper file transcription",
            "bash", "-lc",
            'exec "$1" "$2" --model "$3" --device cpu --fp16 False '
            '--language "$4" --task transcribe --output_dir "$5" --output_format txt',
            "_",
            str(whisper_bin), filepath, model, language, output_dir,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_txt = Path(output_dir) / (Path(filepath).stem + ".txt")
            self.notify(f"Transcribing {Path(filepath).name} → {output_txt.name}")
            print(f"[whisper-dictate] Launched {unit_name}")
            print(f"[whisper-dictate]   Output: {output_txt}")
            print(f"[whisper-dictate]   Status: systemctl --user status {unit_name}")
            print(f"[whisper-dictate]   Logs:   journalctl --user -u {unit_name} -f")
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            self.notify(f"Failed to launch transcription: {err}")
            print(f"[whisper-dictate] systemd-run failed: {err}")

    def open_settings(self, *args):
        """Open config file in editor."""
        subprocess.run(["xdg-open", str(CONFIG_PATH)], check=False)
    
    def quit(self, *args):
        """Quit application."""
        self.stop_remote_monitor()
        Gtk.main_quit()
    
    def on_hotkey(self, keystring):
        """Handle global hotkey press."""
        print(f"Hotkey pressed: {keystring}")
        self.toggle_recording()
    
    def run(self):
        """Run the application."""
        mode = self.config.get("output_mode", "type")
        hotkey = self.config.get("hotkey", "<Alt>d")
        remote_cfg = self.get_remote_config()
        
        print(f"Whisper Dictate starting...")
        print(f"Config: {CONFIG_PATH}")
        print(f"Mode: {mode} | Model: {self.config['model']} | Language: {self.config['language']}")
        print(
            f"Remote: {remote_cfg.get('enabled')} | {remote_cfg.get('host')}:{remote_cfg.get('port')} "
            f"| Remote model: {self.get_remote_model()} | Local fallback: {self.get_cpu_fallback_model()}"
        )
        print(f"Hotkey: {hotkey}")
        print(f"Click tray icon or press hotkey to record")
        
        # Preload model in background
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Start remote monitor when enabled
        if self.get_remote_config().get("enabled", False):
            self.start_remote_monitor()
            threading.Thread(target=self.check_remote_service_once, daemon=True).start()
        
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
