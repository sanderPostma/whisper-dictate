"""Report when a Microsoft Teams call starts, by watching PipeWire.

Teams captures the microphone for the length of a call, which PipeWire shows
as a new "source-output". `pactl subscribe` announces that without polling;
the capturing app is then read from `pactl list source-outputs`.
"""

import subprocess
import threading

_NEW_SOURCE_OUTPUT = "on source-output"
_APP_KEYS = ("application.name", "application.process.binary")


def is_source_output_added(event_line):
    """True for a `pactl subscribe` line reporting a new capture stream."""
    return "'new'" in event_line and _NEW_SOURCE_OUTPUT in event_line


def teams_is_capturing(listing):
    """True if any capturing app in `pactl list source-outputs` is Teams."""
    for line in listing.splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() in _APP_KEYS and "teams" in value.strip().strip('"').lower():
            return True
    return False


class TeamsCallWatcher:
    """Calls on_call() each time a new capture stream appears while Teams holds the mic."""

    def __init__(self, on_call, log=print):
        self.on_call = on_call
        self.log = log
        self._proc = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    def _run(self):
        try:
            self._proc = subprocess.Popen(
                ["pactl", "subscribe"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError as e:
            self.log(f"[teams] call watcher disabled: cannot run pactl: {e}")
            return
        for line in self._proc.stdout:
            if not is_source_output_added(line):
                continue
            try:
                listing = subprocess.run(
                    ["pactl", "list", "source-outputs"],
                    capture_output=True, text=True, timeout=2,
                ).stdout
            except (OSError, subprocess.SubprocessError) as e:
                self.log(f"[teams] cannot list source-outputs: {e}")
                continue
            if teams_is_capturing(listing):
                self.log("[teams] Teams call detected")
                self.on_call()
