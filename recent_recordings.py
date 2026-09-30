"""The last few one-shot recordings, kept in memory to be played back."""

import threading
import time
from collections import deque

DEFAULT_KEEP = 5
LABEL_TEXT_CHARS = 40


class Recording:
    def __init__(self, audio, sample_rate, when, live=False):
        self.audio = audio
        self.live = live  # a live-dictation chunk rather than a one-shot recording
        self.sample_rate = sample_rate
        self.when = when
        self.text = None  # None while transcribing
        self.failed = False

    def sent(self, audio, sample_rate):
        """Keep what the model got instead, so playback is what it heard."""
        self.audio = audio
        self.sample_rate = sample_rate

    @property
    def seconds(self):
        return len(self.audio) / float(self.sample_rate)

    def label(self):
        stamp = time.strftime("%H:%M:%S", time.localtime(self.when))
        if self.failed:
            text = "(transcription failed)"
        elif self.text is None:
            text = "(transcribing...)"
        elif not self.text.strip():
            text = "(nothing heard)"
        else:
            text = " ".join(self.text.split())
            if len(text) > LABEL_TEXT_CHARS:
                text = text[: LABEL_TEXT_CHARS - 1].rstrip() + "…"
            text = f"“{text}”"
        mark = "⚡ " if self.live else ""
        rate = f"{self.sample_rate / 1000:g}k"
        return f"{mark}{stamp}  {self.seconds:.1f}s  {rate}  {text}"


class RecentRecordings:
    def __init__(self, keep=DEFAULT_KEEP):
        self._items = deque(maxlen=max(1, int(keep)))
        self._lock = threading.Lock()

    def add(self, audio, sample_rate, when=None, live=False):
        recording = Recording(audio, sample_rate, time.time() if when is None else when, live=live)
        with self._lock:
            self._items.append(recording)
        return recording

    def newest_first(self):
        with self._lock:
            return list(reversed(self._items))
