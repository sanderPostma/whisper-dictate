"""Pause-based segmentation of a live audio stream (pure, no I/O)."""

from dataclasses import dataclass

import numpy as np

DEFAULT_THRESHOLD = 0.01


@dataclass
class ChunkReady:
    audio: np.ndarray
    t0: float
    t1: float
    forced: bool = False


@dataclass
class LongPause:
    t: float


class LiveSegmenter:
    """Split a stream of mono float32 blocks into speech chunks at pauses.

    feed() returns events: ChunkReady when pause_ms of silence follows speech
    (or when a chunk reaches max_chunk_s), and one LongPause when the silence
    after a chunk reaches long_pause_ms. Times are seconds since the first block.
    """

    def __init__(self, sample_rate, pause_ms=600, max_chunk_s=8.0, threshold=0.0,
                 pad_ms=150, long_pause_ms=1200, min_speech_ms=200):
        self.sample_rate = int(sample_rate)
        self.threshold = float(threshold) if threshold and threshold > 0 else DEFAULT_THRESHOLD
        self.pause_samples = int(self.sample_rate * pause_ms / 1000)
        self.long_pause_samples = int(self.sample_rate * long_pause_ms / 1000)
        self.max_samples = int(self.sample_rate * max_chunk_s)
        self.pad_samples = int(self.sample_rate * pad_ms / 1000)
        self.min_speech_samples = int(self.sample_rate * min_speech_ms / 1000)
        self._pos = 0
        self._pre = np.zeros(0, dtype=np.float32)
        self._buf = []
        self._buf_len = 0
        self._t0_sample = 0
        self._in_speech = False
        self._silence_run = 0
        self._voiced = 0
        self._since_chunk = None  # silence samples since last chunk; None = no LongPause pending

    def _keep_preroll(self, audio):
        if self.pad_samples <= 0:
            self._pre = np.zeros(0, dtype=np.float32)
        else:
            self._pre = audio[-self.pad_samples:].copy()

    def feed(self, block):
        # Copy: the audio callback's buffer is reused once the callback returns,
        # and chunks keep blocks until the pause after them.
        block = np.array(block, dtype=np.float32).reshape(-1)
        events = []
        n = block.size
        if n == 0:
            return events
        voiced = float(np.sqrt(np.mean(block ** 2))) >= self.threshold
        start = self._pos
        self._pos += n

        if not self._in_speech:
            if voiced:
                self._in_speech = True
                self._t0_sample = start - self._pre.size
                self._buf = [self._pre, block]
                self._buf_len = self._pre.size + n
                self._silence_run = 0
                self._voiced = n
                self._since_chunk = None
            else:
                self._keep_preroll(np.concatenate([self._pre, block]))
                if self._since_chunk is not None:
                    self._since_chunk += n
                    if self._since_chunk >= self.long_pause_samples:
                        events.append(LongPause(self._pos / self.sample_rate))
                        self._since_chunk = None
            return events

        self._buf.append(block)
        self._buf_len += n
        if voiced:
            self._silence_run = 0
            self._voiced += n
        else:
            self._silence_run += n

        if self._silence_run >= self.pause_samples:
            silence_run = self._silence_run
            chunk = self._emit(silence_run)
            self._in_speech = False
            if chunk is not None:
                events.append(chunk)
                self._since_chunk = silence_run
        elif self._buf_len >= self.max_samples:
            chunk = self._emit(0, forced=True)
            if chunk is not None:
                events.append(chunk)
            self._t0_sample = self._pos
            self._voiced = 0
            self._silence_run = 0
        return events

    def flush(self):
        """Emit speech still buffered; call once when the stream stops."""
        if not self._in_speech:
            return []
        self._in_speech = False
        if self._buf_len == 0:
            return []
        chunk = self._emit(self._silence_run)
        return [chunk] if chunk is not None else []

    def _emit(self, trailing_silence, forced=False):
        audio = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)
        voiced = self._voiced
        self._keep_preroll(audio)
        self._buf = []
        self._buf_len = 0
        if voiced < self.min_speech_samples:
            return None
        keep = audio.size - max(0, trailing_silence - self.pad_samples)
        t0 = self._t0_sample / self.sample_rate
        t1 = (self._t0_sample + keep) / self.sample_rate
        return ChunkReady(audio[:keep], t0, t1, forced)
