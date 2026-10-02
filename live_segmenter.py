"""Pause-based segmentation of a live audio stream (pure, no I/O)."""

from dataclasses import dataclass

import numpy as np

DEFAULT_THRESHOLD = 0.01
# How much audio before the opening window to keep once the gate was closed.
# A "t" is often quieter than the vowel and sits in the previous block.
ATTACK_LOOKBACK_MS = 40


def window_voiced(block, threshold, attack_samples):
    """True when any attack-sized slice reaches the threshold.

    Whole-block RMS hides a consonant that fills only the start or the end
    of a 50 ms callback. The gate then opens on the vowel and the consonant
    is already gone.
    """
    block = np.asarray(block, dtype=np.float32).reshape(-1)
    if block.size == 0:
        return False
    if threshold <= 0:
        return True
    win = int(attack_samples) if attack_samples else block.size
    if win <= 1 or win >= block.size:
        return float(np.sqrt(np.mean(block * block))) >= threshold
    n = block.size // win
    slices = block[:n * win].reshape(n, win)
    peak = float(np.max(np.mean(slices * slices, axis=1)))
    if peak >= threshold * threshold:
        return True
    tail = block[n * win:]
    if tail.size == 0:
        return False
    return float(np.mean(tail * tail)) >= threshold * threshold


class SilenceGate:
    """One-shot noise gate. Closes after delay_ms below threshold.

    Opens on window_voiced, and prepends the last lookback of audio so the
    consonant that did not itself cross the threshold is still recorded.
    """

    def __init__(self, sample_rate, threshold, delay_ms, attack_ms=10,
                 lookback_ms=ATTACK_LOOKBACK_MS):
        self.threshold = float(threshold)
        self.delay_samples = max(0, int(sample_rate * delay_ms / 1000))
        self.attack_samples = max(1, int(sample_rate * attack_ms / 1000))
        self.lookback_samples = max(self.attack_samples, int(sample_rate * lookback_ms / 1000))
        self._silence = 0
        self._closed = False
        self._tail = np.zeros(0, dtype=np.float32)

    def _remember(self, block):
        n = self.lookback_samples
        if block.size >= n:
            self._tail = block[-n:].copy()
        elif self._tail.size == 0:
            self._tail = block.copy()
        else:
            self._tail = np.concatenate([self._tail, block])[-n:]

    def accept(self, block):
        """Audio to keep, or None when this block falls inside a closed gate."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return None
        if self.threshold <= 0:
            return block
        if not window_voiced(block, self.threshold, self.attack_samples):
            self._silence += block.size
            self._remember(block)
            if self._silence >= self.delay_samples:
                self._closed = True
                return None
            return block
        self._silence = 0
        tail = self._tail if self._closed else None
        self._closed = False
        self._tail = np.zeros(0, dtype=np.float32)
        if tail is not None and tail.size:
            return np.concatenate([tail, block])
        return block


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
                 pad_ms=150, long_pause_ms=1200, min_speech_ms=200, attack_ms=10):
        self.sample_rate = int(sample_rate)
        self.threshold = float(threshold) if threshold and threshold > 0 else DEFAULT_THRESHOLD
        self.pause_samples = int(self.sample_rate * pause_ms / 1000)
        self.long_pause_samples = int(self.sample_rate * long_pause_ms / 1000)
        self.max_samples = int(self.sample_rate * max_chunk_s)
        self.pad_samples = int(self.sample_rate * pad_ms / 1000)
        self.min_speech_samples = int(self.sample_rate * min_speech_ms / 1000)
        self.attack_samples = max(1, int(self.sample_rate * attack_ms / 1000))
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
        voiced = window_voiced(block, self.threshold, self.attack_samples)
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
