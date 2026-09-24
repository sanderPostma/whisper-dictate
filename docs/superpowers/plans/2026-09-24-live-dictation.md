# Live Dictation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live dictation mode that types text at each speech pause, and uses a bounded audio window to correct recently typed words.

**Architecture:** Four new pure or near-pure modules:
- `live_segmenter.py` detects pauses.
- `live_session.py` keeps the window and committed text and computes diffs.
- `live_output.py` sends edits through WezTerm or xdotool.
- `live_controller.py` is a single worker thread. For each chunk it runs a fast pass, and when the queue is idle it runs a correction.

`whisper_dictate.py` only wires the pieces together: a hotkey, a tray item, the audio stream and the transcriber callables. One-shot mode is unchanged.

**Tech Stack:** Python 3, numpy, sounddevice, GTK/GLib (existing), `wezterm cli`, `xdotool`. Tests use `unittest`, run with `venv/bin/python -m unittest` (there is no pytest in the venv).

**Spec:** `docs/superpowers/specs/2026-09-24-live-dictation-design.md`

**Deliberate deviation from spec:** the spec puts output on a separate output queue. This plan applies edits in the same single worker thread, right after each transcription. That keeps edits strictly in order and keeps the code deterministic. The cost is small: typing takes 0.1 to 0.4 s, and it delays the next transcription by that much. Also, a pending correction is not queued. It simply runs whenever the job queue is empty after a chunk, which gives the same effect as "a newer correction replaces an older one".

## Global Constraints

- Config defaults: `live_pause_ms` 600, `live_max_chunk_s` 8, `live_window_max_s` 12, `live_commit_pause_ms` 1200, `live_max_backspace` 80, `live_corrections` true, `live_committed_context_chars` 400. Also add a new `live_hotkey`, default `"<Alt><Shift>d"`.
- The server protocol is unchanged. Each call is an ordinary request with a `prompt`.
- One-shot mode (`toggle_recording`) and live mode are mutually exclusive. One-shot behaviour must not change otherwise.
- Kitty and clipboard modes are not supported in live mode. Anything that isn't WezTerm uses xdotool.
- Corrections run only when the remote server is enabled and up. Local fallback does fast passes only.
- Committed text is never edited again.
- Backspace is `\x7f` for WezTerm, and `xdotool key BackSpace` everywhere else.
- Branch is `live-dictation`. Never create git worktrees.

## Review Focus

1. **Talking non-stop for more than 8 s:** the forced split must not lose or duplicate audio. The next chunk starts exactly where the forced one ended. (Task 1 test `test_forced_split_is_contiguous`.)
2. **Focus switches to another window mid-session:** no backspaces or corrections may go to the new window. The window is committed and typing continues append-only in the new target. (Task 4 test `test_focus_change_commits_and_retargets`.)
3. **Correction returns empty or just punctuation** (noise, a hiccup): typed text must not be deleted. (Task 2 test `test_empty_correction_keeps_text`.)
4. **A fast pass fails** (timeout, server blip): the words for that chunk still appear after the next correction, and the user is notified once, not once per chunk. (Task 4 test `test_fast_failure_recovered_by_correction`.)
5. **A correction would rewrite more than 80 characters:** it is skipped rather than wiping a long line. (Task 2 test `test_backspace_cap`.)

---

### Task 1: Pause segmenter

**Files:**
- Create: `live_segmenter.py`
- Test: `tests/test_live_segmenter.py`

**Interfaces:**
- Produces:
  - `ChunkReady(audio: np.ndarray, t0: float, t1: float, forced: bool = False)`
  - `LongPause(t: float)`
  - `LiveSegmenter(sample_rate, pause_ms=600, max_chunk_s=8.0, threshold=0.0, pad_ms=150, long_pause_ms=1200, min_speech_ms=200)`, with `.threshold`, `.feed(block) -> list` and `.flush() -> list`.
  - Times are seconds since the stream started.

- [ ] **Step 1: Write the failing tests**

`tests/test_live_segmenter.py`:
```python
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_segmenter import ChunkReady, LiveSegmenter, LongPause

SR = 16000
BLOCK = 800  # 50 ms


def tone(seconds, amp=0.1):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def feed_all(seg, audio):
    events = []
    for i in range(0, len(audio), BLOCK):
        events.extend(seg.feed(audio[i:i + BLOCK]))
    return events


class LiveSegmenterTests(unittest.TestCase):
    def test_pause_emits_chunk_with_trimmed_trailing_silence(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(1.0), silence(0.7)]))
        chunks = [e for e in events if isinstance(e, ChunkReady)]
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].t0, 0.0)
        self.assertAlmostEqual(chunks[0].t1, 1.15)
        self.assertEqual(len(chunks[0].audio), int(1.15 * SR))
        self.assertFalse(chunks[0].forced)

    def test_short_pause_does_not_split(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(1.0), silence(0.5), tone(0.5)]))
        self.assertEqual([e for e in events if isinstance(e, ChunkReady)], [])

    def test_leading_preroll_is_kept(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([silence(1.0), tone(1.0), silence(0.7)]))
        chunk = [e for e in events if isinstance(e, ChunkReady)][0]
        self.assertAlmostEqual(chunk.t0, 0.85)

    def test_long_pause_fires_once_after_chunk(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(1.0), silence(3.0)]))
        kinds = [type(e) for e in events]
        self.assertEqual(kinds, [ChunkReady, LongPause])

    def test_forced_split_is_contiguous(self):
        seg = LiveSegmenter(SR, max_chunk_s=8.0)
        events = feed_all(seg, tone(9.0))
        events += seg.flush()
        chunks = [e for e in events if isinstance(e, ChunkReady)]
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].forced)
        self.assertAlmostEqual(chunks[0].t1, 8.0)
        self.assertAlmostEqual(chunks[1].t0, chunks[0].t1)
        self.assertEqual(len(chunks[0].audio) + len(chunks[1].audio), 9 * SR)

    def test_zero_threshold_uses_default(self):
        seg = LiveSegmenter(SR, threshold=0.0)
        self.assertEqual(seg.threshold, 0.01)
        events = feed_all(seg, np.concatenate([tone(1.0, amp=0.001), silence(1.0)]))
        self.assertEqual(events, [])

    def test_blip_shorter_than_min_speech_is_dropped(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(0.1), silence(0.7)]))
        self.assertEqual([e for e in events if isinstance(e, ChunkReady)], [])

    def test_flush_without_speech_is_empty(self):
        seg = LiveSegmenter(SR)
        feed_all(seg, silence(1.0))
        self.assertEqual(seg.flush(), [])

    def test_flush_emits_pending_speech(self):
        seg = LiveSegmenter(SR)
        feed_all(seg, tone(1.0))
        chunks = seg.flush()
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].t1, 1.0)
        self.assertEqual(seg.flush(), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `venv/bin/python -m unittest tests.test_live_segmenter -v`
Expected: ERROR `ModuleNotFoundError: No module named 'live_segmenter'`

- [ ] **Step 3: Implement**

`live_segmenter.py`:
```python
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
    """Split a stream of audio blocks into speech chunks at pauses.

    feed() takes mono float32 blocks and returns events: ChunkReady when a pause
    of pause_ms follows speech (or when a chunk reaches max_chunk_s), and one
    LongPause when the silence after a chunk reaches long_pause_ms.
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
        self._since_chunk = None  # silence samples since last chunk, None = no LongPause pending

    def _keep_preroll(self, audio):
        if self.pad_samples <= 0:
            self._pre = np.zeros(0, dtype=np.float32)
        else:
            self._pre = audio[-self.pad_samples:].copy()

    def feed(self, block):
        block = np.asarray(block, dtype=np.float32).reshape(-1)
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
            self._buf = []
            self._buf_len = 0
            self._voiced = 0
            self._silence_run = 0
        return events

    def flush(self):
        """Emit any speech still buffered (call when the stream stops)."""
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `venv/bin/python -m unittest tests.test_live_segmenter -v`
Expected: 9 tests OK.

- [ ] **Step 5: Commit**

```bash
git add live_segmenter.py tests/test_live_segmenter.py
git commit -m "Add pause segmenter for live dictation"
```

---

### Task 2: Live session bookkeeping

**Files:**
- Create: `live_session.py`
- Test: `tests/test_live_session.py`

**Interfaces:**
- Produces:
  - `Edit(backspace: int, insert: str)`, frozen.
  - `strip_echo(raw, context) -> str`.
  - `normalise(raw, before) -> str`.
  - `LiveSession(base_prompt="", context_chars=400, max_backspace=80, window_max_s=12.0)` with:
    - Attributes: `.committed_text`, `.generation`, `.typed_window` (property), `.chunk_count`, `.piece_count`.
    - `.window_seconds() -> float`
    - `.add_chunk(audio, t0, t1)`
    - `.add_fast_result(t1, raw) -> Edit | None`
    - `.window_audio() -> (np.ndarray, t_upto: float, generation: int) | None`
    - `.apply_correction(t_upto, raw, generation) -> Edit | None`
    - `.should_commit(long_pause=False, focus_ok=True, incoming_s=0.0) -> bool`
    - `.commit()`, `.fast_prompt() -> str`, `.correction_prompt() -> str`
- Note: `apply_correction` takes a `generation` argument, which the spec didn't have. The window may have been committed while the correction was running, and this is how a stale correction is detected.

- [ ] **Step 1: Write the failing tests**

`tests/test_live_session.py`:
```python
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_session import Edit, LiveSession, normalise, strip_echo


def audio(seconds):
    return np.zeros(int(16000 * seconds), dtype=np.float32)


def session_with(pieces, **kwargs):
    """pieces: list of (t0, t1, raw) fed as chunk + fast result."""
    s = LiveSession(**kwargs)
    for t0, t1, raw in pieces:
        s.add_chunk(audio(t1 - t0), t0, t1)
        s.add_fast_result(t1, raw)
    return s


class NormaliseTests(unittest.TestCase):
    def test_first_text_unchanged(self):
        self.assertEqual(normalise("Hello world", ""), "Hello world")

    def test_continuation_lowercases_and_spaces(self):
        self.assertEqual(normalise("The branch", "merge"), " the branch")

    def test_keeps_pronoun_i_and_acronyms(self):
        self.assertEqual(normalise("I think", "so"), " I think")
        self.assertEqual(normalise("API calls", "the"), " API calls")

    def test_after_sentence_end_keeps_capital(self):
        self.assertEqual(normalise("Next one", "Done."), " Next one")

    def test_after_newline_or_space_no_extra_space(self):
        self.assertEqual(normalise("Next", "line\n"), "Next")
        self.assertEqual(normalise("next", "word "), "next")

    def test_empty_and_punctuation_only_are_dropped(self):
        self.assertEqual(normalise("", "x"), "")
        self.assertEqual(normalise("  . ", "x"), "")

    def test_collapses_whitespace(self):
        self.assertEqual(normalise("a   b\n c", ""), "a b c")

    def test_strip_echo_removes_repeated_context(self):
        self.assertEqual(
            strip_echo("merge the branch today please", "we should merge the branch"),
            "today please",
        )

    def test_strip_echo_ignores_short_overlap(self):
        self.assertEqual(strip_echo("the branch", "merge the branch"), "the branch")


class FastResultTests(unittest.TestCase):
    def test_fast_result_is_append_edit(self):
        s = LiveSession()
        s.add_chunk(audio(1), 0.0, 1.0)
        self.assertEqual(s.add_fast_result(1.0, "Hello there"), Edit(0, "Hello there"))
        self.assertEqual(s.typed_window, "Hello there")

    def test_empty_fast_result_is_none(self):
        s = LiveSession()
        s.add_chunk(audio(1), 0.0, 1.0)
        self.assertIsNone(s.add_fast_result(1.0, "  "))
        self.assertEqual(s.piece_count, 0)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.s = session_with([
            (0.0, 1.0, "We should merge the"),
            (1.0, 2.0, "brunch"),
            (2.0, 3.0, "today"),
        ])
        self.assertEqual(self.s.typed_window, "We should merge the brunch today")

    def test_correction_rewrites_only_differing_tail(self):
        gen = self.s.generation
        edit = self.s.apply_correction(2.0, "We should merge the branch", gen)
        self.assertEqual(edit, Edit(10, "anch today"))
        self.assertEqual(self.s.typed_window, "We should merge the branch today")

    def test_identical_correction_is_none(self):
        self.assertIsNone(self.s.apply_correction(3.0, "We should merge the brunch today", self.s.generation))

    def test_empty_correction_keeps_text(self):
        self.assertIsNone(self.s.apply_correction(3.0, " . ", self.s.generation))
        self.assertEqual(self.s.typed_window, "We should merge the brunch today")

    def test_stale_generation_is_none(self):
        gen = self.s.generation
        self.s.commit()
        self.assertIsNone(self.s.apply_correction(3.0, "Anything else", gen))

    def test_backspace_cap(self):
        s = session_with([(0.0, 1.0, "We should merge the"), (1.0, 2.0, "brunch")], max_backspace=3)
        self.assertIsNone(s.apply_correction(2.0, "We should merge the branch", s.generation))
        self.assertEqual(s.typed_window, "We should merge the brunch")

    def test_correction_recovers_missing_fast_result(self):
        s = session_with([(0.0, 1.0, "Hello there")])
        s.add_chunk(audio(1), 1.0, 2.0)  # fast pass failed: no piece
        edit = s.apply_correction(2.0, "Hello there general Kenobi", s.generation)
        self.assertEqual(edit, Edit(0, " general Kenobi"))

    def test_window_audio_covers_all_chunks(self):
        audio_all, t_upto, gen = self.s.window_audio()
        self.assertEqual(len(audio_all), 3 * 16000)
        self.assertEqual(t_upto, 3.0)
        self.assertEqual(gen, self.s.generation)

    def test_window_audio_empty_is_none(self):
        self.assertIsNone(LiveSession().window_audio())


class CommitTests(unittest.TestCase):
    def test_long_pause_after_sentence_commits(self):
        s = session_with([(0.0, 1.0, "Done.")])
        self.assertTrue(s.should_commit(long_pause=True))

    def test_long_pause_mid_sentence_does_not_commit(self):
        s = session_with([(0.0, 1.0, "and then")])
        self.assertFalse(s.should_commit(long_pause=True))

    def test_window_limit_commits(self):
        s = session_with([(0.0, 8.0, "a"), (8.0, 11.0, "b")], window_max_s=12.0)
        self.assertFalse(s.should_commit(incoming_s=1.0))
        self.assertTrue(s.should_commit(incoming_s=2.0))

    def test_focus_loss_commits(self):
        s = session_with([(0.0, 1.0, "a")])
        self.assertTrue(s.should_commit(focus_ok=False))

    def test_empty_window_never_commits(self):
        self.assertFalse(LiveSession().should_commit(long_pause=True, focus_ok=False))

    def test_commit_moves_text_and_bumps_generation(self):
        s = session_with([(0.0, 1.0, "Hello there")], context_chars=5)
        gen = s.generation
        s.commit()
        self.assertEqual(s.committed_text, "there")
        self.assertEqual(s.typed_window, "")
        self.assertEqual(s.chunk_count, 0)
        self.assertEqual(s.generation, gen + 1)
        s.add_chunk(audio(1), 1.0, 2.0)
        self.assertEqual(s.add_fast_result(2.0, "General"), Edit(0, " general"))


class PromptTests(unittest.TestCase):
    def test_fast_prompt_has_base_committed_and_window(self):
        s = LiveSession(base_prompt="Vocabulary: git")
        s.add_chunk(audio(1), 0.0, 1.0)
        s.add_fast_result(1.0, "One.")
        s.commit()
        s.add_chunk(audio(1), 1.0, 2.0)
        s.add_fast_result(2.0, "Two")
        self.assertEqual(s.fast_prompt(), "Vocabulary: git\nOne. Two")
        self.assertEqual(s.correction_prompt(), "Vocabulary: git\nOne.")

    def test_prompt_tail_is_trimmed(self):
        s = LiveSession(context_chars=4)
        s.add_chunk(audio(1), 0.0, 1.0)
        s.add_fast_result(1.0, "abcdefgh")
        self.assertEqual(s.fast_prompt(), "efgh")

    def test_no_base_no_text_is_empty(self):
        self.assertEqual(LiveSession().fast_prompt(), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `venv/bin/python -m unittest tests.test_live_session -v`
Expected: ERROR `ModuleNotFoundError: No module named 'live_session'`

- [ ] **Step 3: Implement**

`live_session.py`:
```python
"""Window bookkeeping for live dictation (pure, deterministic).

The window is the audio and text since the last commit. Text typed for the
window may be rewritten by a correction; committed text is only used as
context for the next transcriptions.
"""

from dataclasses import dataclass

import numpy as np

SENTENCE_END = (".", "?", "!")
_KEEP_CASE = {"I", "I'm", "I'll", "I've", "I'd"}


@dataclass(frozen=True)
class Edit:
    backspace: int
    insert: str


def strip_echo(raw, context, max_words=8, min_words=3):
    """Drop a repeat of the context's last words from the start of raw."""
    words = context.split()
    low = raw.lower()
    for n in range(min(max_words, len(words)), min_words - 1, -1):
        tail = " ".join(words[-n:]).lower()
        if low.startswith(tail):
            return raw[len(tail):].lstrip(" ,")
    return raw


def normalise(raw, before):
    """Turn a raw transcription into the text to type after `before`."""
    text = " ".join((raw or "").split())
    text = strip_echo(text, before).strip()
    if not any(ch.isalnum() for ch in text):
        return ""
    stripped = before.rstrip()
    if stripped and not stripped.endswith(SENTENCE_END) and not before.endswith("\n"):
        first = text.split(" ", 1)[0].rstrip(",.?!;:")
        acronym = len(first) > 1 and first.isupper()
        if first not in _KEEP_CASE and not acronym and first[:1].isupper():
            text = text[0].lower() + text[1:]
    if before and not before.endswith((" ", "\n")):
        text = " " + text
    return text


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class LiveSession:
    def __init__(self, base_prompt="", context_chars=400, max_backspace=80, window_max_s=12.0):
        self.base_prompt = base_prompt or ""
        self.context_chars = int(context_chars)
        self.max_backspace = int(max_backspace)
        self.window_max_s = float(window_max_s)
        self.committed_text = ""
        self.generation = 0
        self._chunks = []  # (audio, t0, t1)
        self._pieces = []  # (t1, typed text)

    @property
    def typed_window(self):
        return "".join(text for _, text in self._pieces)

    @property
    def chunk_count(self):
        return len(self._chunks)

    @property
    def piece_count(self):
        return len(self._pieces)

    def window_seconds(self):
        return sum(t1 - t0 for _, t0, t1 in self._chunks)

    def add_chunk(self, audio, t0, t1):
        self._chunks.append((audio, t0, t1))

    def add_fast_result(self, t1, raw):
        text = normalise(raw, self.committed_text + self.typed_window)
        if not text:
            return None
        self._pieces.append((t1, text))
        return Edit(0, text)

    def window_audio(self):
        if not self._chunks:
            return None
        audio = np.concatenate([a for a, _, _ in self._chunks])
        return audio, self._chunks[-1][2], self.generation

    def apply_correction(self, t_upto, raw, generation):
        if generation != self.generation:
            return None
        new = normalise(raw, self.committed_text)
        if not new:
            return None
        covered = [p for p in self._pieces if p[0] <= t_upto]
        rest = [p for p in self._pieces if p[0] > t_upto]
        old = "".join(text for _, text in covered)
        if new == old:
            return None
        prefix = _common_prefix(old, new)
        backspace = len(self.typed_window) - prefix
        if backspace > self.max_backspace:
            return None
        insert = new[prefix:] + "".join(text for _, text in rest)
        self._pieces = [(t_upto, new)] + rest
        return Edit(backspace, insert)

    def should_commit(self, long_pause=False, focus_ok=True, incoming_s=0.0):
        if not self._chunks:
            return False
        if not focus_ok:
            return True
        if self.window_seconds() + incoming_s > self.window_max_s:
            return True
        return bool(long_pause and self.typed_window.rstrip().endswith(SENTENCE_END))

    def commit(self):
        text = self.committed_text + self.typed_window
        self.committed_text = text[-max(self.context_chars, 1):]
        self._chunks = []
        self._pieces = []
        self.generation += 1

    def _prompt(self, text):
        tail = text[-self.context_chars:].strip() if self.context_chars > 0 else ""
        return "\n".join(p for p in (self.base_prompt.strip(), tail) if p)

    def fast_prompt(self):
        return self._prompt(self.committed_text + self.typed_window)

    def correction_prompt(self):
        return self._prompt(self.committed_text)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `venv/bin/python -m unittest tests.test_live_session -v`
Expected: all tests OK.

- [ ] **Step 5: Commit**

```bash
git add live_session.py tests/test_live_session.py
git commit -m "Add live dictation session bookkeeping and correction diff"
```

---

### Task 3: Output targets

**Files:**
- Create: `live_output.py`
- Test: `tests/test_live_output.py`

**Interfaces:**
- Consumes: `Edit` from `live_session`.
- Produces:
  - `WezTermTarget(pane_id, run=subprocess.run)` and `XdotoolTarget(window_id, run=subprocess.run)`. Each has `.name`, `.send(edit) -> bool`, `.still_focused() -> bool` and the classmethod `.detect(run) -> target | None`.
  - `wezterm_focused_pane(run) -> int | None`
  - `choose_target(wm_class, run=subprocess.run) -> target | None`
  - `run` has the same signature as `subprocess.run` and must return an object with `.returncode` and `.stdout`.

- [ ] **Step 1: Write the failing tests**

`tests/test_live_output.py`:
```python
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_output import WezTermTarget, XdotoolTarget, choose_target, wezterm_focused_pane
from live_session import Edit


class FakeRun:
    """Records commands; answers by the command's first two words."""

    def __init__(self, answers=None, fail=False):
        self.calls = []
        self.answers = answers or {}
        self.fail = fail

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        out = self.answers.get(tuple(cmd[:2]), "")
        if callable(out):
            out = out()
        return SimpleNamespace(returncode=1 if self.fail else 0, stdout=out)


def clients(*entries):
    return json.dumps([
        {"focused_pane_id": pane, "idle_time": {"secs": secs, "nanos": 0}}
        for pane, secs in entries
    ])


class WezTermTests(unittest.TestCase):
    def test_focused_pane_picks_most_recently_active_client(self):
        run = FakeRun({("wezterm", "cli"): clients((3, 50), (7, 1))})
        self.assertEqual(wezterm_focused_pane(run), 7)

    def test_focused_pane_bad_json_is_none(self):
        self.assertIsNone(wezterm_focused_pane(FakeRun({("wezterm", "cli"): "nope"})))

    def test_send_backspaces_and_insert_via_stdin(self):
        run = FakeRun()
        ok = WezTermTarget(7, run).send(Edit(3, "abc"))
        self.assertTrue(ok)
        cmd, kwargs = run.calls[0]
        self.assertEqual(cmd, ["wezterm", "cli", "send-text", "--pane-id", "7", "--no-paste"])
        self.assertEqual(kwargs["input"], b"\x7f\x7f\x7fabc")

    def test_send_failure_returns_false(self):
        self.assertFalse(WezTermTarget(7, FakeRun(fail=True)).send(Edit(0, "x")))

    def test_empty_edit_runs_nothing(self):
        run = FakeRun()
        self.assertTrue(WezTermTarget(7, run).send(Edit(0, "")))
        self.assertEqual(run.calls, [])

    def test_still_focused_compares_pane(self):
        run = FakeRun({("wezterm", "cli"): clients((7, 0))})
        self.assertTrue(WezTermTarget(7, run).still_focused())
        self.assertFalse(WezTermTarget(8, run).still_focused())


class XdotoolTests(unittest.TestCase):
    def test_send_backspace_then_type(self):
        run = FakeRun()
        self.assertTrue(XdotoolTarget("42", run).send(Edit(2, "hi")))
        self.assertEqual(run.calls[0][0][:2], ["xdotool", "key"])
        self.assertIn("--repeat", run.calls[0][0])
        self.assertEqual(run.calls[0][0][-1], "BackSpace")
        self.assertIn("2", run.calls[0][0])
        self.assertEqual(run.calls[1][0][:2], ["xdotool", "type"])
        self.assertEqual(run.calls[1][0][-1], "hi")

    def test_append_only_skips_backspace(self):
        run = FakeRun()
        XdotoolTarget("42", run).send(Edit(0, "hi"))
        self.assertEqual(len(run.calls), 1)

    def test_still_focused_compares_window(self):
        run = FakeRun({("xdotool", "getactivewindow"): "42\n"})
        self.assertTrue(XdotoolTarget("42", run).still_focused())
        self.assertFalse(XdotoolTarget("43", run).still_focused())


class ChooseTargetTests(unittest.TestCase):
    def test_wezterm_class_gives_wezterm_target(self):
        run = FakeRun({("wezterm", "cli"): clients((5, 0))})
        t = choose_target('WM_CLASS(STRING) = "org.wezfurlong.wezterm", "org.wezfurlong.wezterm"', run)
        self.assertIsInstance(t, WezTermTarget)
        self.assertEqual(t.pane_id, 5)

    def test_other_class_gives_xdotool(self):
        run = FakeRun({("xdotool", "getactivewindow"): "42\n"})
        t = choose_target('WM_CLASS(STRING) = "firefox", "firefox"', run)
        self.assertIsInstance(t, XdotoolTarget)

    def test_wezterm_unreachable_falls_back_to_xdotool(self):
        run = FakeRun({("wezterm", "cli"): "", ("xdotool", "getactivewindow"): "42\n"})
        self.assertIsInstance(choose_target("wezterm", run), XdotoolTarget)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `venv/bin/python -m unittest tests.test_live_output -v`
Expected: ERROR `ModuleNotFoundError: No module named 'live_output'`

- [ ] **Step 3: Implement**

`live_output.py`:
```python
"""Output targets for live dictation: apply Edits (backspace N, then insert)."""

import json
import subprocess


def wezterm_focused_pane(run=subprocess.run):
    """Pane id focused in the most recently active WezTerm client, or None."""
    try:
        res = run(["wezterm", "cli", "list-clients", "--format", "json"],
                  capture_output=True, text=True, timeout=2)
    except Exception:
        return None
    if res.returncode != 0:
        return None
    try:
        entries = json.loads(res.stdout or "[]")
    except ValueError:
        return None
    best = None
    for entry in entries:
        pane = entry.get("focused_pane_id")
        if pane is None:
            continue
        idle = entry.get("idle_time") or {}
        idle_s = float(idle.get("secs", 0)) + float(idle.get("nanos", 0)) / 1e9
        if best is None or idle_s < best[0]:
            best = (idle_s, int(pane))
    return None if best is None else best[1]


def xdotool_active_window(run=subprocess.run):
    try:
        res = run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=1)
    except Exception:
        return None
    if res.returncode != 0:
        return None
    return (res.stdout or "").strip() or None


class WezTermTarget:
    name = "wezterm"

    def __init__(self, pane_id, run=subprocess.run):
        self.pane_id = int(pane_id)
        self._run = run

    @classmethod
    def detect(cls, run=subprocess.run):
        pane = wezterm_focused_pane(run)
        return cls(pane, run) if pane is not None else None

    def send(self, edit):
        text = "\x7f" * edit.backspace + edit.insert
        if not text:
            return True
        try:
            res = self._run(
                ["wezterm", "cli", "send-text", "--pane-id", str(self.pane_id), "--no-paste"],
                input=text.encode("utf-8"), capture_output=True, timeout=5,
            )
        except Exception:
            return False
        return res.returncode == 0

    def still_focused(self):
        return wezterm_focused_pane(self._run) == self.pane_id


class XdotoolTarget:
    name = "xdotool"

    def __init__(self, window_id, run=subprocess.run):
        self.window_id = str(window_id)
        self._run = run

    @classmethod
    def detect(cls, run=subprocess.run):
        wid = xdotool_active_window(run)
        return cls(wid, run) if wid else None

    def send(self, edit):
        try:
            if edit.backspace > 0:
                res = self._run(
                    ["xdotool", "key", "--clearmodifiers", "--delay", "8",
                     "--repeat", str(edit.backspace), "BackSpace"],
                    capture_output=True, timeout=10,
                )
                if res.returncode != 0:
                    return False
            if edit.insert:
                res = self._run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "8", "--", edit.insert],
                    capture_output=True, timeout=30,
                )
                if res.returncode != 0:
                    return False
        except Exception:
            return False
        return True

    def still_focused(self):
        return xdotool_active_window(self._run) == self.window_id


def choose_target(wm_class, run=subprocess.run):
    """WezTerm pane target when a WezTerm window is focused, else xdotool."""
    if "wezterm" in (wm_class or "").lower():
        target = WezTermTarget.detect(run)
        if target is not None:
            return target
    return XdotoolTarget.detect(run)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `venv/bin/python -m unittest tests.test_live_output -v`
Expected: all tests OK.

- [ ] **Step 5: Commit**

```bash
git add live_output.py tests/test_live_output.py
git commit -m "Add WezTerm and xdotool output targets for live dictation"
```

---

### Task 4: Live controller (worker)

**Files:**
- Create: `live_controller.py`
- Test: `tests/test_live_controller.py`

**Interfaces:**
- Consumes:
  - From Task 1: `ChunkReady` and `LongPause`.
  - From Task 2: `LiveSession`, `Edit`.
  - From Task 3: any target with `send(edit) -> bool`, `still_focused() -> bool` and `name`.
- Produces: `LiveController(session, target, fast_transcribe, correct_transcribe=None, make_target=None, postprocess=None, on_error=None, on_done=None, log=print)`:
  - `start()`, `submit(event)`, `stop()`, `join(timeout=None)`
  - `run_until_stopped()`, the worker loop, public so tests can run it synchronously.
  - `process(event)`, `correct()`, `finish()`
  - Attributes: `target`, `session`, `corrections_enabled`.
- `fast_transcribe(audio, prompt) -> str` and `correct_transcribe(audio, prompt) -> str` may raise. `make_target() -> target | None`. `postprocess(str) -> str` handles text replacements. `on_error(msg)` is called at most once per session. `on_done()` is called after the final commit.

- [ ] **Step 1: Write the failing tests**

`tests/test_live_controller.py`:
```python
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_controller import LiveController
from live_segmenter import ChunkReady, LongPause
from live_session import Edit, LiveSession


def chunk(t0, t1):
    return ChunkReady(np.zeros(int(16000 * (t1 - t0)), dtype=np.float32), t0, t1)


class FakeTarget:
    name = "fake"

    def __init__(self, ok=True):
        self.edits = []
        self.focused = True
        self.ok = ok

    def send(self, edit):
        self.edits.append(edit)
        return self.ok

    def still_focused(self):
        return self.focused


class Script:
    """Transcriber returning scripted results; an Exception item is raised."""

    def __init__(self, *results):
        self.results = list(results)
        self.prompts = []

    def __call__(self, audio, prompt):
        self.prompts.append(prompt)
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class LiveControllerTests(unittest.TestCase):
    def make(self, fast, correct=None, target=None, **kwargs):
        self.target = target or FakeTarget()
        self.errors = []
        self.done = []
        return LiveController(
            LiveSession(), self.target, fast, correct,
            on_error=self.errors.append, on_done=lambda: self.done.append(True),
            log=lambda *_: None, **kwargs,
        )

    def test_fast_result_is_typed(self):
        ctl = self.make(Script("Hello there"))
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(self.target.edits, [Edit(0, "Hello there")])

    def test_postprocess_applied(self):
        ctl = self.make(Script("hello cube control"), postprocess=lambda t: t.replace("cube control", "kubectl"))
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(self.target.edits, [Edit(0, "hello kubectl")])

    def test_correction_rewrites_tail(self):
        ctl = self.make(Script("We should merge the", "brunch today"),
                        Script("We should merge the branch today"))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.target.edits[-1], Edit(10, "anch today"))

    def test_single_typed_chunk_skips_correction(self):
        correct = Script()
        ctl = self.make(Script("Hello"), correct)
        ctl.process(chunk(0.0, 1.0))
        ctl.correct()
        self.assertEqual(correct.prompts, [])

    def test_focus_change_commits_and_retargets(self):
        new_target = FakeTarget()
        correct = Script()
        ctl = self.make(Script("Hello there", "More"), correct, make_target=lambda: new_target)
        ctl.process(chunk(0.0, 1.0))
        self.target.focused = False
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.target.edits, [Edit(0, "Hello there")])
        self.assertEqual(new_target.edits, [Edit(0, " more")])
        self.assertIs(ctl.target, new_target)
        self.assertEqual(correct.prompts, [])

    def test_focus_lost_before_correction_skips_it(self):
        correct = Script("x")
        ctl = self.make(Script("a", "b"), correct)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.target.focused = False
        ctl.correct()
        self.assertEqual(correct.prompts, [])
        self.assertEqual(ctl.session.chunk_count, 0)

    def test_fast_failure_recovered_by_correction(self):
        ctl = self.make(Script("Hello there", RuntimeError("timeout"), RuntimeError("again")),
                        Script("Hello there general Kenobi"))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.target.edits[-1], Edit(0, " general Kenobi"))
        ctl.process(chunk(2.0, 3.0))
        self.assertEqual(len(self.errors), 1)

    def test_correction_failure_is_silent(self):
        ctl = self.make(Script("a", "b"), Script(RuntimeError("boom")))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.errors, [])
        self.assertEqual(len(self.target.edits), 2)

    def test_output_failure_disables_corrections(self):
        correct = Script()
        ctl = self.make(Script("a", "b"), correct, target=FakeTarget(ok=False))
        ctl.process(chunk(0.0, 1.0))
        self.assertFalse(ctl.corrections_enabled)
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(correct.prompts, [])

    def test_long_pause_after_sentence_commits(self):
        ctl = self.make(Script("Done."))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(LongPause(2.0))
        self.assertEqual(ctl.session.chunk_count, 0)
        self.assertEqual(ctl.session.committed_text, "Done.")

    def test_fast_prompt_carries_previous_text(self):
        fast = Script("Hello", "world")
        ctl = self.make(fast)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(fast.prompts, ["", "Hello"])

    def test_loop_coalesces_corrections_and_finishes(self):
        fast = Script("We should merge the", "brunch")
        correct = Script("We should merge the branch")
        ctl = self.make(fast, correct)
        ctl.submit(chunk(0.0, 1.0))
        ctl.submit(chunk(1.0, 2.0))
        ctl.stop()
        ctl.run_until_stopped()
        self.assertEqual(len(fast.prompts), 2)
        self.assertEqual(len(correct.prompts), 1)
        self.assertEqual(self.target.edits[-1], Edit(4, "anch"))
        self.assertEqual(self.done, [True])
        self.assertEqual(ctl.session.chunk_count, 0)

    def test_no_correct_transcriber_means_no_corrections(self):
        ctl = self.make(Script("a", "b"))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(len(self.target.edits), 2)

    def test_threaded_start_and_join(self):
        ctl = self.make(Script("Hello"))
        ctl.start()
        ctl.submit(chunk(0.0, 1.0))
        ctl.stop()
        ctl.join(timeout=5)
        self.assertEqual(self.target.edits, [Edit(0, "Hello")])
        self.assertEqual(self.done, [True])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `venv/bin/python -m unittest tests.test_live_controller -v`
Expected: ERROR `ModuleNotFoundError: No module named 'live_controller'`

- [ ] **Step 3: Implement**

`live_controller.py`:
```python
"""Drive a live dictation session on one worker thread.

Each ChunkReady gets a fast pass (new chunk only, typed immediately). When no
further chunk is waiting, the whole window is re-transcribed and the typed
tail rewritten where it differs. Edits are applied in order on this thread.
"""

import queue
import threading

from live_segmenter import ChunkReady, LongPause

_STOP = object()


class LiveController:
    def __init__(self, session, target, fast_transcribe, correct_transcribe=None,
                 make_target=None, postprocess=None, on_error=None, on_done=None, log=print):
        self.session = session
        self.target = target
        self.fast_transcribe = fast_transcribe
        self.correct_transcribe = correct_transcribe
        self.make_target = make_target
        self.postprocess = postprocess or (lambda text: text)
        self.on_error = on_error
        self.on_done = on_done
        self.log = log
        self.corrections_enabled = correct_transcribe is not None
        self._jobs = queue.Queue()
        self._thread = None
        self._error_reported = False

    def start(self):
        self._thread = threading.Thread(target=self.run_until_stopped, daemon=True)
        self._thread.start()

    def submit(self, event):
        self._jobs.put(event)

    def stop(self):
        self._jobs.put(_STOP)

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def run_until_stopped(self):
        while True:
            event = self._jobs.get()
            if event is _STOP:
                self.finish()
                return
            try:
                self.process(event)
                if isinstance(event, ChunkReady) and self._jobs.empty():
                    self.correct()
            except Exception as e:  # keep the worker alive
                self.log(f"[live] worker error: {e}")

    def process(self, event):
        if isinstance(event, ChunkReady):
            self._handle_chunk(event)
        elif isinstance(event, LongPause):
            if self.session.should_commit(long_pause=True):
                self.session.commit()

    def _handle_chunk(self, chunk):
        if not self.target.still_focused():
            self._retarget()
        if self.session.should_commit(incoming_s=chunk.t1 - chunk.t0):
            self.session.commit()
        self.session.add_chunk(chunk.audio, chunk.t0, chunk.t1)
        try:
            raw = self.fast_transcribe(chunk.audio, self.session.fast_prompt())
        except Exception as e:
            self._report(f"Live transcription failed: {e}")
            return
        edit = self.session.add_fast_result(chunk.t1, self.postprocess(raw or ""))
        if edit is not None:
            self._send(edit)

    def correct(self):
        if not self.corrections_enabled:
            return
        s = self.session
        if s.chunk_count == 0 or (s.chunk_count == 1 and s.piece_count == 1):
            return
        window = s.window_audio()
        if window is None:
            return
        audio, t_upto, generation = window
        if not self.target.still_focused():
            self._retarget()
            return
        try:
            raw = self.correct_transcribe(audio, s.correction_prompt())
        except Exception as e:
            self.log(f"[live] correction failed: {e}")
            return
        edit = s.apply_correction(t_upto, self.postprocess(raw or ""), generation)
        if edit is not None:
            self._send(edit)

    def finish(self):
        try:
            self.correct()
        except Exception as e:
            self.log(f"[live] final correction failed: {e}")
        self.session.commit()
        if self.on_done:
            self.on_done()

    def _retarget(self):
        self.session.commit()
        if self.make_target is not None:
            new = self.make_target()
            if new is not None:
                self.target = new

    def _send(self, edit):
        if self.target.send(edit):
            return True
        self.log("[live] output failed; committing window and disabling corrections")
        self.session.commit()
        self.corrections_enabled = False
        return False

    def _report(self, message):
        self.log(f"[live] {message}")
        if not self._error_reported and self.on_error:
            self._error_reported = True
            self.on_error(message)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `venv/bin/python -m unittest tests.test_live_controller -v`
Expected: all tests OK.

- [ ] **Step 5: Commit**

```bash
git add live_controller.py tests/test_live_controller.py
git commit -m "Add live dictation controller with fast pass and window correction"
```

---

### Task 5: Wire live mode into the tray app

**Files:**
- Modify: `whisper_dictate.py`:
  - `DEFAULT_CONFIG` (~line 71)
  - `transcribe_remote` (~667)
  - `_transcribe_local` (~701)
  - `_toggle_recording_impl` (~441)
  - `create_menu` (~1258)
  - `run` (Keybinder section)
  - new methods placed after `toggle_transcribe_session` (~806)
- Modify: `README.md` (add a "Live dictation" section)
- Test: the full suite, plus a manual checklist.

**Interfaces:**
- Consumes:
  - `LiveSegmenter` from Task 1
  - `LiveSession` from Task 2
  - `choose_target` from Task 3
  - `LiveController` from Task 4
- Produces: `WhisperDictate.toggle_live()`, the config keys from Global Constraints, and a `prompt=None` keyword on `transcribe_remote` and `_transcribe_local`.

- [ ] **Step 1: Add the config defaults**

In `DEFAULT_CONFIG`, after `"transcribe_chunk_seconds": 25,`, add:
```python
    "live_hotkey": "<Alt><Shift>d",
    "live_pause_ms": 600,
    "live_max_chunk_s": 8,
    "live_window_max_s": 12,
    "live_commit_pause_ms": 1200,
    "live_max_backspace": 80,
    "live_corrections": True,
    "live_committed_context_chars": 400,
```

- [ ] **Step 2: Let the transcribers take an explicit prompt**

Change `def transcribe_remote(self, audio, timeout=60):` to `def transcribe_remote(self, audio, timeout=60, prompt=None):`. In its `build_remote_header(...)` call, replace `prompt=self.get_asr_context(),` with:
```python
            prompt=self.get_asr_context() if prompt is None else prompt,
```
Change `def _transcribe_local(self, audio, model_name=None):` to `def _transcribe_local(self, audio, model_name=None, prompt=None):`, and replace its line `prompt = self.get_asr_context()` with:
```python
        if prompt is None:
            prompt = self.get_asr_context()
```

- [ ] **Step 3: Add the imports**

Next to the other local imports near the top (around the `from asr_models import ...` block), add:
```python
from live_controller import LiveController
from live_output import choose_target
from live_segmenter import LiveSegmenter
from live_session import LiveSession
```

- [ ] **Step 4: Make one-shot mode and live mode exclusive**

In `_toggle_recording_impl`, before `if self.recording:`, insert:
```python
        if getattr(self, "live_active", False):
            self.notify("Live dictation is running; stop it first.")
            return False
```

- [ ] **Step 5: Add the live-mode methods**

Insert after `toggle_transcribe_session`:
```python
    def toggle_live(self, *args):
        """Toggle live dictation (type at pauses, correct recent words)."""
        self.saved_window = self.get_focused_window()
        GLib.idle_add(self._toggle_live_impl)

    def _toggle_live_impl(self):
        if getattr(self, "live_active", False):
            self.stop_live()
        else:
            self.start_live()
        return False

    def _live_transcribers(self):
        """(fast, correct) callables; correct is None unless the remote server is usable."""
        remote_enabled = self.get_remote_config().get("enabled", False)
        with self.remote_state_lock:
            remote_down = self.remote_available is False
        if not remote_enabled:
            model = self.config.get("model", "base")
            return (lambda audio, prompt: self._transcribe_local(audio, model_name=model, prompt=prompt)), None

        fallback = self.get_cpu_fallback_model()

        def local(audio, prompt):
            return self._transcribe_local(audio, model_name=fallback, prompt=prompt)

        if remote_down:
            return local, None

        def fast(audio, prompt):
            try:
                text = self.transcribe_remote(audio, timeout=10, prompt=prompt)
                self.set_remote_available(True)
                return text
            except Exception as e:
                self.set_remote_available(False, str(e))
                return local(audio, prompt)

        def correct(audio, prompt):
            with self.remote_state_lock:
                if self.remote_available is False:
                    raise RuntimeError("remote unavailable")
            return self.transcribe_remote(audio, timeout=10, prompt=prompt)

        return fast, correct

    def start_live(self):
        if self.recording or getattr(self, "transcribe_active", False):
            self.notify("Stop the current recording before starting live dictation.")
            return
        target = choose_target(self.get_focused_window_class())
        if target is None:
            self.notify("Live dictation: no focused window found.")
            return
        cfg = self.config
        sample_rate = cfg["sample_rate"]
        segmenter = LiveSegmenter(
            sample_rate,
            pause_ms=int(cfg.get("live_pause_ms", 600)),
            max_chunk_s=float(cfg.get("live_max_chunk_s", 8)),
            threshold=float(cfg.get("silence_threshold", 0.0)),
            long_pause_ms=int(cfg.get("live_commit_pause_ms", 1200)),
        )
        session = LiveSession(
            base_prompt=self.get_asr_context(),
            context_chars=int(cfg.get("live_committed_context_chars", 400)),
            max_backspace=int(cfg.get("live_max_backspace", 80)),
            window_max_s=float(cfg.get("live_window_max_s", 12)),
        )
        fast, correct = self._live_transcribers()
        if not cfg.get("live_corrections", True):
            correct = None
        controller = LiveController(
            session, target, fast, correct,
            make_target=lambda: choose_target(self.get_focused_window_class()),
            postprocess=self.apply_replacements,
            on_error=lambda msg: GLib.idle_add(lambda: self.notify(msg) or False),
            on_done=lambda: GLib.idle_add(self._live_done),
        )
        controller.start()

        def audio_callback(indata, frames, time_info, status):
            for event in segmenter.feed(indata[:, 0]):
                controller.submit(event)

        self.live_segmenter = segmenter
        self.live_controller = controller
        self.live_stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype=np.float32, callback=audio_callback
        )
        self.live_stream.start()
        self.live_active = True
        self.beep_start()
        self.update_icon(True)
        self.update_status(f"⚡ Live → {target.name}" + ("" if correct else " (no corrections)"))
        print(f"[whisper-dictate] Live dictation started ({target.name}, corrections={correct is not None})")
        GLib.timeout_add(50, lambda: self.restore_focus(getattr(self, 'saved_window', None)) or False)

    def stop_live(self):
        if not getattr(self, "live_active", False):
            return
        self.live_active = False
        if self.live_stream:
            self.live_stream.stop()
            self.live_stream.close()
            self.live_stream = None
        for event in self.live_segmenter.flush():
            self.live_controller.submit(event)
        self.live_controller.stop()
        self.update_status("Finishing live dictation...")

    def _live_done(self):
        self.beep_stop()
        self.update_icon(False)
        self.update_status("Ready")
        print("[whisper-dictate] Live dictation stopped")
        return False
```

- [ ] **Step 6: Add the tray item and the hotkey**

In `create_menu`, right after the `transcribe_session_item` block (after it is appended to the menu), add, using the same append call that block uses:
```python
        live_item = Gtk.MenuItem(label="⚡ Live dictation")
        live_item.connect("activate", self.toggle_live)
        menu.append(live_item)
```
(If the menu variable in that function has a different name than `menu`, use that name.)

In `run`, after the `Keybinder.bind(hotkey, ...)` if/else block, add:
```python
        live_hotkey = self.config.get("live_hotkey", "<Alt><Shift>d")
        if live_hotkey:
            if Keybinder.bind(live_hotkey, lambda _ks: self.toggle_live()):
                print(f"✓ Live hotkey {live_hotkey} registered")
            else:
                print(f"✗ Failed to register live hotkey {live_hotkey}")
```
After `Keybinder.unbind(hotkey)` at the end of `run`, add:
```python
        if live_hotkey:
            Keybinder.unbind(live_hotkey)
```

- [ ] **Step 7: Run the full suite and an import check**

Run: `venv/bin/python -m unittest discover -s tests -v`
Expected: all tests OK (the new ones plus the existing ones).

Run: `venv/bin/python -c "import ast,sys; ast.parse(open('whisper_dictate.py').read())" && venv/bin/python -c "import whisper_dictate"`
Expected: no output and exit 0. If `import whisper_dictate` fails only because there is no display or GI, the `ast` parse passing is enough; say so in the report.

- [ ] **Step 8: Document it in the README**

Add a `## Live dictation` section to `README.md`:
```markdown
## Live dictation

Press `<Alt><Shift>d` (config `live_hotkey`) or pick **⚡ Live dictation** in the tray,
then talk. Text is typed at each pause (~0.6 s). With the remote server up, the last
≤ 12 s are re-transcribed after each pause and recently typed words are corrected in
place (backspace + retype). Press the hotkey again to stop.

- WezTerm: typed into the pane focused at start via `wezterm cli send-text`.
- Other windows: `xdotool`.
- Switching window/pane mid-session freezes what was typed and continues in the new
  target without corrections to earlier text.

Config keys: `live_pause_ms`, `live_max_chunk_s`, `live_window_max_s`,
`live_commit_pause_ms`, `live_max_backspace`, `live_corrections`,
`live_committed_context_chars`.
```

- [ ] **Step 9: Commit**

```bash
git add whisper_dictate.py README.md
git commit -m "Wire live dictation mode into tray app and hotkey"
```

- [ ] **Step 10: Manual checklist (the operator runs this; the implementer lists it in the report)**

1. Restart the app. The log shows `✓ Live hotkey <Alt><Shift>d registered`.
2. In a WezTerm pane running Claude Code through `achat run`, press the hotkey and talk for about 30 s with natural pauses. Text appears within about 0.5 s of each pause.
3. Say something ambiguous early in a sentence. It sometimes fixes itself a second or two later.
4. Switch to another pane mid-sentence. The earlier pane is not touched again, and new text goes to the new pane.
5. Stop the remote server and start live mode. The status shows "(no corrections)" and text is still typed.
6. Stop live mode during silence. There is a beep, and the status goes back to Ready.
7. One-shot `<Alt>d` still works while live mode is off, and refuses with a notification while live mode is on.
