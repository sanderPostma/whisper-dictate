# Live Dictation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live dictation mode that types text at each speech pause while the operator keeps talking. It uses a bounded audio window (at most 12 s) to correct words it typed recently.

**Architecture:** Four new, mostly pure modules:
- `live_segmenter.py` detects pauses.
- `live_session.py` holds the window and committed text, does normalisation and computes correction diffs.
- `live_output.py` holds the WezTerm and xdotool targets that apply "backspace N, then insert".
- `live_controller.py` is one worker thread, plus the choice of transcribers.

For each chunk the controller runs a fast pass and types the result. When no further event is waiting, it runs a correction pass. `whisper_dictate.py` only wires these together: a hotkey, a tray item, the audio stream and the transcriber callables. One-shot mode is unchanged.

**Tech Stack:** Python 3 and numpy (2.3). The existing app uses sounddevice and GTK/GLib/Keybinder. Output goes through `wezterm cli` and `xdotool`. Tests use `unittest`, run with `venv/bin/python -m unittest` (the venv has no pytest).

**Spec:** `docs/superpowers/specs/2026-09-24-live-dictation-design.md`

**Verification status:** every code block in Tasks 1 to 4 was run before this plan was written: 77 new and existing tests pass. The Task 5 edits were applied to a copy of `whisper_dictate.py`, which then imported cleanly. `_live_transcribers` was also smoke-tested on a stub instance. The code blocks are therefore meant to be used as written.

## Deviations from the spec (deliberate)

1. **No separate output queue.** Edits are applied on the single worker thread, straight after the transcription that produced them. This keeps edits strictly in order, and a correction can never go stale because commits happen on the same thread. As a result, `apply_correction` needs no staleness token: its signature stays `apply_correction(t_upto, raw)`, as in the spec. The cost is that typing an edit (0.1 to 0.4 s with xdotool, near zero with WezTerm) delays the next transcription by that much.
2. **"A queued correction is replaced by a newer one"** becomes: a correction runs only when the job queue is empty right after a chunk. A burst of chunks therefore gets one correction instead of several.
3. **`WezTermTarget` also pins the X window.** Switching away from WezTerm counts as losing focus, the same as switching panes. Otherwise text would keep going into a terminal the operator is no longer looking at.
4. **The transcriber choice is a pure function,** `select_transcribers` in `live_controller.py`, so the remote/local rules have unit tests. Its per-call checks mean corrections resume if the server comes back mid-session.
5. **Two new defaults beyond the spec:**
   - `live_hotkey` = `<Alt><Shift>d`. The spec had no hotkey, and `toggle-recording.sh` has no signal handler in the app to hook into.
   - The segmenter drops voiced bursts shorter than `min_speech_ms` = 200. This filters the start beep and clicks.

## Global Constraints

- Config defaults, copied from the spec: `live_pause_ms` 600, `live_max_chunk_s` 8, `live_window_max_s` 12, `live_commit_pause_ms` 1200, `live_max_backspace` 80, `live_corrections` true, `live_committed_context_chars` 400. Plus `live_hotkey` `"<Alt><Shift>d"`.
- The server protocol is unchanged. Each call is an ordinary request with a `prompt`.
- One-shot mode and live mode are mutually exclusive. One-shot behaviour must not change otherwise.
- Kitty and clipboard modes are not supported in live mode: anything that isn't WezTerm goes through xdotool.
- Corrections need the remote server to be enabled and not known to be down. Local models do fast passes only.
- Committed text is never edited again.
- Backspace is `\x7f` sent through `wezterm cli send-text --no-paste`, or `xdotool key BackSpace`.
- Work on branch `live-dictation` in the current checkout. Never create git worktrees.
- Tests follow the existing style: `unittest.TestCase`, and `sys.path.insert(0, <repo root>)` at the top of each test file.

## Review Focus

1. **Talking non-stop for more than 8 s:** the forced split must not lose or duplicate audio. The next chunk starts exactly where the forced one ended. (Task 1, `test_forced_split_is_contiguous`.)
2. **Focus moves to another pane or window mid-session:** no backspaces or corrections may land in the new place. The window is committed and typing continues append-only in the new target. (Task 3, `test_switching_to_another_x_window_is_focus_loss`; Task 4, `test_focus_change_commits_and_retargets`.)
3. **A correction comes back empty or as bare punctuation** (noise, a cough): the typed text stays. (Task 2, `test_empty_correction_keeps_text`.)
4. **A fast pass fails** (timeout, server blip): that chunk's words still appear after the next correction, and the operator is notified once per session, not once per chunk. (Task 4, `test_fast_failure_recovered_by_correction`.)
5. **A correction would rewrite more than 80 characters:** it is skipped rather than wiping a long line. (Task 2, `test_backspace_cap`.)

---

### Task 1: Pause segmenter

**Files:**
- Create: `live_segmenter.py`
- Test: `tests/test_live_segmenter.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ChunkReady(audio: np.ndarray, t0: float, t1: float, forced: bool = False)`
  - `LongPause(t: float)`
  - `LiveSegmenter(sample_rate, pause_ms=600, max_chunk_s=8.0, threshold=0.0, pad_ms=150, long_pause_ms=1200, min_speech_ms=200)`, with `.threshold`, `.feed(block) -> list[ChunkReady | LongPause]` and `.flush() -> list[ChunkReady]`.
  - Times are seconds since the first block fed.
  - `threshold <= 0` means the default 0.01.

**Behaviour:**
- A block is voiced when its RMS is at least `threshold`.
- A chunk starts at the first voiced block and includes up to `pad_ms` of pre-roll.
- A chunk ends after `pause_ms` of continuous silence. At most `pad_ms` of trailing silence is kept.
- A chunk is also forced out at `max_chunk_s`. The next chunk then starts at that exact sample.
- After a chunk ends in a pause, one `LongPause` fires once the silence reaches `long_pause_ms`.
- A chunk with less than `min_speech_ms` of voiced audio is dropped silently.

- [ ] **Step 1: Write the failing tests** in `tests/test_live_segmenter.py`:

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


def feed_all(seg, audio, block=BLOCK):
    events = []
    for i in range(0, len(audio), block):
        events.extend(seg.feed(audio[i:i + block]))
    return events


def chunks_of(events):
    return [e for e in events if isinstance(e, ChunkReady)]


class LiveSegmenterTests(unittest.TestCase):
    def test_pause_emits_chunk_with_trimmed_trailing_silence(self):
        seg = LiveSegmenter(SR)
        chunks = chunks_of(feed_all(seg, np.concatenate([tone(1.0), silence(0.7)])))
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].t0, 0.0)
        self.assertAlmostEqual(chunks[0].t1, 1.15)
        self.assertEqual(len(chunks[0].audio), int(1.15 * SR))
        self.assertFalse(chunks[0].forced)

    def test_short_pause_does_not_split(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(1.0), silence(0.5), tone(0.5)]))
        self.assertEqual(chunks_of(events), [])

    def test_leading_preroll_is_kept(self):
        seg = LiveSegmenter(SR)
        chunk = chunks_of(feed_all(seg, np.concatenate([silence(1.0), tone(1.0), silence(0.7)])))[0]
        self.assertAlmostEqual(chunk.t0, 0.85)

    def test_long_pause_fires_once_after_chunk(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(1.0), silence(3.0)]))
        self.assertEqual([type(e) for e in events], [ChunkReady, LongPause])

    def test_forced_split_is_contiguous(self):
        seg = LiveSegmenter(SR, max_chunk_s=8.0)
        chunks = chunks_of(feed_all(seg, tone(9.0)) + seg.flush())
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].forced)
        self.assertAlmostEqual(chunks[0].t1, 8.0)
        self.assertAlmostEqual(chunks[1].t0, chunks[0].t1)
        self.assertEqual(len(chunks[0].audio) + len(chunks[1].audio), 9 * SR)

    def test_odd_block_sizes_keep_timing(self):
        seg = LiveSegmenter(SR)
        chunk = chunks_of(feed_all(seg, np.concatenate([silence(0.5), tone(1.0), silence(0.8)]), block=333))[0]
        self.assertAlmostEqual(chunk.t0 + len(chunk.audio) / SR, chunk.t1)
        self.assertLess(abs(chunk.t0 - 0.35), 0.03)

    def test_zero_threshold_uses_default(self):
        seg = LiveSegmenter(SR, threshold=0.0)
        self.assertEqual(seg.threshold, 0.01)
        self.assertEqual(feed_all(seg, np.concatenate([tone(1.0, amp=0.001), silence(1.0)])), [])

    def test_blip_shorter_than_min_speech_is_dropped(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(0.1), silence(2.0)]))
        self.assertEqual(events, [])

    def test_flush_without_speech_is_empty(self):
        seg = LiveSegmenter(SR)
        feed_all(seg, silence(1.0))
        self.assertEqual(seg.flush(), [])

    def test_flush_emits_pending_speech_once(self):
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
Expected: ERROR, `ModuleNotFoundError: No module named 'live_segmenter'`

- [ ] **Step 3: Implement** `live_segmenter.py`:

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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `venv/bin/python -m unittest tests.test_live_segmenter -v`
Expected: `Ran 10 tests`, `OK`

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
- Consumes: nothing. Its audio arguments are numpy arrays such as `ChunkReady.audio`.
- Produces:
  - `Edit(backspace: int, insert: str)`, a frozen dataclass.
  - `strip_echo(raw, context) -> str`
  - `normalise(raw, before) -> str`
  - `LiveSession(base_prompt="", context_chars=400, max_backspace=80, window_max_s=12.0)` with:
    - Attributes: `.committed_text`, `.typed_window` (property), `.chunk_count`, `.piece_count`.
    - `.window_seconds() -> float`
    - `.add_chunk(audio, t0, t1)`
    - `.add_fast_result(t1, raw) -> Edit | None`
    - `.window_audio() -> tuple[np.ndarray, float] | None`, which returns the audio and the `t_upto` value.
    - `.apply_correction(t_upto, raw) -> Edit | None`
    - `.should_commit(long_pause=False, focus_ok=True, incoming_s=0.0) -> bool`
    - `.commit()`
    - `.fast_prompt() -> str`, `.correction_prompt() -> str`

**Behaviour:**
- **Normalise:**
  - Collapse whitespace.
  - Strip an echo of the preceding 3 to 8 words.
  - Drop results that have no alphanumeric characters.
  - Mid-sentence, lowercase the first letter. A sentence has ended when the preceding text ends in `.?!` or a newline. Keep `I`, `I'm`, `I'll`, `I've`, `I'd` and all-caps acronyms of 2 or more letters as they are.
  - Prefix a space unless nothing precedes the text or it ends in a space or newline.
- **Correction:**
  - The corrected text is normalised against the committed text only.
  - `backspace = len(typed_window) - common_prefix(old, new)`, where `old` is the text typed for pieces with `t1 <= t_upto`.
  - `insert = new[prefix:]` plus the text of any later pieces.
  - The result is `None` when the text is identical, the new text is empty, or `backspace > max_backspace`.
- **Commit rules:**
  - The window is non-empty, and one of the following holds:
    - focus is lost,
    - window length plus the incoming chunk would exceed `window_max_s`,
    - a long pause follows text ending in `.?!`.
  - The committed text is trimmed to the last `context_chars` characters.
- **Prompts:**
  - Fast prompt: `base_prompt`, a newline, then the tail of (committed text + typed window).
  - Correction prompt: `base_prompt`, a newline, then the tail of the committed text.
  - Empty parts are omitted.

- [ ] **Step 1: Write the failing tests** in `tests/test_live_session.py`:

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
    """pieces: (t0, t1, raw) triples, each fed as a chunk plus its fast result."""
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

    def test_after_newline_keeps_capital_and_adds_no_space(self):
        self.assertEqual(normalise("Next", "line\n"), "Next")

    def test_after_trailing_space_adds_no_space(self):
        self.assertEqual(normalise("next", "word "), "next")

    def test_empty_and_punctuation_only_are_dropped(self):
        self.assertEqual(normalise("", "x"), "")
        self.assertEqual(normalise(None, "x"), "")
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
        self.assertEqual(s.chunk_count, 1)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.s = session_with([
            (0.0, 1.0, "We should merge the"),
            (1.0, 2.0, "brunch"),
            (2.0, 3.0, "today"),
        ])
        self.assertEqual(self.s.typed_window, "We should merge the brunch today")

    def test_correction_rewrites_only_differing_tail(self):
        edit = self.s.apply_correction(3.0, "We should merge the branch today")
        self.assertEqual(edit, Edit(10, "anch today"))
        self.assertEqual(self.s.typed_window, "We should merge the branch today")
        self.assertEqual(self.s.piece_count, 1)

    def test_pieces_after_t_upto_are_retyped(self):
        edit = self.s.apply_correction(2.0, "We should merge the branch")
        self.assertEqual(edit, Edit(10, "anch today"))
        self.assertEqual(self.s.typed_window, "We should merge the branch today")
        self.assertEqual(self.s.piece_count, 2)

    def test_identical_correction_is_none(self):
        self.assertIsNone(self.s.apply_correction(3.0, "We should merge the brunch today"))

    def test_empty_correction_keeps_text(self):
        self.assertIsNone(self.s.apply_correction(3.0, " . "))
        self.assertEqual(self.s.typed_window, "We should merge the brunch today")

    def test_backspace_cap(self):
        s = session_with([(0.0, 1.0, "We should merge the"), (1.0, 2.0, "brunch")], max_backspace=3)
        self.assertIsNone(s.apply_correction(2.0, "We should merge the branch"))
        self.assertEqual(s.typed_window, "We should merge the brunch")

    def test_correction_recovers_missing_fast_result(self):
        s = session_with([(0.0, 1.0, "Hello there")])
        s.add_chunk(audio(1), 1.0, 2.0)  # its fast pass failed: no piece
        self.assertEqual(s.apply_correction(2.0, "Hello there general Kenobi"), Edit(0, " general Kenobi"))

    def test_correction_after_commit_uses_committed_spacing(self):
        s = session_with([(0.0, 1.0, "One.")])
        s.commit()
        s.add_chunk(audio(1), 1.0, 2.0)
        s.add_fast_result(2.0, "Too")
        s.add_chunk(audio(1), 2.0, 3.0)
        s.add_fast_result(3.0, "more")
        self.assertEqual(s.apply_correction(3.0, "Two more"), Edit(7, "wo more"))

    def test_window_audio_covers_all_chunks(self):
        audio_all, t_upto = self.s.window_audio()
        self.assertEqual(len(audio_all), 3 * 16000)
        self.assertEqual(t_upto, 3.0)

    def test_window_audio_empty_is_none(self):
        self.assertIsNone(LiveSession().window_audio())


class CommitTests(unittest.TestCase):
    def test_long_pause_after_sentence_commits(self):
        self.assertTrue(session_with([(0.0, 1.0, "Done.")]).should_commit(long_pause=True))

    def test_long_pause_mid_sentence_does_not_commit(self):
        self.assertFalse(session_with([(0.0, 1.0, "and then")]).should_commit(long_pause=True))

    def test_window_limit_commits(self):
        s = session_with([(0.0, 8.0, "a"), (8.0, 11.0, "b")], window_max_s=12.0)
        self.assertFalse(s.should_commit(incoming_s=1.0))
        self.assertTrue(s.should_commit(incoming_s=2.0))

    def test_focus_loss_commits(self):
        self.assertTrue(session_with([(0.0, 1.0, "a")]).should_commit(focus_ok=False))

    def test_empty_window_never_commits(self):
        self.assertFalse(LiveSession().should_commit(long_pause=True, focus_ok=False))

    def test_commit_moves_text_and_clears_window(self):
        s = session_with([(0.0, 1.0, "Hello there")], context_chars=5)
        s.commit()
        self.assertEqual(s.committed_text, "there")
        self.assertEqual(s.typed_window, "")
        self.assertEqual(s.chunk_count, 0)
        s.add_chunk(audio(1), 1.0, 2.0)
        self.assertEqual(s.add_fast_result(2.0, "General"), Edit(0, " general"))


class PromptTests(unittest.TestCase):
    def test_fast_prompt_has_base_committed_and_window(self):
        s = session_with([(0.0, 1.0, "One.")], base_prompt="Vocabulary: git")
        s.commit()
        s.add_chunk(audio(1), 1.0, 2.0)
        s.add_fast_result(2.0, "Two")
        self.assertEqual(s.fast_prompt(), "Vocabulary: git\nOne. Two")
        self.assertEqual(s.correction_prompt(), "Vocabulary: git\nOne.")

    def test_prompt_tail_is_trimmed(self):
        s = session_with([(0.0, 1.0, "abcdefgh")], context_chars=4)
        self.assertEqual(s.fast_prompt(), "efgh")

    def test_no_base_no_text_is_empty(self):
        self.assertEqual(LiveSession().fast_prompt(), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `venv/bin/python -m unittest tests.test_live_session -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'live_session'`

- [ ] **Step 3: Implement** `live_session.py`:

```python
"""Window bookkeeping for live dictation (pure, deterministic).

The window is the audio and text since the last commit. Text typed for the
window may be rewritten by a correction; committed text is never edited again
and is only used as context for later transcriptions.
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
    """Turn a raw transcription into the exact text to type after `before`."""
    text = " ".join((raw or "").split())
    text = strip_echo(text, before).strip()
    if not any(ch.isalnum() for ch in text):
        return ""
    stripped = before.rstrip()
    mid_sentence = stripped and not stripped.endswith(SENTENCE_END) and not before.endswith("\n")
    if mid_sentence:
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
        self._chunks = []  # (audio, t0, t1)
        self._pieces = []  # (t1, typed text), in typing order

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
        """(audio of the whole window, t1 of its last chunk), or None if empty."""
        if not self._chunks:
            return None
        return np.concatenate([a for a, _, _ in self._chunks]), self._chunks[-1][2]

    def apply_correction(self, t_upto, raw):
        """Rewrite the text typed for chunks ending at or before t_upto."""
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
Expected: `Ran 30 tests`, `OK`

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
- Consumes: `Edit` from Task 2, used only in tests. Targets read `.backspace` and `.insert`.
- Produces:
  - `WezTermTarget(pane_id, window_id, run=subprocess.run)` and `XdotoolTarget(window_id, run=subprocess.run)`. Both have `.name`, `.send(edit) -> bool`, `.still_focused() -> bool` and the classmethod `.detect(run) -> target | None`.
  - `wezterm_focused_pane(run) -> int | None`
  - `xdotool_active_window(run) -> str | None`
  - `choose_target(wm_class: str | None, run=subprocess.run) -> target | None`
  - `run` has the call signature of `subprocess.run`, and its result has `.returncode` and `.stdout`.

**Behaviour:**
- `wezterm cli list-clients --format json` returns a list of clients, each with `focused_pane_id` and `idle_time {secs, nanos}` (checked against the installed WezTerm). The client with the least idle time wins.
- `send-text --no-paste` reads the text from stdin when no TEXT argument is given (checked against `--help`).
- Every command failure or exception becomes `False` or `None`. Nothing in this module raises.

- [ ] **Step 1: Write the failing tests** in `tests/test_live_output.py`:

```python
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_output import WezTermTarget, XdotoolTarget, choose_target, wezterm_focused_pane
from live_session import Edit

LIST_CLIENTS = ("wezterm", "cli", "list-clients")
ACTIVE_WINDOW = ("xdotool", "getactivewindow")


class FakeRun:
    """Stands in for subprocess.run: records calls, answers stdout by command prefix."""

    def __init__(self, answers=None, returncode=0):
        self.calls = []
        self.answers = answers or {}
        self.returncode = returncode

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        stdout = ""
        for prefix, out in self.answers.items():
            if tuple(cmd[:len(prefix)]) == prefix:
                stdout = out
        return SimpleNamespace(returncode=self.returncode, stdout=stdout)


def clients(*entries):
    return json.dumps([
        {"focused_pane_id": pane, "idle_time": {"secs": secs, "nanos": 0}}
        for pane, secs in entries
    ])


class WezTermTests(unittest.TestCase):
    def test_focused_pane_picks_most_recently_active_client(self):
        run = FakeRun({LIST_CLIENTS: clients((3, 50), (7, 1))})
        self.assertEqual(wezterm_focused_pane(run), 7)

    def test_focused_pane_bad_json_is_none(self):
        self.assertIsNone(wezterm_focused_pane(FakeRun({LIST_CLIENTS: "nope"})))

    def test_focused_pane_command_failure_is_none(self):
        self.assertIsNone(wezterm_focused_pane(FakeRun({LIST_CLIENTS: clients((7, 0))}, returncode=1)))

    def test_detect_pins_pane_and_window(self):
        run = FakeRun({LIST_CLIENTS: clients((7, 0)), ACTIVE_WINDOW: "42\n"})
        target = WezTermTarget.detect(run)
        self.assertEqual((target.pane_id, target.window_id), (7, "42"))

    def test_send_backspaces_and_insert_via_stdin(self):
        run = FakeRun()
        self.assertTrue(WezTermTarget(7, "42", run).send(Edit(3, "abc")))
        cmd, kwargs = run.calls[0]
        self.assertEqual(cmd, ["wezterm", "cli", "send-text", "--pane-id", "7", "--no-paste"])
        self.assertEqual(kwargs["input"], b"\x7f\x7f\x7fabc")

    def test_send_failure_returns_false(self):
        self.assertFalse(WezTermTarget(7, "42", FakeRun(returncode=1)).send(Edit(0, "x")))

    def test_empty_edit_runs_nothing(self):
        run = FakeRun()
        self.assertTrue(WezTermTarget(7, "42", run).send(Edit(0, "")))
        self.assertEqual(run.calls, [])

    def test_still_focused_needs_same_pane_and_window(self):
        run = FakeRun({LIST_CLIENTS: clients((7, 0)), ACTIVE_WINDOW: "42\n"})
        self.assertTrue(WezTermTarget(7, "42", run).still_focused())
        self.assertFalse(WezTermTarget(8, "42", run).still_focused())

    def test_switching_to_another_x_window_is_focus_loss(self):
        run = FakeRun({LIST_CLIENTS: clients((7, 0)), ACTIVE_WINDOW: "99\n"})
        self.assertFalse(WezTermTarget(7, "42", run).still_focused())


class XdotoolTests(unittest.TestCase):
    def test_send_backspace_then_type(self):
        run = FakeRun()
        self.assertTrue(XdotoolTarget("42", run).send(Edit(2, "hi")))
        self.assertEqual(run.calls[0][0], ["xdotool", "key", "--clearmodifiers", "--delay", "8",
                                           "--repeat", "2", "BackSpace"])
        self.assertEqual(run.calls[1][0], ["xdotool", "type", "--clearmodifiers", "--delay", "8",
                                           "--", "hi"])

    def test_append_only_skips_backspace(self):
        run = FakeRun()
        XdotoolTarget("42", run).send(Edit(0, "hi"))
        self.assertEqual(len(run.calls), 1)

    def test_failed_backspace_stops_before_typing(self):
        run = FakeRun(returncode=1)
        self.assertFalse(XdotoolTarget("42", run).send(Edit(2, "hi")))
        self.assertEqual(len(run.calls), 1)

    def test_still_focused_compares_window(self):
        run = FakeRun({ACTIVE_WINDOW: "42\n"})
        self.assertTrue(XdotoolTarget("42", run).still_focused())
        self.assertFalse(XdotoolTarget("43", run).still_focused())


class ChooseTargetTests(unittest.TestCase):
    def test_wezterm_class_gives_wezterm_target(self):
        run = FakeRun({LIST_CLIENTS: clients((5, 0)), ACTIVE_WINDOW: "42\n"})
        target = choose_target('WM_CLASS(STRING) = "org.wezfurlong.wezterm", "org.wezfurlong.wezterm"', run)
        self.assertIsInstance(target, WezTermTarget)
        self.assertEqual(target.pane_id, 5)

    def test_other_class_gives_xdotool(self):
        run = FakeRun({ACTIVE_WINDOW: "42\n"})
        self.assertIsInstance(choose_target('WM_CLASS(STRING) = "firefox", "firefox"', run), XdotoolTarget)

    def test_wezterm_unreachable_falls_back_to_xdotool(self):
        run = FakeRun({ACTIVE_WINDOW: "42\n"})
        self.assertIsInstance(choose_target("wezterm", run), XdotoolTarget)

    def test_nothing_focused_is_none(self):
        self.assertIsNone(choose_target(None, FakeRun()))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `venv/bin/python -m unittest tests.test_live_output -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'live_output'`

- [ ] **Step 3: Implement** `live_output.py`:

```python
"""Output targets for live dictation: apply an Edit (backspace N, then insert).

A target is pinned to what had focus when it was created; still_focused()
reports whether that is still where the operator is working.
"""

import json
import subprocess


def wezterm_focused_pane(run=subprocess.run):
    """Pane focused in the most recently active WezTerm client, or None."""
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
    """X id of the active window as a string, or None."""
    try:
        res = run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=1)
    except Exception:
        return None
    if res.returncode != 0:
        return None
    return (res.stdout or "").strip() or None


class WezTermTarget:
    """Types into one WezTerm pane via `wezterm cli send-text --no-paste`."""

    name = "wezterm"

    def __init__(self, pane_id, window_id, run=subprocess.run):
        self.pane_id = int(pane_id)
        self.window_id = str(window_id)
        self._run = run

    @classmethod
    def detect(cls, run=subprocess.run):
        window = xdotool_active_window(run)
        pane = wezterm_focused_pane(run)
        if window is None or pane is None:
            return None
        return cls(pane, window, run)

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
        if xdotool_active_window(self._run) != self.window_id:
            return False
        return wezterm_focused_pane(self._run) == self.pane_id


class XdotoolTarget:
    """Types into the active X window with xdotool."""

    name = "xdotool"

    def __init__(self, window_id, run=subprocess.run):
        self.window_id = str(window_id)
        self._run = run

    @classmethod
    def detect(cls, run=subprocess.run):
        window = xdotool_active_window(run)
        return cls(window, run) if window else None

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
Expected: `Ran 17 tests`, `OK`

- [ ] **Step 5: Check against the real desktop**

Run from a WezTerm pane: `venv/bin/python -c "from live_output import *; print(wezterm_focused_pane(), xdotool_active_window(), type(choose_target('wezterm')).__name__)"`
Expected: a pane number, a window id, and `WezTermTarget`.

- [ ] **Step 6: Commit**

```bash
git add live_output.py tests/test_live_output.py
git commit -m "Add WezTerm and xdotool output targets for live dictation"
```

---

### Task 4: Live controller and transcriber selection

**Files:**
- Create: `live_controller.py`
- Test: `tests/test_live_controller.py`

**Interfaces:**
- Consumes:
  - From Task 1: `ChunkReady`, `LongPause`.
  - From Task 2: `LiveSession` and its full method set, plus `Edit`.
  - From Task 3: any object with `send(edit) -> bool`, `still_focused() -> bool` and `name`.
- Produces:
  - `select_transcribers(remote_enabled, remote_is_down, remote, local, local_fallback, on_remote_result) -> (fast, correct | None)`
  - `LiveController(session, target, fast_transcribe, correct_transcribe=None, make_target=None, postprocess=None, on_error=None, on_done=None, log=print)`, with:
    - `.start()`, `.submit(event)`, `.stop()`, `.join(timeout=None)`, `.is_running()`
    - `.run_until_stopped()`, the worker loop. It is public so tests can run it synchronously.
    - `.process(event)`, `.correct()`, `.finish()`
    - Attributes `.session`, `.target`, `.corrections_enabled`.
  - Transcribers have the form `(audio, prompt) -> str` and may raise.
  - `on_error(msg)` is called at most once per session.
  - `on_done()` runs on the worker thread after the final commit.

**Behaviour:**
- **Chunk:**
  1. If focus was lost, commit and retarget.
  2. If the incoming chunk would overflow the window, commit.
  3. Add the chunk, run the fast pass and type the result.
  4. A fast-pass error is reported once and types nothing. The audio stays in the window so the next correction can recover it.
- **Correction:**
  - Skipped when corrections are disabled, the window is empty, or the window is one chunk that already has its typed text.
  - Skipped, with a commit and retarget, if focus was lost.
  - A correction error is logged only.
- **Output failure:** commit and disable corrections for the rest of the session.
- **Stop:** one final correction, then commit, then `on_done`.

- [ ] **Step 1: Write the failing tests** in `tests/test_live_controller.py`:

```python
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_controller import LiveController, select_transcribers
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
    """Transcriber returning scripted results in order; Exception items are raised."""

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

    def test_postprocess_applied_before_typing(self):
        ctl = self.make(Script("hello cube control"),
                        postprocess=lambda t: t.replace("cube control", "kubectl"))
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(self.target.edits, [Edit(0, "hello kubectl")])

    def test_fast_prompt_carries_previous_text(self):
        fast = Script("Hello", "world")
        ctl = self.make(fast)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(fast.prompts, ["", "Hello"])

    def test_correction_rewrites_tail(self):
        correct = Script("We should merge the branch today")
        ctl = self.make(Script("We should merge the", "brunch today"), correct)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.target.edits[-1], Edit(10, "anch today"))
        self.assertEqual(correct.prompts, [""])

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
        self.assertEqual(len(self.target.edits), 2)

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

    def test_window_limit_commits_before_adding_chunk(self):
        ctl = self.make(Script("a", "b", "c"))
        ctl.process(chunk(0.0, 6.0))
        ctl.process(chunk(6.0, 11.0))
        ctl.process(chunk(11.0, 13.0))
        self.assertEqual(ctl.session.chunk_count, 1)
        self.assertEqual(ctl.session.committed_text, "a b")

    def test_no_correct_transcriber_means_no_corrections(self):
        ctl = self.make(Script("a", "b"))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(len(self.target.edits), 2)

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

    def test_worker_survives_a_crashing_target(self):
        class Boom(FakeTarget):
            def send(self, edit):
                raise RuntimeError("boom")

        ctl = self.make(Script("a"), target=Boom())
        ctl.submit(chunk(0.0, 1.0))
        ctl.stop()
        ctl.run_until_stopped()
        self.assertEqual(self.done, [True])

    def test_threaded_start_and_join(self):
        ctl = self.make(Script("Hello"))
        ctl.start()
        ctl.submit(chunk(0.0, 1.0))
        ctl.stop()
        ctl.join(timeout=5)
        self.assertFalse(ctl.is_running())
        self.assertEqual(self.target.edits, [Edit(0, "Hello")])
        self.assertEqual(self.done, [True])


class SelectTranscribersTests(unittest.TestCase):
    def setUp(self):
        self.down = False
        self.results = []
        self.remote = Script()
        self.local = Script()
        self.fallback = Script()

    def select(self, remote_enabled=True):
        return select_transcribers(
            remote_enabled, lambda: self.down, self.remote, self.local, self.fallback,
            lambda ok, reason: self.results.append(ok),
        )

    def test_remote_disabled_uses_local_without_corrections(self):
        fast, correct = self.select(remote_enabled=False)
        self.assertIs(fast, self.local)
        self.assertIsNone(correct)

    def test_remote_up_uses_remote_for_both(self):
        self.remote.results = ["fast", "corrected"]
        fast, correct = self.select()
        self.assertEqual(fast(None, "p"), "fast")
        self.assertEqual(correct(None, "p"), "corrected")
        self.assertEqual(self.results, [True])

    def test_remote_error_falls_back_and_reports(self):
        self.remote.results = [RuntimeError("refused")]
        self.fallback.results = ["local"]
        fast, _ = self.select()
        self.assertEqual(fast(None, "p"), "local")
        self.assertEqual(self.results, [False])

    def test_remote_down_skips_remote_and_corrections_raise(self):
        self.down = True
        self.fallback.results = ["local"]
        fast, correct = self.select()
        self.assertEqual(fast(None, "p"), "local")
        self.assertEqual(self.remote.prompts, [])
        with self.assertRaises(RuntimeError):
            correct(None, "p")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `venv/bin/python -m unittest tests.test_live_controller -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'live_controller'`

- [ ] **Step 3: Implement** `live_controller.py`:

```python
"""Drive a live dictation session on one worker thread.

Each ChunkReady gets a fast pass (the new chunk only, typed immediately).
When no further event is waiting, the whole window is re-transcribed and the
typed tail is rewritten where it differs. All edits are applied in order on
this one thread, so the session never sees a correction go stale.
"""

import queue
import threading

from live_segmenter import ChunkReady, LongPause

_STOP = object()


def select_transcribers(remote_enabled, remote_is_down, remote, local, local_fallback,
                        on_remote_result):
    """Pick the (fast, correct) transcribers for a live session.

    remote/local/local_fallback: (audio, prompt) -> str, may raise.
    remote_is_down: () -> bool, the app's current view of the server.
    on_remote_result: (ok: bool, reason: str | None) -> None.
    correct is None when corrections are impossible for the whole session.
    """
    if not remote_enabled:
        return local, None

    def fast(audio, prompt):
        if remote_is_down():
            return local_fallback(audio, prompt)
        try:
            text = remote(audio, prompt)
        except Exception as e:
            on_remote_result(False, str(e))
            return local_fallback(audio, prompt)
        on_remote_result(True, None)
        return text

    def correct(audio, prompt):
        if remote_is_down():
            raise RuntimeError("remote server unavailable")
        return remote(audio, prompt)

    return fast, correct


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

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

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
            except Exception as e:  # never let one bad event kill the session
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
        """Re-transcribe the window and rewrite the typed tail where it differs."""
        if not self.corrections_enabled:
            return
        s = self.session
        if s.chunk_count == 0 or (s.chunk_count == 1 and s.piece_count == 1):
            return  # nothing to gain: same audio as the fast pass
        audio, t_upto = s.window_audio()
        if not self.target.still_focused():
            self._retarget()
            return
        try:
            raw = self.correct_transcribe(audio, s.correction_prompt())
        except Exception as e:
            self.log(f"[live] correction skipped: {e}")
            return
        edit = s.apply_correction(t_upto, self.postprocess(raw or ""))
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
        """Focus moved: freeze the window and follow the new focus, append-only."""
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
Expected: `Ran 20 tests`, `OK`

- [ ] **Step 5: Commit**

```bash
git add live_controller.py tests/test_live_controller.py
git commit -m "Add live dictation controller and transcriber selection"
```

---

### Task 5: Wire live mode into the tray app

**Files:**
- Modify: `whisper_dictate.py`:
  - imports (~line 42)
  - `DEFAULT_CONFIG` (~84)
  - `transcribe_remote` (~667, 679)
  - `_transcribe_local` (~701, 707)
  - `_toggle_recording_impl` (~441)
  - new methods before `get_focused_window_class` (~788)
  - `create_menu` (~1261)
  - `run` (~1762, 1792)
- Modify: `README.md`, adding a "Live dictation" section.

**Interfaces:**
- Consumes: `LiveSegmenter` (Task 1), `LiveSession` (Task 2), `choose_target` (Task 3), `LiveController` and `select_transcribers` (Task 4).
- Consumes existing app methods: `transcribe_remote`, `_transcribe_local`, `get_cpu_fallback_model` (already removes `.en` for non-English languages), `set_remote_available`, `remote_state_lock`, `remote_available`, `get_asr_context`, `apply_replacements`, `get_focused_window`, `get_focused_window_class`, `restore_focus`, `notify`, `beep_start`, `beep_stop`, `update_icon`, `update_status`.
- Produces: `WhisperDictate.toggle_live()`, the live config keys, and a `prompt=None` keyword on `transcribe_remote` and `_transcribe_local`. `None` means "use `get_asr_context()`", which is the existing behaviour.

- [ ] **Step 1: Add the imports.** After `from asr_qwen import load_qwen, transcribe_qwen`, add:

```python
from live_controller import LiveController, select_transcribers
from live_output import choose_target
from live_segmenter import LiveSegmenter
from live_session import LiveSession
```

- [ ] **Step 2: Add the config defaults.** In `DEFAULT_CONFIG`, after `"transcribe_chunk_seconds": 25,`, add:

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

- [ ] **Step 3: Let the transcribers take an explicit prompt.**

In `transcribe_remote`, change the signature to `def transcribe_remote(self, audio, timeout=60, prompt=None):`. In its `build_remote_header(...)` call, replace `prompt=self.get_asr_context(),` with:

```python
            prompt=self.get_asr_context() if prompt is None else prompt,
```

In `_transcribe_local`, change the signature to `def _transcribe_local(self, audio, model_name=None, prompt=None):` and replace the line `prompt = self.get_asr_context()` with:

```python
        if prompt is None:
            prompt = self.get_asr_context()
```

- [ ] **Step 4: Make one-shot and transcribe-session modes refuse while live mode runs.** In `_toggle_recording_impl`, directly after its docstring, insert:

```python
        if getattr(self, "live_active", False):
            self.notify("Live dictation is running; stop it first.")
            return False
```

In `start_transcribe_session`, change its first guard from
`if getattr(self, "transcribe_active", False) or self.recording:` to:

```python
        if getattr(self, "transcribe_active", False) or self.recording or getattr(self, "live_active", False):
```

- [ ] **Step 5: Add the live-mode methods.** Insert this block immediately before `def get_focused_window_class(self):`:

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
        """(fast, correct) transcribers for a live session; correct may be None."""
        model = self.config.get("model", "base")
        fallback_model = self.get_cpu_fallback_model()

        def remote(audio, prompt):
            return self.transcribe_remote(audio, timeout=10, prompt=prompt)

        def local(audio, prompt):
            return self._transcribe_local(audio, model_name=model, prompt=prompt)

        def local_fallback(audio, prompt):
            return self._transcribe_local(audio, model_name=fallback_model, prompt=prompt)

        def remote_is_down():
            with self.remote_state_lock:
                return self.remote_available is False

        return select_transcribers(
            self.get_remote_config().get("enabled", False),
            remote_is_down, remote, local, local_fallback,
            lambda ok, reason: self.set_remote_available(ok, reason),
        )

    def start_live(self):
        if self.recording or getattr(self, "transcribe_active", False):
            self.notify("Stop the current recording before starting live dictation.")
            return
        previous = getattr(self, "live_controller", None)
        if previous is not None and previous.is_running():
            self.notify("Live dictation is still finishing; try again in a moment.")
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
            samplerate=sample_rate, channels=1, dtype=np.float32,
            blocksize=int(sample_rate * 0.05), callback=audio_callback,
        )
        self.live_stream.start()
        self.live_active = True
        self.beep_start()
        self.update_icon(True)
        suffix = "" if correct else " (no corrections)"
        self.update_status(f"⚡ Live → {target.name}{suffix}")
        print(f"[whisper-dictate] Live dictation started: {target.name}, corrections={correct is not None}")
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

Notes:
- `start_live` refuses while a previous session's worker is still typing, so two sessions can never interleave keystrokes.
- `blocksize` fixes audio blocks at 50 ms, so RMS voicing is measured over a meaningful span.
- `on_done` runs on the worker thread and hops to GTK through `GLib.idle_add`.

- [ ] **Step 6: Add the tray item and the hotkey.**

In `create_menu`, after the line `self.transcribe_session_item = transcribe_session_item`, add:

```python

        # Live dictation: type at pauses, correct recent words
        live_item = Gtk.MenuItem(label="⚡ Live dictation")
        live_item.connect("activate", self.toggle_live)
        menu.append(live_item)
```

In `run`, after the line `print(f"✗ Failed to register hotkey {hotkey}")` (the `else` branch of the main `Keybinder.bind`), add at the same indentation as that `if`:

```python
        live_hotkey = self.config.get("live_hotkey", "<Alt><Shift>d")
        if live_hotkey:
            if Keybinder.bind(live_hotkey, lambda _keystring: self.toggle_live()):
                print(f"✓ Live hotkey {live_hotkey} registered")
            else:
                print(f"✗ Failed to register live hotkey {live_hotkey}")
```

After `Keybinder.unbind(hotkey)` at the end of `run`, add:

```python
        if live_hotkey:
            Keybinder.unbind(live_hotkey)
```

- [ ] **Step 7: Run the full suite and import the app**

Run: `venv/bin/python -m unittest discover -s tests`
Expected: `Ran 98 tests`, `OK` (21 existing plus 77 new).

Run: `venv/bin/python -c "import whisper_dictate as w; assert hasattr(w.WhisperDictate, 'toggle_live'); print('ok')"`
Expected: `ok`

- [ ] **Step 8: Smoke-test the transcriber wiring without GTK or audio**

Run:
```bash
venv/bin/python -c "
import threading, whisper_dictate as w
app = object.__new__(w.WhisperDictate)
app.config = dict(w.DEFAULT_CONFIG, remote_server=dict(w.DEFAULT_CONFIG['remote_server'], enabled=True))
app.remote_state_lock = threading.Lock(); app.remote_available = None
app.transcribe_remote = lambda a, timeout, prompt: (_ for _ in ()).throw(ConnectionError('refused'))
app._transcribe_local = lambda a, model_name, prompt: 'local:' + model_name
app.set_remote_available = lambda ok, reason=None: setattr(app, 'remote_available', ok)
fast, correct = app._live_transcribers()
print(fast(None, 'p'), app.remote_available)
"
```
Expected: `local:base.en False`. The remote call failed, so the local fallback was used and the remote was marked down.

- [ ] **Step 9: Document it in the README.** Add this section to `README.md` after the usage section:

```markdown
## Live dictation

Press `<Alt><Shift>d` (config `live_hotkey`) or pick **⚡ Live dictation** in the tray,
then talk. Text is typed at each pause (~0.6 s). With the remote server up, the last
≤ 12 s are re-transcribed after each pause and recently typed words are corrected in
place (backspace + retype). Press the hotkey again to stop.

- WezTerm: typed into the pane focused at start via `wezterm cli send-text`.
- Other windows: `xdotool`.
- Switching window or pane mid-session freezes what was typed and continues in the
  new place, append-only.
- The local fallback model types but does not correct.

Config keys: `live_pause_ms`, `live_max_chunk_s`, `live_window_max_s`,
`live_commit_pause_ms`, `live_max_backspace`, `live_corrections`,
`live_committed_context_chars`, `live_hotkey`.
```

- [ ] **Step 10: Commit**

```bash
git add whisper_dictate.py README.md
git commit -m "Wire live dictation mode into tray app and hotkey"
```

- [ ] **Step 11: Manual checklist.** The operator runs these. The implementer copies them into the final report.

1. Restart the app (`systemctl --user restart whisper-dictate` or the usual way). The log shows `✓ Live hotkey <Alt><Shift>d registered`.
2. In a WezTerm pane running Claude Code under `achat run`, press the hotkey and talk for about 30 s with natural pauses. Text appears within about 0.5 s of each pause. Watch whether Claude Code folds a long chunk into `[Pasted text]`; if it does, note it for phase 2.
3. Say something ambiguous early in a sentence. It sometimes corrects itself a second or two later.
4. Mid-sentence, switch to another pane, then to a browser. Earlier text is never touched again, and new text follows focus.
5. Stop the remote server and start live mode. The status shows "(no corrections)" and text is still typed.
6. Stop live mode during silence. There is a beep and the status returns to Ready. Pressing the hotkey again immediately either starts a new session or says the previous one is still finishing, and never mixes the two.
7. `<Alt>d` one-shot works as before when live mode is off, and refuses with a notification while live mode is on.
