"""Window bookkeeping for live dictation (pure, deterministic).

The window is the audio and text since the last commit. Text typed for the
window may be rewritten by a correction; committed text is never edited again
and is only used as context for later transcriptions.
"""

import unicodedata
from dataclasses import dataclass

import numpy as np

SENTENCE_END = (".", "?", "!")
_KEEP_CASE = {"I", "I'm", "I'll", "I've", "I'd"}


@dataclass(frozen=True)
class Edit:
    backspace: int
    insert: str


def strip_echo(raw, context, max_words=8, min_words=3, min_words_inside=4):
    """Drop a repeat of the context from raw.

    The model sometimes repeats the end of its prompt before the new words,
    or (on a chunk with no real speech) returns the whole prompt. A repeat of
    the context's last words at the start of raw, or of at least
    `min_words_inside` of them anywhere in raw, is dropped with everything
    before it.
    """
    words = context.split()
    low = " ".join(raw.split()).lower()
    raw = " ".join(raw.split())
    for n in range(min(max_words, len(words)), min_words - 1, -1):
        tail = " ".join(words[-n:]).lower()
        if low.startswith(tail):
            return raw[len(tail):].lstrip(" ,")
        at = low.rfind(tail) if n >= min_words_inside else -1
        if at >= 0:
            return raw[at + len(tail):].lstrip(" ,")
    return raw


def normalise(raw, before, after=""):
    """Turn a raw transcription into the exact text to type between `before`
    and `after` (the rest of the line when dictating mid-line)."""
    # Precomposed: achat refuses combining marks, and backspace counts are
    # per character on every target.
    text = unicodedata.normalize("NFC", " ".join((raw or "").split()))
    text = strip_echo(text, before).strip()
    if not any(ch.isalnum() for ch in text):
        return ""
    stripped = before.rstrip()
    mid_sentence = stripped and not stripped.endswith(SENTENCE_END) and not before.endswith("\n")
    if mid_sentence:
        first = text.split(" ", 1)[0].rstrip(",.?!;:")
        acronym = len(first) > 1 and first.isupper()
        if first not in _KEEP_CASE and not acronym and first[:1].isupper():
            low = text[0].lower()
            if len(low) == 1:  # "İ".lower() grows a combining mark
                text = low + text[1:]
    if before and not before.endswith((" ", "\n")):
        text = " " + text
    if after[:1].isalnum():
        # Mid-sentence: an utterance-final full stop would split the sentence.
        text = text.rstrip(".?!") + " "
    return unicodedata.normalize("NFC", text)


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
        self.after_text = ""  # rest of the line after the cursor (mid-line dictation)
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
        text = normalise(raw, self.committed_text + self.typed_window, self.after_text)
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
        new = normalise(raw, self.committed_text, self.after_text)
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

    def set_context(self, before, after=""):
        """Take context from the real line around the cursor (window empty)."""
        self.committed_text = before[-self.context_chars:] if self.context_chars > 0 else ""
        self.after_text = after

    def _prompt(self, text):
        tail = text[-self.context_chars:].strip() if self.context_chars > 0 else ""
        return "\n".join(p for p in (self.base_prompt.strip(), tail) if p)

    def fast_prompt(self):
        return self._prompt(self.committed_text + self.typed_window)

    def correction_prompt(self):
        return self._prompt(self.committed_text)
