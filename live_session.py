"""Window bookkeeping for live dictation (pure, deterministic).

The window is the audio and text since the last commit. Text typed for the
window may be rewritten by a correction; committed text is never edited again
and is only used as context for later transcriptions.
"""

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

SENTENCE_END = (".", "?", "!")
_KEEP_CASE = {"I", "I'm", "I'll", "I've", "I'd"}


@dataclass(frozen=True)
class Edit:
    backspace: int
    insert: str


def _collapse(text):
    """Collapse runs of spaces and tabs, keeping line breaks ("new line")."""
    return "\n".join(" ".join(line.split()) for line in (text or "").split("\n"))


def strip_echo(raw, context, max_words=8, min_words=3, min_words_inside=4):
    """Drop a repeat of the context from raw.

    The model sometimes repeats the end of its prompt before the new words,
    or (on a chunk with no real speech) returns the whole prompt. A repeat of
    the context's last words at the start of raw, or of at least
    `min_words_inside` of them anywhere in raw, is dropped with everything
    before it.
    """
    words = context.split()
    raw = _collapse(raw)
    low = raw.lower()
    for n in range(min(max_words, len(words)), min_words - 1, -1):
        tail = " ".join(words[-n:]).lower()
        if low.startswith(tail):
            return raw[len(tail):].lstrip(" ,")
        at = low.rfind(tail) if n >= min_words_inside else -1
        if at >= 0:
            return raw[at + len(tail):].lstrip(" ,")
    return raw


def is_slash_command(line):
    """A line like "/compact" or "/review the branch": no sentence punctuation."""
    return line.lstrip().startswith("/")


def normalise(raw, before, after="", keep_lone_mark=False):
    """Turn a raw transcription into the exact text to type between `before`
    and `after` (the rest of the line when dictating mid-line).

    A lone mark (".") is model noise and dropped, unless keep_lone_mark: with
    auto punctuation off it was said ("period").
    """
    # Precomposed: achat refuses combining marks, and backspace counts are
    # per character on every target.
    text = unicodedata.normalize("NFC", _collapse(raw))
    text = strip_echo(text, before).strip(" ")
    if keep_lone_mark and text in (",", ".", "?", "!", ";", ":"):
        return text  # glued to the text before it
    if not any(ch.isalnum() or ch == "\n" for ch in text):
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
    if before and not before.endswith((" ", "\n")) and not text.startswith(("\n", ",", ".", "?", "!", ";", ":")):
        text = " " + text
    if is_slash_command(before.split("\n")[-1] + text):
        text = text.rstrip(".?!")
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


UNDO_STEPS = 20
_LAST_END = re.compile(r"[.?!][^.?!]*$")


class LiveSession:
    def __init__(self, base_prompt="", context_chars=400, max_backspace=80, window_max_s=12.0,
                 command_max_backspace=300):
        self.base_prompt = base_prompt or ""
        self.context_chars = int(context_chars)
        self.max_backspace = int(max_backspace)
        self.command_max_backspace = int(command_max_backspace)
        self.window_max_s = float(window_max_s)
        self.committed_text = ""
        # What this session typed that is still right before the cursor,
        # across commits: the only text a spoken command may rewrite.
        self.history = ""
        # Snapshots of history before each step (an utterance or a repair),
        # newest last: "undo" turns history back into the last one.
        self._undo = []
        self._window_steps = 0
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
        self._push_undo()
        self._window_steps += 1
        self._set_history(self.history + text)
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
        self._set_history(self.history[:max(0, len(self.history) - backspace)] + insert)
        if self._window_steps > 1:
            # The corrected window is one step now: undo removes it as corrected.
            del self._undo[-(self._window_steps - 1):]
            self._window_steps = 1
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
        self._window_steps = 0
        text = self.committed_text + self.typed_window
        self.committed_text = text[-max(self.context_chars, 1):]
        self._chunks = []
        self._pieces = []

    def _set_history(self, text):
        self.history = text
        self.trim_history(max(self.context_chars, self.command_max_backspace))

    def _push_undo(self, snapshot=None):
        self._undo.append(self.history if snapshot is None else snapshot)
        del self._undo[:-UNDO_STEPS]

    def trim_history(self, keep):
        """Keep only the last `keep` characters of history as ours to edit.

        Undo snapshots are shifted along; one that does not share the dropped
        prefix (an older state) is dropped with everything before it, so an
        undo never retypes text from outside what we may touch.
        """
        cut = max(0, len(self.history) - max(0, keep))
        if not cut:
            return
        dropped = self.history[:cut]
        self.history = self.history[cut:]
        kept = []
        for snap in reversed(self._undo):
            if len(snap) < cut or not snap.startswith(dropped):
                break
            kept.append(snap[cut:])
        self._undo = kept[::-1]

    def forget_history(self):
        """The text before the cursor is no longer known to be ours."""
        self.history = ""
        self._undo = []

    def undo(self):
        """Turn history back into what it was before the last step.

        One edit over our own text, like a command; commits the window.
        None when there is no step to undo or the edit would be too long.
        """
        if not self._undo:
            return None
        line = self.committed_text + self.typed_window
        self.commit()
        old, new = self.history, self._undo[-1]
        prefix = _common_prefix(old, new)
        backspace = len(old) - prefix
        if backspace > self.command_max_backspace:
            return None
        self._undo.pop()
        insert = new[prefix:]
        line = line[:max(0, len(line) - backspace)] + insert
        self.committed_text = line[-self.context_chars:] if self.context_chars > 0 else ""
        self.history = new
        return Edit(backspace, insert) if backspace or insert else None

    def apply_command(self, command):
        """Apply a spoken repair (see live_commands) as one edit over the history.

        Rewrites only text this session typed. Always commits the window (the
        command chunk's audio included), so no later correction can undo the
        repair or type the command words. None when nothing changes or the
        edit would backspace more than command_max_backspace.
        """
        line = self.committed_text + self.typed_window
        self.commit()
        old = self.history
        if command.scratch:
            if command.rest:
                return None  # the words before "scratch that" were never typed
            new = self._scratched(old)
        elif not old and not command.rest:
            return None  # nothing of ours to repair, nothing new to type
        else:
            new = self._repaired(old, command, line)
        prefix = _common_prefix(old, new)
        backspace = len(old) - prefix
        insert = new[prefix:]
        if backspace > self.command_max_backspace or (not backspace and not insert):
            return None
        line = line[:max(0, len(line) - backspace)] + insert
        self.committed_text = line[-self.context_chars:] if self.context_chars > 0 else ""
        self._push_undo(old)
        self._set_history(new)
        return Edit(backspace, insert)

    @staticmethod
    def _scratched(history):
        """History without its current sentence (or the one just completed)."""
        body = history.rstrip()
        if body.endswith(SENTENCE_END):
            body = body[:-1]  # the current sentence is empty: take the last one
        m = _LAST_END.search(body)
        return history[:m.start() + 1] if m else ""

    def _repaired(self, history, command, line):
        """History with a join or comma repair and the chunk's words appended."""
        base = history
        # Any mark the line ends with gives way to the spoken one.
        end_at = re.search(r"[.?!,;:]\s*$", base)
        if command.comma:
            if end_at:
                base = base[:end_at.start()] + "," + base[end_at.start() + 1:]
            elif base:
                base = base.rstrip() + ","
        elif command.end and end_at:
            base = base[:end_at.start()] + base[end_at.start() + 1:]
        mid_line = self.after_text[:1].isalnum()
        if not command.rest:
            # Only punctuation changes; keep the space before a following word.
            base = base.rstrip() + command.end if command.end else base.rstrip()
            return base + " " if mid_line else base
        # The line before the history, as the context for the new words.
        before = line[:max(0, len(line) - len(history))] + base
        words = normalise(command.rest, before, self.after_text)
        if command.end:
            words = words.rstrip().rstrip(".?!,;:") + command.end
            if mid_line:
                words += " "
        return base + words

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
