"""Spoken punctuation repairs for live dictation (pure, English).

Automatic punctuation stays the default. These words repair it when a
thinking pause produced a wrong sentence end:

- "... period" / "full stop" / "question mark" / "exclamation mark" at the
  end of a chunk: join the chunk to the previous one and end it so.
- "comma ..." at the start of a chunk: the previous sentence end becomes a
  comma and the chunk continues the sentence.
- "... scratch that" at the end of a chunk: delete the current sentence.

The model may punctuate the command words themselves ("Period.", "Comma,")
or write a spoken comma as "," - both are recognised.
"""

import re
from dataclasses import dataclass

_END_WORDS = {
    "period": ".",
    "full stop": ".",
    "question mark": "?",
    "exclamation mark": "!",
    "exclamation point": "!",
}
# "period", "full stop" and "comma" are also ordinary words ("the trial
# period", "comma separated"). They only count when spoken as a separate
# piece: the whole chunk, or set off by punctuation the model put there
# because it heard a pause. "question mark"/"exclamation mark" are rare
# enough as text to count after plain words too.
_PAUSED_ONLY = {"period", "full stop"}

_END_RE = re.compile(
    r"(^|[,.;:!?]\s*|\s+)(" + "|".join(k.replace(" ", r"\s+") for k in _END_WORDS) + r")[\s.?!,]*$",
    re.IGNORECASE,
)
_COMMA_RE = re.compile(r"^\s*(?:comma(?:\s*[,.;:]+\s*|\s*$)|,\s*)", re.IGNORECASE)
_ENTER_RE = re.compile(r"(^|[,.;:!?]\s*|\s+)press\s+(?:enter|return)[\s.?!,]*$", re.IGNORECASE)
_ENTER_ALONE_RE = re.compile(r"^\s*(?:enter|return)[\s.?!,]*$", re.IGNORECASE)
_FILLER = {"no", "oh", "okay", "ok", "um", "uh", "hmm", "ah", "well", "wait", "sorry", "yeah", "so"}
_SCRATCH_RE = re.compile(r"(?:^|[\s,.?!]+)scratch\s+that[\s.?!,]*$", re.IGNORECASE)


@dataclass(frozen=True)
class Command:
    rest: str = ""  # the words spoken with the command (command words removed)
    comma: bool = False  # the previous sentence end becomes a comma
    end: str = ""  # join to the previous chunk and end with this
    scratch: bool = False  # delete the current sentence
    enter: bool = False  # then press Enter (submit the line)


def split_enter(text):
    """(text without a trailing "press enter", whether Enter was asked for).

    "press enter" / "press return" at the end, or "enter" said alone. A bare
    "enter" inside a sentence stays text ("Enter is not working").
    """
    text = (text or "").strip()
    if _ENTER_ALONE_RE.match(text):
        return "", True
    m = _ENTER_RE.search(text)
    if not m:
        return text, False
    mark = m.group(1).strip()
    return (text[:m.start()] + (mark if mark in (".", "?", "!") else "")).strip(), True


_SET_OFF_RE = re.compile(
    r"(?:^|[,.;:!?])\s*(?:" + "|".join(k.replace(" ", r"\s+") for k in (*_END_WORDS, "comma", "scratch that", "press enter", "press return"))
    + r")\s*(?:[,.;:!?]|$)",
    re.IGNORECASE,
)


def mentions_command(text):
    """Whether a command word stands on its own anywhere in text (between
    punctuation), as in a correction that heard a command the fast pass missed."""
    return parse_command(text) is not None or bool(_SET_OFF_RE.search(text or ""))


def parse_command(text):
    """The spoken command in a chunk's text, or None for plain dictation."""
    text = (text or "").strip()
    m = _SCRATCH_RE.search(text)
    if m:
        rest = text[:m.start()].strip(" ,.")
        if all(w in _FILLER for w in re.findall(r"[a-z']+", rest.lower())):
            rest = ""  # "No, scratch that." / "Okay, scratch that."
        return Command(rest=rest, scratch=True)
    text, enter = split_enter(text)
    end = ""
    m = _END_RE.search(text)
    if m:
        word = " ".join(m.group(2).lower().split())
        paused = m.group(1).strip() != "" or m.start() == 0
        if word not in _PAUSED_ONLY or paused:
            end = _END_WORDS[word]
            text = text[:m.start()]
    comma = False
    m = _COMMA_RE.match(text)
    if m:
        comma = True
        text = text[m.end():]
    if not end and not comma and not enter:
        return None
    return Command(rest=text.strip(), comma=comma, end=end, enter=enter)
