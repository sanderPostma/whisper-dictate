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
_ENTER_RE = re.compile(r"(^|[,.;:!?]\s*|\s+)(?:press|push)\s+(?:enter|return)[\s.?!,]*$", re.IGNORECASE)
# "engage" is an ordinary word too: only alone or set off by the model's punctuation.
_ENGAGE_RE = re.compile(r"(^|[,.;:!?]\s*)engage[\s.?!,]*$", re.IGNORECASE)
_ENTER_ALONE_RE = re.compile(r"^\s*(?:enter|return)[\s.?!,]*$", re.IGNORECASE)
_FILLER = {"no", "oh", "okay", "ok", "um", "uh", "hmm", "ah", "well", "wait", "sorry", "yeah", "so"}
# "stretch" is how the model tends to hear "scratch".
_SCRATCH_RE = re.compile(r"(?:^|[\s,.?!]+)(?:scratch|stretch)\s+that[\s.?!,]*$", re.IGNORECASE)
_SCRATCH_ALONE_RE = re.compile(r"^\s*(?:scratch|stretch)[\s.?!,]*$", re.IGNORECASE)
_CLEAR_RE = re.compile(
    r"^\s*(?:command[\s,]+clear(?:[\s,]+(?:the[\s,]+)?line)?"
    r"|(?:scratch|stretch)[\s,]+(?:all|everything|(?:the[\s,]+)?whole[\s,]+line))[\s.!,]*$",
    re.IGNORECASE,
)


_AUTO_PUNCT_RE = re.compile(
    r"^\s*auto(?:matic)?[\s-]*punctuation[\s,:]+(on|off|of)[\s.!,]*$", re.IGNORECASE)
_LONE_MARK_RE = re.compile(r"^\s*([.?!,])\s*$")


def parse_auto_punctuation(text):
    """True / False for "auto punctuation on" / "off" said as the whole
    utterance ("of" is how the model tends to hear "off"), else None."""
    m = _AUTO_PUNCT_RE.match(text or "")
    if not m:
        return None
    return m.group(1).lower() == "on"


def parse_clear(text):
    """Whether the whole utterance asks to clear the whole line
    ("command clear", "command clear line", "scratch all", "scratch whole line")."""
    return bool(_CLEAR_RE.match(text or ""))


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
    m = _ENTER_RE.search(text) or _ENGAGE_RE.search(text)
    if not m:
        return text, False
    mark = m.group(1).strip()
    return (text[:m.start()] + (mark if mark in (".", "?", "!") else "")).strip(), True


_SET_OFF_RE = re.compile(
    r"(?:^|[,.;:!?])\s*(?:" + "|".join(k.replace(" ", r"\s+") for k in (*_END_WORDS, "comma", "scratch that", "stretch that", "press enter", "press return", "push enter", "engage"))
    + r")\s*(?:[,.;:!?]|$)",
    re.IGNORECASE,
)


def mentions_command(text):
    """Whether a command word stands on its own anywhere in text (between
    punctuation), as in a correction that heard a command the fast pass missed."""
    if _LONE_MARK_RE.match(text or ""):
        return False  # model noise in a correction, not a command
    return (parse_command(text) is not None or parse_auto_punctuation(text) is not None
            or bool(_SET_OFF_RE.search(text or "")))


def parse_command(text):
    """The spoken command in a chunk's text, or None for plain dictation."""
    text = (text or "").strip()
    lone = _LONE_MARK_RE.match(text)
    if lone:
        # A mark said on its own (auto punctuation off): it belongs to the
        # previous chunk.
        mark = lone.group(1)
        return Command(comma=True) if mark == "," else Command(end=mark)
    if _SCRATCH_ALONE_RE.match(text):
        return Command(scratch=True)
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


# --- Cursor movement ("cursor back 3 words", "cursor to start") -----------

_NUMBERS = {
    "one": 1, "a": 1, "two": 2, "to": 2, "too": 2, "three": 3, "four": 4, "for": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_CURSOR_WORD_RE = re.compile(
    r"^\s*cursor[\s,]+(back|backward|backwards|left|forward|forwards|right)"
    r"(?:[\s,]+(\d+|" + "|".join(_NUMBERS) + r")(?:[\s,]+words?)?|[\s,]+words?)?[\s.!,]*$",
    re.IGNORECASE,
)
_CURSOR_EDGE_RE = re.compile(
    r"^\s*cursor[\s,]+to[\s,]+(?:the[\s,]+)?(start|beginning|end)"
    r"(?:[\s,]+of[\s,]+(?:the[\s,]+)?line)?[\s.!,]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CursorMove:
    kind: str  # "word", "start" or "end"
    direction: int = 0  # -1 back, +1 forward (words only)
    count: int = 0  # words


def parse_cursor(text):
    """A cursor command said as the whole utterance, or None."""
    text = (text or "").strip()
    m = _CURSOR_EDGE_RE.match(text)
    if m:
        return CursorMove("start" if m.group(1).lower() in ("start", "beginning") else "end")
    m = _CURSOR_WORD_RE.match(text)
    if not m:
        return None
    direction = -1 if m.group(1).lower().startswith(("back", "left")) else 1
    raw = (m.group(2) or "1").lower()
    count = int(raw) if raw.isdigit() else _NUMBERS[raw]
    return CursorMove("word", direction, max(1, count))


def _is_word(ch):
    return ch.isalnum() or ch == "_"


def word_target(text, cursor, move):
    """Where the cursor lands in text (readline word rules)."""
    if move.kind == "start":
        return 0
    if move.kind == "end":
        return len(text)
    i = cursor
    for _ in range(move.count):
        if move.direction < 0:
            while i > 0 and not _is_word(text[i - 1]):
                i -= 1
            while i > 0 and _is_word(text[i - 1]):
                i -= 1
        else:
            while i < len(text) and not _is_word(text[i]):
                i += 1
            while i < len(text) and _is_word(text[i]):
                i += 1
    return i


# --- Undo ("command undo", "undo that") ------------------------------------

_UNDO_RE = re.compile(
    r"^\s*(?:command[\s,]+(?:undo|and\s+do)|undo[\s,]+that)[\s.!,]*$", re.IGNORECASE)


def parse_undo(text):
    """Whether the whole utterance is an undo ("command undo", "undo that").
    "command and do" is how the model tends to write "command undo"."""
    return bool(_UNDO_RE.match(text or ""))
