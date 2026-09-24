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

_END_RE = re.compile(
    r"(?:^|[\s,.?!]+)(" + "|".join(k.replace(" ", r"\s+") for k in _END_WORDS) + r")[\s.?!,]*$",
    re.IGNORECASE,
)
_COMMA_RE = re.compile(r"^\s*(?:comma\b[\s,.]*|,\s*)", re.IGNORECASE)
_SCRATCH_RE = re.compile(r"(?:^|[\s,.?!]+)scratch\s+that[\s.?!,]*$", re.IGNORECASE)


@dataclass(frozen=True)
class Command:
    rest: str = ""  # the words to type (command words removed)
    comma: bool = False  # the previous sentence end becomes a comma
    end: str = ""  # join to the previous chunk and end with this
    scratch: bool = False  # delete the current sentence


def parse_command(text):
    """The spoken command in a chunk's text, or None for plain dictation."""
    text = (text or "").strip()
    if _SCRATCH_RE.search(text):
        return Command(scratch=True)
    end = ""
    m = _END_RE.search(text)
    if m:
        end = _END_WORDS[" ".join(m.group(1).lower().split())]
        text = text[:m.start()]
    comma = False
    m = _COMMA_RE.match(text)
    if m:
        comma = True
        text = text[m.end():]
    if not end and not comma:
        return None
    return Command(rest=text.strip(), comma=comma, end=end)
