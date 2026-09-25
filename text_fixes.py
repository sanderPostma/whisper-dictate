"""Small text fixes applied to every transcription (pure).

fix_ticket_keys: spoken Jira keys ("v D X dash two eight three", "VDX two A
three") become the real key ("VDX-283"). Only after a configured project key,
so ordinary words are never turned into digits.
"""

import re

_DIGIT_WORDS = {
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
# How the model mishears digits inside a ticket number. Only taken between
# other number words, never as the last one ("VDX 283 for review").
_MISHEARD = {"a": 8, "ate": 8, "to": 2, "too": 2, "for": 4, "won": 1, "tree": 3}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}

_SEPARATOR = re.compile(r"[\s,]*(?:(?:-|–|dash|hyphen|minus)(?![A-Za-z])[\s,]*)?", re.IGNORECASE)
_TOKEN = re.compile(r"(?:[\s,]+|[\s,]*[-–][\s,]*)?(\d+|[A-Za-z]+)")


def _key_pattern(key, aliases=()):
    letters = r"\.?[\s.]*".join(re.escape(c) for c in key) + r"\.?"
    spoken = [re.escape(a).replace(r"\ ", r"\s+") for a in aliases if a.strip()]
    body = "|".join([letters, *spoken])
    return re.compile(r"(?<![A-Za-z0-9])(?:" + body + r")(?![A-Za-z0-9])", re.IGNORECASE)


def _read_number(text, pos):
    """(digits, end) of the ticket number starting at pos, or (None, pos)."""
    parts = []  # (digits, end, misheard)
    tens_open = False  # the last part was a round ten a unit may join ("eighty three")
    hundreds = None
    while True:
        m = _TOKEN.match(text, pos)
        if not m:
            break
        tok = m.group(1).lower()
        if tok.isdigit():
            parts.append((tok, m.end(), False))
            tens_open = False
        elif tok in _DIGIT_WORDS or tok in _MISHEARD:
            value = _DIGIT_WORDS.get(tok, _MISHEARD.get(tok))
            misheard = tok not in _DIGIT_WORDS
            if tens_open and value and not misheard:
                digits, _, _ = parts.pop()
                parts.append((str(int(digits) + value), m.end(), False))
            else:
                parts.append((str(value), m.end(), misheard))
            tens_open = False
        elif tok in _TEENS:
            parts.append((str(_TEENS[tok]), m.end(), False))
            tens_open = False
        elif tok in _TENS:
            parts.append((str(_TENS[tok]), m.end(), False))
            tens_open = True
        elif tok == "hundred" and parts and hundreds is None:
            hundreds = int("".join(p[0] for p in parts))
            parts = []
            pos = m.end()
            hundreds_end = m.end()
            continue
        else:
            break
        pos = m.end()
    while parts and parts[-1][2]:
        parts.pop()  # a trailing "for" / "to" / "a" is an ordinary word
    if hundreds is not None:
        rest = int("".join(p[0] for p in parts) or "0")
        end = parts[-1][1] if parts else hundreds_end
        return str(hundreds * 100 + rest), end
    if not parts:
        return None, pos
    return "".join(p[0] for p in parts), parts[-1][1]


def fix_ticket_keys(text, keys, aliases=None):
    """Rewrite spoken ticket keys for the given project keys to KEY-123.

    aliases: {"VDX": ["videx", ...]} - how the model writes a key it did not
    spell out. Like the letters, they only count when a number follows.
    """
    aliases = {k.upper(): v for k, v in (aliases or {}).items()}
    for key in keys or ():
        key = key.upper()
        pattern = _key_pattern(key, aliases.get(key, ()))
        out, pos = [], 0
        for m in pattern.finditer(text):
            if m.start() < pos:
                continue
            sep = _SEPARATOR.match(text, m.end())
            digits, end = _read_number(text, sep.end())
            if digits is None:
                continue
            out.append(text[pos:m.start()])
            out.append(f"{key}-{digits}")
            pos = end
        text = "".join(out) + text[pos:]
    return text
