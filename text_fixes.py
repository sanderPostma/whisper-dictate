"""Small text fixes applied to every transcription (pure).

fix_ticket_keys: spoken Jira keys ("v D X dash two eight three", "VDX two A
three") become the real key ("VDX-283"). Only after a configured project key,
so ordinary words are never turned into digits.

manual_punctuation: with automatic punctuation off, the model's punctuation
is dropped and only spoken marks ("comma", "period", ...) are written.
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


_SPOKEN_MARKS = {
    ("comma",): ",",
    ("period",): ".",
    ("full", "stop"): ".",
    ("question", "mark"): "?",
    ("exclamation", "mark"): "!",
    ("exclamation", "point"): "!",
    ("colon",): ":",
    ("semicolon",): ";",
    ("semi", "colon"): ";",
}
_MODEL_MARKS = ".,?!;:\u2026"
_KEEP_CASE = {"I", "I'm", "I'll", "I've", "I'd"}


def _lower_first(word):
    alpha = "".join(ch for ch in word if ch.isalpha())
    if word in _KEEP_CASE or (len(alpha) > 1 and alpha.isupper()):
        return word
    low = word[:1].lower()
    return low + word[1:] if len(low) == 1 else word


def _manual_line(line):
    # 1. The model's own punctuation goes; a word it capitalised only because
    #    it started a sentence there goes back to lower case.
    words, sentence_start = [], False
    for token in line.split():
        word = token.rstrip(_MODEL_MARKS)
        if word:
            words.append(_lower_first(word) if sentence_start else word)
        sentence_start = any(ch in ".?!\u2026" for ch in token[len(word):])
    # 2. Spoken marks become punctuation on the word before them.
    out, i, capital = [], 0, False
    while i < len(words):
        for spoken, mark in _SPOKEN_MARKS.items():
            if tuple(w.lower() for w in words[i:i + len(spoken)]) == spoken:
                if out:
                    out[-1] += mark
                else:
                    out.append(mark)
                capital = mark in ".?!"
                i += len(spoken)
                break
        else:
            word = words[i]
            out.append(word[:1].upper() + word[1:] if capital else word)
            capital = False
            i += 1
    return " ".join(out)


def manual_punctuation(text):
    """Text with the model's punctuation removed and spoken marks written."""
    return "\n".join(_manual_line(line) for line in (text or "").split("\n"))


# --- Shell commands ("L s minus L." -> "ls -l") -----------------------------

# Only words that are rarely the first word of an English sentence: an
# utterance starting with one of these is typed as a command line.
DEFAULT_SHELL_COMMANDS = [
    "ls", "cd", "pwd", "git", "sudo", "grep", "rg", "mkdir", "rmdir", "rm", "cp", "mv",
    "chmod", "chown", "ssh", "scp", "rsync", "curl", "wget", "kubectl", "npm", "npx",
    "pnpm", "pip", "systemctl", "journalctl", "apt", "tar", "ps", "htop", "vim", "nvim",
    "du", "df", "jq", "sed", "awk", "tmux", "gh", "bd", "achat", "gradle", "ln",
]
# How the model writes a command it did not hear as one.
DEFAULT_SHELL_ALIASES = {"ls": ["alas"], "sudo": ["pseudo"]}

_FLAG_DASH = r"(?:minus|dash|-)"


def _command_pattern(command, aliases):
    spelled = r"\.?[\s.]*".join(re.escape(c) for c in command) + r"\.?"
    forms = [f"(?P<cmd>{spelled})"]
    # An alias is an ordinary word too ("Alas, it failed"): not before a comma.
    forms += [f"(?P<alias{i}>{re.escape(a)})(?!\\s*,)" for i, a in enumerate(aliases)]
    return re.compile(r"^\s*(?:" + "|".join(forms) + r")(?![\w'])", re.IGNORECASE)


def fix_shell_command(text, commands, aliases=None):
    """A command line when the utterance starts with a known shell command:
    lower case, no punctuation, spoken flags written ("minus l" -> "-l",
    "dash dash force" -> "--force"). None when it is not one."""
    aliases = {k.lower(): v for k, v in (aliases or {}).items()}
    for command in commands or ():
        command = command.lower()
        m = _command_pattern(command, aliases.get(command, ())).match(text or "")
        if not m:
            continue
        rest = text[m.end():]
        rest = re.sub(r"[,?!;]|\.(?=\s|$)", " ", rest).lower()
        rest = re.sub(rf"(?<![\w-]){_FLAG_DASH}\s*{_FLAG_DASH}\s*(?=\w)|\bdouble\s+dash\s+", " --", rest)
        rest = re.sub(rf"(?<![\w-]){_FLAG_DASH}\s*(?=\w)", " -", rest)
        return " ".join([command, *rest.split()])
    return None
