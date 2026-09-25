# Whisper Dictate 🎤

Voice-to-text dictation with system tray integration. Press hotkey to record, press again to transcribe and type.

Uses OpenAI Whisper, distil-whisper, or Qwen3-ASR for speech recognition. Qwen3-ASR can take a vocabulary/context prompt so developer jargon is more likely to land correctly.

## Features

- **System tray icon** - Shows recording status (white = idle, red = recording)
- **Global hotkey** - Press to start/stop recording (default: Alt+D)
- **Audio feedback** - Different beep tones for start/stop recording
- **Multiple output modes** - Type directly, copy to clipboard, or both
- **Text replacements** - Post-process transcriptions (e.g., "slash" → "/", "enter" → execute)
- **Desktop notifications** - Visual feedback for recording status
- **Configurable** - Customize hotkey, model size, language, output mode
- **ASR context packs** - Bias transcription toward developer jargon (or a custom vocabulary file)

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt install xdotool xclip libportaudio2 python3-gi gir1.2-appindicator3-0.1 gir1.2-keybinder-3.0
```

### Install

```bash
# Clone the repo
cd /path/to/whisper-dictate

# Create virtual environment with system packages (needed for GTK)
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
./whisper-dictate.sh
```

Or with options:
```bash
./whisper-dictate.sh --mode both --model small --language nl
```

### Default Hotkey

**Alt+D** - Press to start recording, press again to stop and transcribe

### System Tray Menu

- **Record/Stop** - Toggle recording
- **Mode: Type** - Check to type into active window
- **Mode: Clipboard** - Check to copy to clipboard
- **Model** - Whisper, distil-whisper, or Qwen3-ASR 1.7B
- **Settings** - Open config file

Pressing Enter by voice works in one-shot and live mode: end with "press enter" (or
"press return", or "engage" after a short pause), or say "enter" on its own. A bare "enter" inside a sentence stays text.
"new line" / "new paragraph" insert line breaks through the replacements file.

Cursor commands, said as a whole utterance (one-shot or live): "cursor back" / "cursor
forward" (one word), "cursor back 3 words" / "cursor forward three words" (digits or number
words), "cursor to start" / "cursor to end" (also "the beginning", "… of the line"). In achat
prompts the move is counted from the known line and sent as arrow keys; other WezTerm panes
get Alt+B / Alt+F and Ctrl+A / Ctrl+E; other windows Ctrl+Left / Ctrl+Right and Home / End.
Dictation continues at the new cursor position.

Undo, said as a whole utterance: "command undo", "undo that" or just "undo" (also as misheard: "comment undo"). Reverts the last dictation
step (an utterance, or a repair such as "scratch that"); say it again to go further back.
Typical use: a word came out wrong, say "undo that" and repeat it. Live mode, and one-shot in
achat prompts (the line must still end with exactly that text). Enter and cursor moves are
not undone.

Clipboard, said as a whole utterance: "command paste", "command copy", "command cut" press
Ctrl+V / Ctrl+C / Ctrl+X in the focused window. In terminals (WezTerm, GNOME Terminal,
kitty, ...) that is Ctrl+Shift+V / Ctrl+Shift+C instead, since Ctrl+C would interrupt the
program; cut does nothing there. After a paste or cut, live dictation does not rewrite
anything typed before it.

Jira keys: list your project keys in `ticket_keys` (e.g. `["VDX"]`) and spoken keys are
written properly: "v D X dash two eight three", "VDX two eighty three" and the misheard
"v D X dash two A three" all become `VDX-283`. Only text after one of your keys is touched.
`ticket_aliases` adds how the model writes a key it did not spell out, e.g.
`{"VDX": ["videx"]}` turns "videx two for two" into `VDX-242` (only when a number follows).

Vocabulary: every transcription is sent with the spoken command words (slash, command, scratch
that, ...), your `ticket_keys` and your `vocabulary` list (e.g. `["Sphereon", "achat"]`), so
Qwen prefers those spellings. A misheard word already typed on the line is sent as context
too and tends to come back when you repeat yourself: say "undo that" first, then say it again.

Slash commands: "slash clear" types `/clear`, also when heard as "flash clear", "splash
clear" or "less clear" (the whole utterance, and only for commands in `slash_commands`).

Shell commands: an utterance that starts with a command from `shell_commands` is typed as a
command line: lower case, no punctuation, spoken flags written. "L s minus L." → `ls -l`,
"Git branch." → `git branch`, "Git push dash dash force" → `git push --force`, "Sudo." →
`sudo`. Spelled-out letters ("L s", "C D") count, and `shell_aliases` holds how the model
writes a command it did not hear as one (`{"ls": ["alas"], "sudo": ["pseudo"]}`). An alias
followed by a comma stays a word ("Alas, it failed."). The default list leaves out commands
that are common English words (make, find, cat, head, top, ...); add them yourself if you
want them. In live mode the command line closes the correction window, so a later correction
never retypes it as a sentence.

Auto punctuation: say "auto punctuation off" (whole utterance, one-shot or live) and the
model's punctuation is dropped; only what you say is written: "comma", "period" / "full
stop", "question mark", "exclamation mark", "colon", "semicolon". A word the model
capitalised only because it started a sentence goes back to lower case (proper nouns too,
except "I" and acronyms). "auto punctuation on" switches back. Also a checkbox in the tray;
the setting (`auto_punctuation`) is saved. With it off, "period" and "comma" always count as
marks, also in "the trial period".

## Live dictation

Press the dictation hotkey (`<Alt>d`) twice within a second (double beep), or pick
**⚡ Live dictation** in the tray, then talk. A single press still records one-shot; the
second quick press drops that recording and switches to live. Text is typed at each pause
(~0.6 s). With the remote server up, the last ≤ 12 s are re-transcribed after each pause
and recently typed words are corrected in place (backspace + retype). The next press of the
hotkey stops live dictation. `live_double_press_s` sets the window (default 1.0 s);
`live_hotkey` can add a separate live hotkey (off by default).

- WezTerm pane running `achat run`: typed into the agent's prompt line through achat's
  input-control socket. Every edit is checked against the line's revision, so typing by
  hand or an incoming nudge is never overwritten: dictation just continues after it,
  and text already in the prompt is used as context (spacing, capitalisation). You can
  move the cursor back into a sentence and dictate there: the text before the cursor is
  the context, and a space is added before the word that follows. Mid-line dictation
  needs an achat build whose `edit` supports `at_cursor`.
  Needs an `achat` build with the input socket; restart old `achat run` sessions.
- Other WezTerm panes: typed into the pane focused at start via `wezterm cli send-text`.
- Other windows: `xdotool`.
- Switching window or pane mid-session freezes what was typed and continues in the
  new place, append-only.
- The local fallback model types but does not correct.
- Outside achat, corrections work by blindly backspacing up to `live_max_backspace` characters and
  retyping — don't type by hand in the same window while a live session is running,
  or the backspaces can eat your manual edits.

Spoken repairs, for when a thinking pause put a full stop in the middle of a sentence
(automatic punctuation stays the default; say these only to fix it):

| Say | Where | Effect |
|---|---|---|
| "… period" / "full stop" / "question mark" / "exclamation mark" | end of a chunk | join it to the previous chunk (the stray `.` goes, lowercase) and end it so; said alone, it replaces the mark the text ends with (`.` `,` `;` `:`) or adds one |
| "comma …" | start of a chunk | the previous `.` becomes `,` and the chunk continues the sentence |
| "scratch that" (also heard as "stretch that"; or "scratch" alone) | end of a chunk | delete the current sentence (or the one just finished) |
| "command clear", "command clear line", "scratch all", "scratch whole line" | whole utterance | clear the whole input line, typed text included (achat and WezTerm panes) |
| "… press enter" / "press return" / "engage", or "enter" said alone | end of a chunk | type the words, then press Enter (submit); "engage" only alone or after a pause |

Say the command words as their own piece, with a short pause before them ("… test more —
period"): "period", "full stop" and "comma" are ordinary words too, so "the trial period"
or "comma separated values" stay text. "Scratch that" only deletes text that was typed;
words spoken just before it in the same breath are simply dropped.

Repairs only ever rewrite what live dictation typed itself, up to
`live_max_command_backspace` characters: in achat prompts the line must still end with
exactly that text; in other WezTerm panes the cursor row must still show it; in other
windows (xdotool) a repair only reaches back within the current correction window, since
you may have typed elsewhere during a pause. A mark said alone ("question mark") in a WezTerm pane is also
allowed when the text shows at the end of a row near the bottom of the screen: TUIs like
Claude Code draw their own cursor, so the terminal's cursor row can be another row.

Config keys: `live_pause_ms`, `live_max_chunk_s`, `live_window_max_s`,
`live_commit_pause_ms`, `live_max_backspace`, `live_max_command_backspace`, `live_corrections`,
`live_committed_context_chars`, `live_hotkey`, `live_double_press_s`.

## Configuration

### Main Config

Edit `~/.config/whisper-dictate/config.json`:

```json
{
  "hotkey": "<Alt>d",
  "model": "base",
  "language": "en",
  "language_models": {
    "en": "distil-large-v3",
    "nl": "large"
  },
  "output_mode": "type",
  "context_pack": "developer",
  "context_prompt": ""
}
```

| Setting | Description | Options |
|---------|-------------|---------|
| `hotkey` | Global hotkey | `<Ctrl>`, `<Shift>`, `<Alt>`, `<Super>` + key |
| `model` | ASR model | Whisper sizes, `distil-*`, or `qwen3-asr-1.7b` |
| `language` | Language code | `en`, `nl`, `de`, etc. |
| `language_models` | Last selected model per language | JSON object keyed by language code |
| `output_mode` | How to output text | `type`, `clipboard`, `both` |
| `context_pack` | Vocabulary prompt sent to the ASR model | `none`, `developer`, `custom` |
| `context_prompt` | Extra terms always appended to the pack | free text |

Qwen3-ASR (`qwen3-asr-1.7b`) needs `transformers>=5.13` and about 6–10 GB VRAM on GPU (CPU works, but is slow). First load downloads `Qwen/Qwen3-ASR-1.7B-hf` (~4.5 GB). Select it from the tray **Model** menu; if remote mode is on, that Qwen id is sent to the GPU server.

ASR context is set in config files (no tray control for now):

| Pack | Prompt |
|------|--------|
| `none` | No vocabulary hint (unless `context_prompt` is set) |
| `developer` | Built-in git/K8s/Python/JS/CLI jargon list |
| `custom` | `~/.config/whisper-dictate/context.txt` |

Set `context_pack` / `context_prompt` in `config.json`. `context_prompt` is appended in every case. Qwen receives this as `prompt`; Whisper as `initial_prompt`. Replacements in `replacements.yml` still run after transcription.

### Text Replacements

Create `~/.config/whisper-dictate/replacements.yml` to post-process transcriptions:

```yaml
replacements:
  # Symbols
  "slash ": "/"
  "backslash ": "\\"
  
  # Commands - say "enter" to execute in terminal
  ", enter.": "\n"
  " enter": "\n"
  
  # Fix common mishearings
  "djamal": "yaml"
```

See `replacements.example.yml` for a full example.

**Features:**
- Case-insensitive matching
- Handles Whisper's auto-punctuation (e.g., ", enter." → newline)
- Trailing periods are automatically removed
- Add trailing space to patterns to avoid partial matches

## Model Comparison

| Model | Speed (CPU) | Accuracy | VRAM |
|-------|-------------|----------|------|
| tiny | ~1s | Good | ~1GB |
| base | ~2-3s | Better | ~1GB |
| small | ~5-8s | Great | ~2GB |
| medium | ~15-20s | Excellent | ~5GB |
| large | ~30s+ | Best | ~10GB |
| qwen3-asr-1.7b | GPU: fast / CPU: slow | Strong, promptable | ~6–10GB VRAM |

## Autostart

The installer creates a desktop entry. To enable autostart:

```bash
cp ~/.local/share/applications/whisper-dictate.desktop ~/.config/autostart/
```

## Troubleshooting

### No audio input
- Check your microphone: `arecord -l`
- Test recording: `rec test.wav`

### Hotkey not working
- Ensure you're on X11 (not Wayland)
- Check if another app uses the same hotkey

### Tray icon not showing
- Install appindicator: `sudo apt install gir1.2-appindicator3-0.1`
- Some desktop environments need extensions for tray support

## License

MIT
