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
