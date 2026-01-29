# Whisper Dictate 🎤

Voice-to-text dictation with system tray integration. Hold a hotkey to record, release to transcribe and paste.

Uses OpenAI's Whisper for accurate speech recognition with full context understanding.

## Features

- **System tray icon** - Shows recording status (white = idle, red = recording)
- **Global hotkey** - Hold to record, release to transcribe
- **Auto-paste** - Transcribed text is automatically typed into the active window
- **Desktop notifications** - Visual feedback for recording/transcription status
- **Configurable** - Customize hotkey, model size, language

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt install xdotool xclip libportaudio2 python3-gi gir1.2-appindicator3-0.1
```

### Install

```bash
# Clone the repo
cd /path/to/whisper-dictate

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Activate venv and run
source venv/bin/activate
python whisper_dictate.py
```

Or use the launcher script:
```bash
./whisper-dictate.sh
```

### Default Hotkey

**Ctrl+Shift+D** - Hold to record, release to transcribe and paste

### Configuration

Edit `~/.config/whisper-dictate/config.json`:

```json
{
  "hotkey": "<ctrl>+<shift>+d",
  "model": "base",
  "language": "en",
  "sample_rate": 16000,
  "paste_method": "xdotool"
}
```

#### Options

| Setting | Description | Options |
|---------|-------------|---------|
| `hotkey` | Key combination to hold | `<ctrl>`, `<shift>`, `<alt>`, `<super>`, letters |
| `model` | Whisper model size | `tiny`, `base`, `small`, `medium`, `large` |
| `language` | Language code | `en`, `nl`, `de`, etc. |
| `paste_method` | How to insert text | `xdotool` (types), `xclip` (pastes) |

#### Model Comparison

| Model | Speed (CPU) | Accuracy | VRAM |
|-------|-------------|----------|------|
| tiny | ~1s | Good | ~1GB |
| base | ~2-3s | Better | ~1GB |
| small | ~5-8s | Great | ~2GB |
| medium | ~15-20s | Excellent | ~5GB |
| large | ~30s+ | Best | ~10GB |

## Autostart

To start on login, add to your desktop's startup applications:

```bash
/home/sander/DEV/mine/whisper-dictate/whisper-dictate.sh
```

Or create a systemd user service:

```bash
mkdir -p ~/.config/systemd/user
cp whisper-dictate.service ~/.config/systemd/user/
systemctl --user enable whisper-dictate
systemctl --user start whisper-dictate
```

## Troubleshooting

### No audio input
- Check your microphone: `arecord -l`
- Test recording: `rec test.wav`

### Hotkey not working
- May need to run without Wayland (X11) for global hotkeys
- Try running with sudo for input access (not recommended for daily use)

### Tray icon not showing
- Install appindicator: `sudo apt install gir1.2-appindicator3-0.1`
- Some desktop environments need extensions for tray support

## License

MIT
