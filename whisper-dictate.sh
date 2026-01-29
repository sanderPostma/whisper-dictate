#!/bin/bash
# Launcher script for Whisper Dictate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

exec nice -n -10 python3 whisper_dictate.py "$@"
