#!/bin/bash
# Toggle Whisper Dictate recording
# Sends signal to running instance

PIDFILE="/tmp/whisper-dictate.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill -USR1 "$PID"
        exit 0
    fi
fi

notify-send "Whisper Dictate" "Not running! Start it first."
