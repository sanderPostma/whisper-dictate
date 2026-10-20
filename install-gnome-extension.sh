#!/usr/bin/env bash
# Install the whisper-dictate GNOME Shell helper extension for the current user.
set -euo pipefail
UUID="whisper-dictate@whisper-dictate.local"
SRC="$(cd "$(dirname "$0")" && pwd)/gnome-extension/$UUID"
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/gnome-shell/extensions/$UUID"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$SRC"/. "$DEST"/
echo "Installed to $DEST"
gnome-extensions enable "$UUID" || echo "(enable failed; it will be enabled after re-login: run 'gnome-extensions enable $UUID')"
echo "Log out and back in to load it (Wayland cannot restart gnome-shell)"
