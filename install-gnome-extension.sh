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
# A global "all user extensions off" switch silently overrides per-extension
# enabling, leaving the helper INITIALIZED and the app on the XWayland fallback.
if [ "$(gsettings get org.gnome.shell disable-user-extensions)" = "true" ]; then
    echo "User extensions were globally disabled — re-enabling them"
    gsettings set org.gnome.shell disable-user-extensions false
fi
gnome-extensions enable "$UUID" || echo "(enable failed; it will be enabled after re-login: run 'gnome-extensions enable $UUID')"

sleep 1
if [ "$(gnome-extensions info "$UUID" | awk -F': ' '/State:/{print $2}')" = "ACTIVE" ]; then
    echo "Extension is ACTIVE — restart whisper-dictate to pick it up"
else
    echo "Log out and back in to load it (Wayland cannot restart gnome-shell)"
fi
