# Task 2 report: desktop.py

Build request id: `venv/bin/python -m pytest -q`

## RED
| test name | RED command | failure message | submitted before the first implementation edit: yes/no |
|---|---|---|---|
| all 13 in tests/test_desktop.py (e.g. test_x11_key_matches_xdotool_argv, test_wayland_key_calls_extension, test_get_desktop_picks_by_session_type) | `venv/bin/python -m pytest tests/test_desktop.py -q` | ModuleNotFoundError: No module named 'desktop' (collection error) | yes |

## Result
tests/test_desktop.py: 13 passed. Full suite: 485 passed (472 + 13).

## Concerns
- `call(method, args)` receives a plain tuple (as the brief's test specifies), not a GLib.Variant; the default D-Bus call builds the Variant from a per-method signature table. Fixed/decided.
- bind_hotkey/signal subscription and the default D-Bus call are not unit-tested (need a real bus); UNRESOLVED, covered by later manual testing.
- Real environment: tests make no real wl-copy/xclip/xdotool/D-Bus calls (all injected).
