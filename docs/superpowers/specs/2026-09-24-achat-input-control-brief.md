# Brief: achat wrapper — robust input-line model + local input-control socket

Audience: the agent working in `agentic-chat` (crate `achat`, file `crates/achat/src/pty.rs`).
Requested by: the operator, 2026-09-24. Consumer: whisper-dictate "live mode"
(`/mnt/nvme2tb/DEV/mine/whisper-dictate`, branch `live-dictation`).

## Why

1. **Robustness (nudges).** `Draft` (`pty.rs:~95-230`) is `Clean(String) | Dirty`. Any
   editing escape (arrows, Home/End, word-jump, Delete) turns a non-empty draft `Dirty`,
   after which nudges are held until the operator submits or clears. In practice the draft
   goes dirty often, so nudges stall.
2. **Live dictation.** whisper-dictate will type speech into the wrapped agent's composer
   while the operator talks, and *rewrite the tail* of what it typed when a later
   transcription corrects earlier words (backspace N + retype). Today it can only do this
   blind via `wezterm cli send-text`; it cannot know whether the operator moved the cursor
   or typed in between. The wrapper already sees every byte going to the child, so it is
   the right place to answer "is the line still what I think it is?".
3. **Later (not now, but the model must allow it):** voice commands such as "clear line",
   "cursor before current word", "go to end of line".

## Part 1 — Line model with a cursor (replaces `Clean/Dirty`)

Model the composer line as `text: Vec<char>` + `cursor: usize` + `state: Known | Unknown`.

- Printable chars insert at `cursor`; DEL/BS delete before cursor; Ctrl-W kills word before
  cursor; Ctrl-U / Ctrl-C / Enter reset to empty `Known`.
- Interpret the common editing keys instead of giving up: Left/Right, Home/End (all the
  usual encodings: `CSI D/C/H/F`, `SS3 H/F`, `CSI 1~/4~/7~/8~`), Ctrl-A/Ctrl-E,
  word-left/right (`CSI 1;5D/C`, `ESC b/f`), Delete (`CSI 3~`), Ctrl-K.
- Only genuinely unknowable input goes `Unknown`: Up/Down/history, Tab completion, paste
  (bracketed), mouse clicks in the composer, anything unparsed. Empty-line exemptions
  stay as today (`dirty_if_text`).
- Multi-line composers (Claude Code: Shift+Enter / `\` newline) — decide explicitly; if
  unsupported, go `Unknown` rather than guess.
- Keep a monotonically increasing **`rev`** bumped on every change to the line (operator
  keys *and* injected edits). This is the concurrency token for Part 2.
- Nudge logic: steal a `Known` line by moving to end (End), erasing `len` chars, typing the
  nudge, then restoring text *and cursor position*. `Unknown` behaves like today's `Dirty`.
- Table-driven unit tests: byte sequence in → (text, cursor, state) out, per platform
  encoding (the kitty/Windows VT cases in recent commits must keep passing).

Acknowledged limit: this is a reconstruction of what the TUI shows, not ground truth
(autocomplete popups, TUI-side rewrites). `Unknown` is the safety valve; err toward it.

## Part 2 — Local input-control socket (one per wrapped session)

A Unix socket owned by the `achat run` process, accepting newline-delimited JSON
requests, one JSON response per request. Local user only (0600, in a 0700 dir).

Discovery: `$XDG_RUNTIME_DIR/achat/input/<pid>.sock` plus `<pid>.json` sidecar with
`{"pid", "child_cmd", "cwd", "wezterm_pane": $WEZTERM_PANE, "kitty_window_id":
$KITTY_WINDOW_ID, "tty"}`. whisper-dictate maps WezTerm's `focused_pane_id` to the socket
via `wezterm_pane`. Remove both files on exit; tolerate stale files (connect fails → skip).

Ops (minimum for live dictation):

```json
{"op":"state"}
→ {"ok":true,"rev":41,"known":true,"text":"I think we should merge the brunch","cursor":34}

{"op":"edit","expect_rev":41,"backspace":4,"insert":"anch"}
→ {"ok":true,"rev":42}
→ {"ok":false,"error":"conflict","rev":43}        // line changed since rev 41: nothing written
→ {"ok":false,"error":"unknown_line"}             // state Unknown: nothing written
```

- `edit` is compare-and-swap: applied only if `rev == expect_rev`, the line is `Known` and
  the cursor is at end of text (v1). Otherwise **write nothing** and report why.
- `expect_rev` optional → append-only insert still allowed when `Known` (fast chunks), but
  whisper-dictate will normally pass it.
- Writes go through the existing single child-input channel (`pty.rs:~1583`) so they
  serialize with operator keystrokes and nudges; `rev` is checked and bumped under the same
  lock that orders those writes. An edit must never interleave with a nudge's steal/restore.
- Send injected text as plain keystrokes (not bracketed paste) so the TUI treats it like
  typing; respect the existing paste-burst/`ENTER_SETTLE` lessons — `edit` never sends
  Enter.
- Optional (nice now, needed later): `{"op":"subscribe"}` streaming `{"rev","known"}` on
  every change, so the client can lock its window the moment the operator touches keys.

Reserved for later (design the model so these are trivial, do not build yet):
`{"op":"cursor","to":"word_start"|"line_end"|...}`, `{"op":"clear"}`,
`{"op":"submit"}`.

## Out of scope for the achat agent

Speech, pause detection, diffing corrections — that all stays in whisper-dictate. The
wrapper only models the line and applies guarded edits.
