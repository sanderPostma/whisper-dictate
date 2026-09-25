# Brief: achat line model: know multi-line drafts (without letting nudges steal them)

For the agent working in `agentic-chat`. Follows up the input-control socket
(`2026-09-24-achat-input-control-brief.md`) and edit-at-cursor
(`2026-09-24-achat-edit-at-cursor-brief.md`), both already built.

## Why

The operator dictates into agent prompts with whisper-dictate through the input socket.
Prompts are often multi-line: Alt+Enter, Ctrl+J, Shift+Enter, or a trailing backslash then
Enter. Every newline variant makes the draft **Unknown**, and it stays Unknown until Enter or
Ctrl+C. After that, `state` reports no text or cursor, and `edit` answers `unknown_line`. So
in a multi-line prompt:

- Live dictation cannot type through the socket. whisper-dictate now falls back to keystrokes
  on the pane: blind, with only approximate checks read off the screen.
- Spoken repairs ("scratch that", "command undo", "period", "command clear line") cannot be
  checked exactly. A real case: text on line 2, cursor moved back to line 1, "scratch that"
  gets refused.

The Unknown rule for newlines exists for a good reason. It was built so a nudge would never be
typed into a multi-line message and submitted as part of the operator's text. **That
protection must stay.** The fix is to separate two things the model currently merges:

- **modelled:** the wrapper knows the exact text (newlines included) and the cursor.
- **stealable:** a nudge may cut the draft, submit, and restore it (today's "Known").

## Change

1. Keep text and cursor across newline chords. Alt+Enter, Ctrl+J, Shift+Enter and
   backslash-then-Enter insert `"\n"` into the modelled text at the cursor, where the host is
   known to do that (Claude Code, Codex). The draft stays **modelled**, but it becomes
   **not stealable**: nudges keep being held exactly as they are held for Unknown today.
2. Movement and editing inside a multi-line draft: Left/Right, Backspace/Delete and typing
   work across `"\n"` as plain characters. Home/End (Ctrl+A/Ctrl+E) go to the start and end
   of the current hard line.
3. Up/Down: move to the same column on the previous or next hard line, clamped to that line's
   length, but only where the host behaves that way for hard lines. On the first line, Up is
   history recall in Claude Code: that stays Unknown, as does anything soft-wrap dependent. If
   a host's behaviour is not certain, leave Up/Down as Unknown. whisper-dictate falls back
   safely.
4. `state`: report `known: true` with the multi-line `text` and `cursor` (in characters,
   newlines counted) whenever the draft is modelled, and add `"stealable": bool`. Consider
   whether existing `subscribe` clients need `stealable` in their lines too.
5. `edit` with `at_cursor: true` works on a modelled multi-line draft: backspace erases to
   the left of the cursor across `"\n"`, and insert goes in at the cursor. `insert` still may
   not contain control bytes; whisper-dictate never inserts newlines through `edit`.
6. Enter still submits and resets the draft. Things that were Unknown for other reasons
   (paste, Tab, clicks, combining characters, word motions the hosts disagree on) stay
   Unknown.

## Tests (in `session.rs` / `line.rs`)

- `abc`, Alt+Enter, `def`: text `"abc\ndef"`, cursor 7, modelled, not stealable. A pending
  nudge is still held.
- The same draft with Left ×4: cursor 3, at the end of line 1. `edit` at_cursor
  backspace 3 → `"\ndef"`.
- Home on line 2 → cursor 4. End → cursor 7.
- Up from line 2, column 2 → line 1, column 2 (hosts where this is certain). Up on line 1
  stays Unknown.
- Enter on a modelled multi-line draft → submitted, reset to empty.
- Ctrl+C resets.
- The nudge regression: a nudge due while a multi-line draft is open is never typed into it.

## Docs

`docs/operator-guide.md`: the "Known"/"Unknown" section gets "modelled" versus "stealable",
the newline chords move from the Unknown list to the modelled-but-not-stealable list, and the
socket section documents `stealable` in `state`.

## Out of scope

Soft-wrap aware Up/Down, pastes, completion popups.
