# Brief: achat input socket — allow `edit` at the cursor

For the agent working in `agentic-chat`. Follows up
`2026-09-24-achat-input-control-brief.md`, which is already built (`control.rs`, `Session::edit`).

## Why

The operator dictates with whisper-dictate into an `achat run` prompt. They often move the
cursor back into a sentence to add words:

    His is a |test sentence.        (| = cursor)

Today `edit` refuses with `cursor_not_at_end` unless the cursor is at the end of the line, so
whisper-dictate cannot type there. It already reads `state` (`text`, `cursor`), which gives
it the text before and after the cursor. It needs `edit` to work at the cursor too.

## Change

Add an opt-in field to `edit`:

```json
{"op":"edit","expect_rev":42,"backspace":0,"insert":"quick ","at_cursor":true}
```

- `at_cursor` absent or `false` keeps today's behaviour exactly, including `cursor_not_at_end`.
- `at_cursor: true` skips the `cursor_not_at_end` check. `backspace` erases characters to the
  left of the cursor, then `insert` goes in at the cursor, which ends up right after the
  inserted text with the rest of the line still after it. This is what `Line::backspace`
  and `Line::insert_char` already do; the decoder replay in `Session::edit` needs no change.
- `range` for `at_cursor: true` becomes `backspace > cursor` (not `> len`). A client must
  never erase past the start of the line.
- Everything else is unchanged: `busy`, `unknown_line`, `conflict`, `invalid_insert`, and the
  final "replayed line must still be Known" check.

Please check `busy`, `pending_restore` and the draft steal/restore path with a mid-line
cursor too. The restore already puts the cursor back (Left presses); an `edit` must still be
refused (`busy`) while that is in flight.

## Tests (in `session.rs`)

- Cursor mid-line, `at_cursor: true`, `backspace 0` → text inserted at the cursor, cursor
  after it, tail intact, rev bumped once per character.
- `at_cursor: true` with `backspace 2` mid-line → erases the 2 characters left of the cursor
  only.
- `at_cursor: true`, `backspace` greater than `cursor` but at most `len` → `range`.
- No `at_cursor` with the cursor mid-line → still `cursor_not_at_end`.
- `at_cursor: true` with the cursor at the end → same result as today's `edit`.

## Docs

`docs/operator-guide.md`, "The input-control socket": describe `at_cursor` and the new
`range` rule, with one example like the one above. Also update the "cursor sits at the end"
wording in the `edit` paragraph and the error table's `cursor_not_at_end` row ("an `edit`
only ever appends"), and state that `cursor` in `state` counts characters (code points,
like `Line`'s `Vec<char>`): the client now splits `text` at it.

## Out of scope

Moving the cursor (`cursor` op), `clear`, `submit`: still reserved, not needed for this.
