# Live dictation with rolling correction — design

Date: 2026-09-24 · Branch: `live-dictation` · Status: approved in brainstorm, awaiting spec review

## Goal

While the operator speaks continuously, text is typed at each natural pause. Each new
piece is transcribed with the preceding text as context, and already-typed words in the
current sentence are corrected in place when later audio disambiguates them.

Success: talk for ~30 s with natural pauses into a WezTerm pane; text appears within
~0.5 s of each pause, reads as coherent sentences, and misheard words in the current
sentence sometimes fix themselves ~1–2 s later. The existing one-shot mode is unchanged.

## Measured constraints

Qwen3-ASR 1.7B on the GPU server (192.168.1.28, HF `generate`, 2026-09-23):
latency ≈ 0.11 s per second of audio (2 s → 0.20 s, 10 s → 1.10 s, 20 s → 2.36 s),
dominated by decoding (~45 ms/word). ~300 chars of text prompt adds ~0.18 s.
Hence: fast pass on the new chunk only, correction pass on a window capped at ~12 s.

## Approach

Hybrid "fast then correct":

- **Fast pass** — at each pause, transcribe only the new chunk with prompt =
  jargon context + committed text + current window text; type the result immediately.
- **Correction pass** — then re-transcribe the whole current window (audio since the last
  commit, ≤ 12 s) with prompt = jargon context + committed text; diff against what was
  typed for that window and rewrite only the differing tail.
- Text older than the window is **committed**: never edited again, used only as text
  context (last ~400 chars).

Server protocol is unchanged (each call is an ordinary request with `prompt`).

## Components

### `live_segmenter.py` — pause detection (pure)
`LiveSegmenter(sample_rate, pause_ms=600, max_chunk_s=8, threshold)`.
`feed(block) -> list[Event]`; events `ChunkReady(audio, t0, t1)` on a pause after speech,
or forced at `max_chunk_s`. Leading/trailing silence trimmed to ~150 ms. RMS threshold
reuses `silence_threshold` (with a sane default when that is 0). No I/O.

### `live_session.py` — window bookkeeping (pure, deterministic)
State: `committed_text`, window audio (list of chunks with times), window pieces
`[(t1, text)]` = what was typed per chunk, `typed_window` = concatenation.

- `add_fast_result(t1, raw_text) -> Edit` — normalise (strip context echo; if the window
  does not end in `.?!` lowercase first letter; leading space unless window empty and
  committed text empty/ends in newline); returns `Edit(backspace=0, insert=text)`.
- `apply_correction(t_upto, raw_text) -> Edit | None` — compares with the typed text for
  pieces with `t1 <= t_upto`; common prefix → `backspace = len(typed_window) - prefix`,
  `insert = corrected[prefix:] + pieces after t_upto`. Returns `None` if identical, stale
  (window committed since), or `backspace > live_max_backspace` (default 80).
- `should_commit(pause_s, focus_ok) -> bool` — true when window text ends in `.?!` and
  pause ≥ 1.2 s, window audio > `live_window_max_s` (12), focus/pane changed, or stop.
- `commit()` — move window text to `committed_text` (trim to last 400 chars), clear window.
- `fast_prompt()`, `correction_prompt()`, `window_audio()`.

### `live_output.py` — output targets
Interface: `send(edit) -> bool`, `still_focused() -> bool`.
- `WezTermTarget` — pins `focused_pane_id` (`wezterm cli list-clients --format json`) at
  start; `wezterm cli send-text --pane-id N --no-paste` with `"\x7f" * backspace + insert`.
  `still_focused()` = focused pane id unchanged.
- `XdotoolTarget` — pins X window id; `xdotool key --repeat N BackSpace` then
  `xdotool type --delay 8`. `still_focused()` = active window unchanged.
- Target chosen from the focused window class at start (`wezterm` → WezTerm, else xdotool).
  Kitty and clipboard modes are not supported in live mode v1 (fall back to xdotool).
- Phase 2: `AchatTarget` (see below).

### Wiring in `whisper_dictate.py`
- New tray item + hotkey entry (`toggle-recording.sh --live` or a separate script) to
  start/stop live mode; one-shot mode untouched. Live and one-shot are mutually exclusive.
- Audio callback feeds `LiveSegmenter`; `ChunkReady` enqueues a fast job.
- **One transcription worker** with a priority queue: fast jobs before corrections; a
  correction is enqueued only after its chunk's fast result was typed; a queued correction
  is replaced by a newer one.
- **One output queue** (GLib main thread or a dedicated thread) applies `Edit`s strictly in
  order; before each correction edit, checks `still_focused()`; if false → `commit()` and
  continue append-only with the new target state.
- Stop: flush the last chunk (fast), run one final correction, commit, beep.
- Remote down → local fallback model for fast passes only; corrections disabled.

## Config (`config.json`)
`live_pause_ms` 600 · `live_max_chunk_s` 8 · `live_window_max_s` 12 ·
`live_commit_pause_ms` 1200 · `live_max_backspace` 80 · `live_corrections` true ·
`live_committed_context_chars` 400.

## Error handling
- Transcription error on fast pass → log and type nothing, but keep the chunk's audio in
  the window so the next correction can still recover it; notify once per session.
- Correction error or timeout → skip silently; next correction covers it.
- Output command fails → commit window, stop corrections for the session, keep appending.
- Empty/whitespace results → no edit.

## Known risks / limits
- Cursor moved by mouse click inside the same pane is undetectable in v1 (phase 2 fixes it).
- TUI autocomplete / auto-inserted characters desync backspace counts (mitigated by the
  backspace cap; phase 2 detects it).
- Corrections cause visible flicker in the current sentence — by design.
- GPU contention: two calls per pause; fine at measured latencies.

## Testing
- `tests/test_live_segmenter.py` — synthetic sine/silence blocks: pause detection, forced
  split, trimming, threshold default.
- `tests/test_live_session.py` — table tests for normalisation, echo stripping, correction
  diff (incl. pieces typed after `t_upto`), backspace cap, stale corrections, commit rules,
  prompt construction.
- `tests/test_live_output.py` — command construction with `subprocess.run` mocked.
- Manual: WezTerm pane running Claude Code via `achat run`; checklist in the plan.

## Phase 2 (separate repo, in progress by another agent)
`achat run` gains a cursor-aware line model and a per-session input-control socket with a
compare-and-swap `edit` op — see `2026-09-24-achat-input-control-brief.md`.
whisper-dictate then adds `AchatTarget`: discover socket by `wezterm_pane`, send fast
chunks as `edit` with `expect_rev`, and on `conflict`/`unknown_line` commit the window.
Preferred over `WezTermTarget` when a socket for the focused pane exists.

## Later (out of scope)
Voice commands ("command clear line", "cursor before current word") — built on the
achat `cursor`/`clear` ops reserved in the brief. Optional LLM text correction for
homophones. Server speed-ups (vLLM) to shrink correction latency.
