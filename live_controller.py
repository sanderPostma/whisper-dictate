"""Drive a live dictation session on one worker thread.

Each ChunkReady gets a fast pass (the new chunk only, typed immediately).
When no further event is waiting, the whole window is re-transcribed and the
typed tail is rewritten where it differs. All edits are applied in order on
this one thread, so the session never sees a correction go stale.
"""

import queue
import threading
import time

import numpy as np

from live_commands import mentions_command, parse_command
from live_segmenter import ChunkReady, LongPause

_STOP = object()


def _show(edit):
    return "no edit" if edit is None else f"-{edit.backspace} +{edit.insert!r}"


def select_transcribers(remote_enabled, remote_is_down, remote, local, local_fallback,
                        on_remote_result):
    """Pick the (fast, correct) transcribers for a live session.

    remote/local/local_fallback: (audio, prompt) -> str, may raise.
    remote_is_down: () -> bool, the app's current view of the server.
    on_remote_result: (ok: bool, reason: str | None) -> None.
    correct is None when corrections are impossible for the whole session.
    """
    if not remote_enabled:
        return local, None

    def fast(audio, prompt):
        if remote_is_down():
            return local_fallback(audio, prompt)
        try:
            text = remote(audio, prompt)
        except Exception as e:
            on_remote_result(False, str(e))
            return local_fallback(audio, prompt)
        on_remote_result(True, None)
        return text

    def correct(audio, prompt):
        if remote_is_down():
            raise RuntimeError("remote server unavailable")
        return remote(audio, prompt)

    return fast, correct


class LiveController:
    def __init__(self, session, target, fast_transcribe, correct_transcribe=None,
                 make_target=None, postprocess=None, on_error=None, on_done=None, log=print):
        self.session = session
        self.target = target
        self.fast_transcribe = fast_transcribe
        self.correct_transcribe = correct_transcribe
        self.make_target = make_target
        self.postprocess = postprocess or (lambda text: text)
        self.on_error = on_error
        self.on_done = on_done
        self.log = log
        self.corrections_enabled = correct_transcribe is not None
        self._jobs = queue.Queue()
        self._thread = None
        self._error_reported = False

    def start(self):
        self._thread = threading.Thread(target=self.run_until_stopped, daemon=True)
        self._thread.start()

    def submit(self, event):
        self._jobs.put(event)

    def stop(self):
        self._jobs.put(_STOP)

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def run_until_stopped(self):
        while True:
            event = self._jobs.get()
            if event is _STOP:
                self.finish()
                return
            try:
                self.process(event)
                if isinstance(event, ChunkReady) and self._jobs.empty():
                    self.correct()
            except Exception as e:  # never let one bad event kill the session
                self.log(f"[live] worker error: {e}")

    def process(self, event):
        if isinstance(event, ChunkReady):
            self._handle_chunk(event)
        elif isinstance(event, LongPause):
            if self.session.should_commit(long_pause=True):
                self.log("[live] commit: long pause after sentence end")
                self.session.commit()

    def _handle_chunk(self, chunk):
        rms = float(np.sqrt(np.mean(chunk.audio ** 2))) if len(chunk.audio) else 0.0
        self.log(f"[live] chunk {chunk.t0:.2f}-{chunk.t1:.2f}s ({chunk.t1 - chunk.t0:.2f}s, rms {rms:.4f})")
        if not self.target.still_focused():
            self.log("[live] commit: focus moved")
            self._retarget()
        if self.session.should_commit(incoming_s=chunk.t1 - chunk.t0):
            self.log("[live] commit: window full")
            self.session.commit()
        if self.session.chunk_count == 0:
            self._sync_line_context()
        self.session.add_chunk(chunk.audio, chunk.t0, chunk.t1)
        started = time.monotonic()
        try:
            raw = self.fast_transcribe(chunk.audio, self.session.fast_prompt())
        except Exception as e:
            self._report(f"Live transcription failed: {e}")
            return
        text = self.postprocess(raw or "")
        command = parse_command(text)
        if command is not None:
            self._run_command(command, raw)
            return
        edit = self.session.add_fast_result(chunk.t1, text)
        self.log(f"[live] fast {time.monotonic() - started:.2f}s raw={raw!r} -> {_show(edit)}")
        if edit is not None:
            self._send(edit)

    def correct(self):
        """Re-transcribe the window and rewrite the typed tail where it differs."""
        if not self.corrections_enabled:
            return
        s = self.session
        if s.chunk_count == 0 or (s.chunk_count == 1 and s.piece_count == 1):
            return  # nothing to gain: same audio as the fast pass
        audio, t_upto = s.window_audio()
        if not self.target.still_focused():
            self._retarget()
            return
        started = time.monotonic()
        try:
            raw = self.correct_transcribe(audio, s.correction_prompt())
        except Exception as e:
            self.log(f"[live] correction skipped: {e}")
            return
        text = self.postprocess(raw or "")
        if mentions_command(text):
            # A command the fast pass did not hear: never type its words, and
            # close the window so no later correction brings them back.
            self.log(f"[live] correction skipped: it holds a spoken command ({raw!r})")
            s.commit()
            return
        before = s.typed_window
        edit = s.apply_correction(t_upto, text)
        self.log(f"[live] correct {time.monotonic() - started:.2f}s over {s.chunk_count} chunks "
                 f"raw={raw!r} window={before!r} -> {_show(edit)}")
        if edit is not None:
            self._send(edit)

    def finish(self):
        try:
            try:
                self.correct()
            except Exception as e:
                self.log(f"[live] final correction failed: {e}")
            try:
                self.session.commit()
            except Exception as e:
                self.log(f"[live] final commit failed: {e}")
        finally:
            if self.on_done:
                self.on_done()

    def _retarget(self):
        """Focus moved: freeze the window and follow the new focus, append-only."""
        self.session.commit()
        self.session.forget_history()
        # The old line's text after the cursor means nothing in the new place;
        # a target that knows its line sets it again at the next window.
        self.session.after_text = ""
        if self.make_target is not None:
            new = self.make_target()
            if new is not None:
                self.target = new

    def _run_command(self, command, raw):
        """A spoken repair: one edit over what this session typed."""
        self._verify_history()
        edit = self.session.apply_command(command)
        self.log(f"[live] command {command} raw={raw!r} -> {_show(edit)}")
        if edit is not None:
            self._send(edit)

    def _verify_history(self):
        """Keep only history a command may still rewrite.

        Exact line (achat): the history must still be exactly what precedes
        the cursor. Screen-read line (WezTerm): the cursor row must still be
        part of the history's last line. No line at all (xdotool): only what
        was typed in the current window, since the operator may have typed
        anywhere during a pause.
        """
        s = self.session
        line_context = getattr(self.target, "line_context", None)
        if line_context is None:
            window = s.typed_window
            s.history = s.history[-len(window):] if window else ""
            return
        exact = getattr(self.target, "exact_line", False)
        try:
            # Read without adopting the line's rev: the window may still be open.
            context = line_context(adopt_rev=False) if exact else line_context()
        except Exception as e:
            self.log(f"[live] could not read the target line: {e}")
            context = None
        before = context[0] if context else ""
        if exact:
            ok = context is not None and before.endswith(s.history)
        else:
            last = s.history.split("\n")[-1]
            ok = bool(before) and bool(last) and (last.endswith(before) or before.endswith(last))
        if not ok:
            if s.history:
                self.log("[live] line changed since we typed; commands will not edit it")
            s.forget_history()

    def _sync_line_context(self):
        """At the start of a window, take context from the target's real line."""
        line_context = getattr(self.target, "line_context", None)
        if line_context is None:
            return
        try:
            context = line_context()
        except Exception as e:
            self.log(f"[live] could not read the target line: {e}")
            return
        if context is not None:
            if getattr(self.target, "exact_line", False) and not context[0].endswith(self.session.history):
                self.session.forget_history()
            self.session.set_context(*context)
            self.log(f"[live] line context before={context[0][-60:]!r} after={context[1][:30]!r}")

    def _send(self, edit):
        try:
            if self.target.send(edit):
                return True
        except Exception as e:
            self.log(f"[live] target.send raised: {e}")
        self.session.commit()
        self.session.forget_history()
        if getattr(self.target, "recoverable_rejections", False):
            self.log("[live] edit rejected; committing window")
            if getattr(self.target, "last_send_dropped", False):
                self._report("Live dictation: some words could not be typed "
                             "into the prompt line.")
        else:
            self.log("[live] output failed; committing window and disabling corrections")
            self.corrections_enabled = False
        return False

    def _report(self, message):
        self.log(f"[live] {message}")
        if not self._error_reported and self.on_error:
            self._error_reported = True
            self.on_error(message)
