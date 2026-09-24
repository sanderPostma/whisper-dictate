"""Drive a live dictation session on one worker thread.

Each ChunkReady gets a fast pass (the new chunk only, typed immediately).
When no further event is waiting, the whole window is re-transcribed and the
typed tail is rewritten where it differs. All edits are applied in order on
this one thread, so the session never sees a correction go stale.
"""

import queue
import threading

from live_segmenter import ChunkReady, LongPause

_STOP = object()


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
                self.session.commit()

    def _handle_chunk(self, chunk):
        if not self.target.still_focused():
            self._retarget()
        if self.session.should_commit(incoming_s=chunk.t1 - chunk.t0):
            self.session.commit()
        self.session.add_chunk(chunk.audio, chunk.t0, chunk.t1)
        try:
            raw = self.fast_transcribe(chunk.audio, self.session.fast_prompt())
        except Exception as e:
            self._report(f"Live transcription failed: {e}")
            return
        edit = self.session.add_fast_result(chunk.t1, self.postprocess(raw or ""))
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
        try:
            raw = self.correct_transcribe(audio, s.correction_prompt())
        except Exception as e:
            self.log(f"[live] correction skipped: {e}")
            return
        edit = s.apply_correction(t_upto, self.postprocess(raw or ""))
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
        if self.make_target is not None:
            new = self.make_target()
            if new is not None:
                self.target = new

    def _send(self, edit):
        try:
            if self.target.send(edit):
                return True
        except Exception as e:
            self.log(f"[live] target.send raised: {e}")
        self.log("[live] output failed; committing window and disabling corrections")
        self.session.commit()
        self.corrections_enabled = False
        return False

    def _report(self, message):
        self.log(f"[live] {message}")
        if not self._error_reported and self.on_error:
            self._error_reported = True
            self.on_error(message)
