import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import whisper_dictate
from whisper_dictate import WhisperDictate


class FakeDialog:
    def __init__(self, *args, **kwargs):
        self.handlers = []
        self.destroyed = False
        self.filename = None

    def add_buttons(self, *args): pass
    def set_current_name(self, name): pass
    def set_do_overwrite_confirmation(self, value): pass
    def set_position(self, position): pass
    def set_keep_above(self, value): pass
    def show_all(self): pass
    def present(self): pass

    def run(self):
        raise AssertionError("dialog.run() blocks the main loop")

    def connect(self, signal, handler, *data):
        self.handlers.append((signal, handler, data))

    def respond(self, response):
        for signal, handler, data in self.handlers:
            if signal == "response":
                handler(self, response, *data)

    def get_filename(self):
        return self.filename

    def destroy(self):
        self.destroyed = True


def app():
    a = object.__new__(WhisperDictate)
    a.notices = []
    a.notify = a.notices.append
    a.update_status = lambda status: None
    return a


class SaveDialogTests(unittest.TestCase):
    def show(self, a, src):
        with mock.patch.object(whisper_dictate.Gtk, "FileChooserDialog", FakeDialog):
            a._show_save_transcribe_dialog(src)
        return next(iter(a.transcribe_save_dialogs - getattr(self, "seen", set())))

    def test_every_session_can_save(self):
        a = app()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        self.seen = set()
        for n in range(3):
            src = out / f"src{n}.txt"
            src.write_text(f"session {n}")
            dialog = self.show(a, str(src))
            self.seen.add(dialog)
            dialog.filename = str(out / f"saved{n}.txt")
            dialog.respond(whisper_dictate.Gtk.ResponseType.OK)
            self.assertTrue(dialog.destroyed)
        for n in range(3):
            self.assertEqual((out / f"saved{n}.txt").read_text(), f"session {n}")
        self.assertEqual(a.transcribe_save_dialogs, set())

    def test_quit_closes_open_save_dialogs(self):
        a = app()
        a.stop_remote_monitor = lambda: None
        dialog = self.show(a, "/nonexistent")
        with mock.patch.object(whisper_dictate.Gtk, "main_quit") as main_quit:
            a.quit()
        self.assertTrue(dialog.destroyed)
        main_quit.assert_called_once()


class ChunkSendTests(unittest.TestCase):
    """Unlike live dictation, a transcribe session sends each chunk once."""

    def test_each_chunk_is_transcribed_once(self):
        a = app()
        a.transcribe_chunk_queue = queue.Queue()
        a.transcribe_pending_lock = threading.Lock()
        a.transcribe_tempfile_lock = threading.Lock()
        a.transcribe_pending_chunks = 3
        a.transcribe_completed_chunks = 0
        a.transcribe_failed_chunks = 0
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "chunks.txt"
        a.transcribe_tempfile_path = str(out)
        sent = []

        def transcribe(audio):
            sent.append(float(audio[0]))
            return f"c{int(audio[0])}"
        a._transcribe_session_audio = transcribe
        for i in range(3):
            a.transcribe_chunk_queue.put((i, np.full(10, float(i), dtype=np.float32)))
        a.transcribe_chunk_queue.put(None)
        a._transcribe_session_worker()
        self.assertEqual(sent, [0.0, 1.0, 2.0])
        self.assertEqual(out.read_text(), "c0 c1 c2 ")
        self.assertEqual(a.transcribe_pending_chunks, 0)


if __name__ == "__main__":
    unittest.main()
