import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recent_recordings import RecentRecordings


def audio(seconds, rate=16000):
    return np.zeros(int(seconds * rate), dtype=np.float32)


class RecentRecordingsTests(unittest.TestCase):
    def test_keeps_only_the_newest_five_newest_first(self):
        recent = RecentRecordings(keep=5)
        for i in range(7):
            recent.add(audio(i + 1), 16000, when=1000 + i)
        self.assertEqual([r.when for r in recent.newest_first()], [1006, 1005, 1004, 1003, 1002])

    def test_label_shows_duration_and_transcript_state(self):
        r = RecentRecordings().add(audio(2.5), 16000, when=0)
        self.assertIn("2.5s", r.label())
        self.assertIn("(transcribing...)", r.label())
        r.text = ""
        self.assertIn("(nothing heard)", r.label())
        r.text = "push the\nfix"
        self.assertIn("“push the fix”", r.label())

    def test_long_transcript_is_shortened(self):
        r = RecentRecordings().add(audio(1), 16000, when=0)
        r.text = "word " * 30
        label = r.label()
        self.assertTrue(label.endswith("…”"))
        self.assertLess(len(label), 70)


if __name__ == "__main__":
    unittest.main()


class LiveChunkTests(unittest.TestCase):
    def app(self):
        from unittest import mock
        import whisper_dictate as wd
        a = object.__new__(wd.WhisperDictate)
        a.recent_recordings = RecentRecordings()
        a.recent_menu = None
        self.idle = mock.patch.object(wd.GLib, "idle_add").start()
        self.addCleanup(mock.patch.stopall)
        return a

    def test_live_chunks_are_kept_with_their_transcript(self):
        a = self.app()
        fast = a._keep_live_chunks(lambda audio, prompt: "hello there", 16000)
        self.assertEqual(fast(audio(1.5), "p"), "hello there")
        r = a.recent_recordings.newest_first()[0]
        self.assertTrue(r.live)
        self.assertEqual(r.text, "hello there")
        self.assertTrue(r.label().startswith("⚡ "))

    def test_failed_live_chunk_is_kept_and_error_still_raised(self):
        a = self.app()

        def boom(audio, prompt):
            raise RuntimeError("down")
        with self.assertRaises(RuntimeError):
            a._keep_live_chunks(boom, 16000)(audio(1), "p")
        self.assertTrue(a.recent_recordings.newest_first()[0].failed)

    def test_playback_window_mutes_then_ends(self):
        from unittest import mock
        import whisper_dictate as wd
        a = self.app()
        a.recording = False
        r = a.recent_recordings.add(audio(2), 16000)
        with mock.patch.object(wd.sd, "play"), mock.patch.object(wd.sd, "stop"):
            a.play_recording(r)
            self.assertTrue(a.playback_active())
            a.stop_playback()
            self.assertFalse(a.playback_active())


class PlaybackLeadTests(unittest.TestCase):
    def test_playback_starts_with_silence_then_the_whole_recording(self):
        from unittest import mock
        import whisper_dictate as wd
        a = object.__new__(wd.WhisperDictate)
        a.recording = False
        r = RecentRecordings().add(np.ones(16000, dtype=np.float32), 16000)
        with mock.patch.object(wd.sd, "play") as play:
            a.play_recording(r)
        played = play.call_args.args[0]
        lead = int(16000 * wd.PLAYBACK_LEAD_S)
        self.assertFalse(played[:lead].any())
        self.assertEqual(played[lead:].tolist(), r.audio.tolist())


class PlaybackIsWhatTheModelGotTests(unittest.TestCase):
    def test_live_chunk_keeps_the_audio_sent_not_the_recording(self):
        from unittest import mock
        import whisper_dictate as wd
        a = object.__new__(wd.WhisperDictate)
        a.recent_recordings = RecentRecordings()
        a.recent_menu = None
        sent = np.full(24000, 0.5, dtype=np.float32)

        def fast(audio, prompt):
            wd.note_sent(sent, 24000)
            return "hi"
        with mock.patch.object(wd.GLib, "idle_add"):
            a._keep_live_chunks(fast, 48000)(audio(1, rate=48000), "p")
        r = a.recent_recordings.newest_first()[0]
        self.assertEqual(r.sample_rate, 24000)
        self.assertEqual(r.audio.tolist(), sent.tolist())
        self.assertIn("24k", r.label())

    def test_nothing_sent_keeps_the_recording(self):
        from unittest import mock
        import whisper_dictate as wd
        a = object.__new__(wd.WhisperDictate)
        a.recent_recordings = RecentRecordings()
        a.recent_menu = None
        wd.note_sent(np.zeros(5, dtype=np.float32), 16000)  # stale value from an earlier call

        def fast(audio, prompt):
            raise RuntimeError("down")
        with mock.patch.object(wd.GLib, "idle_add"), self.assertRaises(RuntimeError):
            a._keep_live_chunks(fast, 48000)(audio(1, rate=48000), "p")
        self.assertEqual(a.recent_recordings.newest_first()[0].sample_rate, 48000)
