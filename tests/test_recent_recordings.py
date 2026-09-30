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
