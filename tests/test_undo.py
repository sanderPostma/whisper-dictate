import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import Command, parse_undo
from live_session import Edit, LiveSession


def audio(seconds):
    return np.zeros(int(16000 * seconds), dtype=np.float32)


def typed(session, t0, t1, raw):
    session.add_chunk(audio(t1 - t0), t0, t1)
    return session.add_fast_result(t1, raw)


class ParseUndoTests(unittest.TestCase):
    def test_accepted_forms(self):
        for text in ("Command undo.", "command, undo", "Command and do.", "Undo that.", "undo that"):
            self.assertTrue(parse_undo(text), text)

    def test_not_a_command(self):
        for text in ("Undo.", "I will undo that change", "undo the last commit", "command"):
            self.assertFalse(parse_undo(text), text)


class SessionUndoTests(unittest.TestCase):
    def test_undo_removes_the_last_utterance(self):
        s = LiveSession()
        typed(s, 0, 1, "Fix the login.")
        typed(s, 1, 2, "Then deploy it too.")
        self.assertEqual(s.undo(), Edit(20, ""))  # " Then deploy it too."
        self.assertEqual(s.history, "Fix the login.")
        self.assertEqual(s.chunk_count, 0)

    def test_undo_twice_goes_further_back(self):
        s = LiveSession()
        typed(s, 0, 1, "One.")
        s.commit()
        typed(s, 1, 2, "Two.")
        s.undo()
        self.assertEqual(s.undo(), Edit(4, ""))
        self.assertIsNone(s.undo())

    def test_undo_a_repair(self):
        s = LiveSession()
        typed(s, 0, 1, "Done.")
        typed(s, 1, 2, "I want to go home.")
        s.apply_command(Command(scratch=True))
        self.assertEqual(s.undo(), Edit(0, " I want to go home."))
        self.assertEqual(s.history, "Done. I want to go home.")

    def test_a_correction_merges_its_window_into_one_step(self):
        s = LiveSession()
        typed(s, 0, 1, "Done.")
        s.commit()
        typed(s, 1, 2, "merge the")
        typed(s, 2, 3, "brunch")
        s.apply_correction(3, "merge the branch")
        self.assertEqual(s.undo(), Edit(17, ""))  # " merge the branch" as one step
        self.assertEqual(s.history, "Done.")

    def test_forgetting_history_forgets_undo(self):
        s = LiveSession()
        typed(s, 0, 1, "Done.")
        s.forget_history()
        self.assertIsNone(s.undo())

    def test_trimming_to_the_window_keeps_undo_inside_it(self):
        s = LiveSession()
        typed(s, 0, 1, "Old text.")
        s.commit()
        typed(s, 1, 2, "New words")
        s.trim_history(len(" New words"))
        self.assertEqual(s.undo(), Edit(10, ""))
        self.assertIsNone(s.undo())  # the older step is outside what we may touch


if __name__ == "__main__":
    unittest.main()
