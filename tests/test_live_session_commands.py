import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import Command
from live_session import Edit, LiveSession


def audio(seconds):
    return np.zeros(int(16000 * seconds), dtype=np.float32)


def typed(session, t0, t1, raw):
    session.add_chunk(audio(t1 - t0), t0, t1)
    return session.add_fast_result(t1, raw)


class HistoryTests(unittest.TestCase):
    def test_history_spans_commits(self):
        s = LiveSession()
        typed(s, 0, 1, "I want to.")
        s.commit()
        typed(s, 5, 6, "Go home.")
        self.assertEqual(s.history, "I want to. Go home.")

    def test_corrections_rewrite_history(self):
        s = LiveSession()
        typed(s, 0, 1, "merge the")
        typed(s, 1, 2, "brunch")
        s.apply_correction(2, "merge the branch")
        self.assertEqual(s.history, "merge the branch")

    def test_forget(self):
        s = LiveSession()
        typed(s, 0, 1, "Hello.")
        s.forget_history()
        self.assertEqual(s.history, "")


class JoinTests(unittest.TestCase):
    def test_period_joins_across_a_long_pause(self):
        s = LiveSession()
        typed(s, 0, 1, "I want to.")
        s.commit()  # a thinking pause committed the window
        edit = s.apply_command(Command(rest="test more", end="."))
        self.assertEqual(edit, Edit(1, " test more."))
        self.assertEqual(s.history, "I want to test more.")
        self.assertEqual(s.chunk_count, 0)

    def test_question_mark(self):
        s = LiveSession()
        typed(s, 0, 1, "Is it.")
        self.assertEqual(s.apply_command(Command(rest="Done", end="?")), Edit(1, " done?"))

    def test_joined_words_lose_the_models_own_end(self):
        s = LiveSession()
        typed(s, 0, 1, "I want to.")
        self.assertEqual(s.apply_command(Command(rest="Test more.", end=".")), Edit(1, " test more."))

    def test_period_alone_just_ends_the_sentence(self):
        s = LiveSession()
        typed(s, 0, 1, "I want to")
        self.assertEqual(s.apply_command(Command(end=".")), Edit(0, "."))

    def test_join_without_history_only_types(self):
        s = LiveSession()
        s.set_context("Somebody else typed this.")
        self.assertEqual(s.apply_command(Command(rest="Test more", end=".")), Edit(0, " Test more."))


class CommaTests(unittest.TestCase):
    def test_comma_replaces_the_wrong_sentence_end(self):
        s = LiveSession()
        typed(s, 0, 1, "I want to.")
        s.commit()
        edit = s.apply_command(Command(rest="test more.", comma=True))
        self.assertEqual(edit, Edit(1, ", test more."))
        self.assertEqual(s.history, "I want to, test more.")

    def test_comma_with_period(self):
        s = LiveSession()
        typed(s, 0, 1, "I want to.")
        edit = s.apply_command(Command(rest="Test more", comma=True, end="."))
        self.assertEqual(edit, Edit(1, ", test more."))

    def test_comma_alone(self):
        s = LiveSession()
        typed(s, 0, 1, "I want to.")
        self.assertEqual(s.apply_command(Command(comma=True)), Edit(1, ","))

    def test_comma_after_text_without_end(self):
        s = LiveSession()
        typed(s, 0, 1, "I want to")
        self.assertEqual(s.apply_command(Command(rest="go", comma=True)), Edit(0, ", go"))


class ScratchTests(unittest.TestCase):
    def test_scratch_the_sentence_in_progress(self):
        s = LiveSession()
        typed(s, 0, 1, "Done.")
        typed(s, 1, 2, "I want to go")
        s.commit()
        self.assertEqual(s.apply_command(Command(scratch=True)), Edit(13, ""))
        self.assertEqual(s.history, "Done.")

    def test_scratch_a_just_completed_sentence(self):
        s = LiveSession()
        typed(s, 0, 1, "Done.")
        typed(s, 1, 2, "I want to go home.")
        self.assertEqual(s.apply_command(Command(scratch=True)), Edit(19, ""))
        self.assertEqual(s.history, "Done.")

    def test_scratch_only_our_own_text(self):
        s = LiveSession()
        s.set_context("The operator wrote this and")
        typed(s, 0, 1, "then I said")
        self.assertEqual(s.apply_command(Command(scratch=True)), Edit(12, ""))  # " then I said"

    def test_nothing_to_scratch(self):
        self.assertIsNone(LiveSession().apply_command(Command(scratch=True)))

    def test_text_after_scratch_starts_a_new_sentence(self):
        s = LiveSession()
        typed(s, 0, 1, "Done.")
        typed(s, 1, 2, "Oops")
        s.apply_command(Command(scratch=True))
        self.assertEqual(typed(s, 3, 4, "Next one"), Edit(0, " Next one"))


class ReviewFixTests(unittest.TestCase):
    def test_scratch_with_untyped_words_deletes_nothing(self):
        s = LiveSession()
        typed(s, 0, 1, "Ship the release today.")
        self.assertIsNone(s.apply_command(Command(rest="Actually wait for QA", scratch=True)))
        self.assertEqual(s.history, "Ship the release today.")

    def test_lone_period_without_history_types_nothing(self):
        s = LiveSession()
        s.set_context("the operator's text")
        self.assertIsNone(s.apply_command(Command(end=".")))

    def test_every_command_commits_the_window(self):
        s = LiveSession()
        typed(s, 0, 1, "Hello there.")
        s.add_chunk(audio(1), 1, 2)  # the command chunk's own audio
        s.apply_command(Command(end="."))  # nothing to change
        self.assertEqual(s.chunk_count, 0)

    def test_mid_line_period_alone(self):
        s = LiveSession()
        s.set_context("The ", "jumps.")
        typed(s, 0, 1, "brown fox.")
        self.assertEqual(s.history, "brown fox ")
        self.assertEqual(s.apply_command(Command(end=".")), Edit(1, ". "))

    def test_mid_line_comma_alone(self):
        s = LiveSession()
        s.set_context("The ", "jumps.")
        typed(s, 0, 1, "brown fox")
        self.assertEqual(s.apply_command(Command(comma=True)), Edit(1, ", "))


class LimitTests(unittest.TestCase):
    def test_command_backspace_cap(self):
        s = LiveSession(command_max_backspace=10)
        typed(s, 0, 1, "A sentence that is quite long")
        self.assertIsNone(s.apply_command(Command(scratch=True)))
        self.assertEqual(s.history, "A sentence that is quite long")


if __name__ == "__main__":
    unittest.main()
