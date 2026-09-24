import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_session import Edit, LiveSession, normalise, strip_echo


def audio(seconds):
    return np.zeros(int(16000 * seconds), dtype=np.float32)


def session_with(pieces, **kwargs):
    """pieces: (t0, t1, raw) triples, each fed as a chunk plus its fast result."""
    s = LiveSession(**kwargs)
    for t0, t1, raw in pieces:
        s.add_chunk(audio(t1 - t0), t0, t1)
        s.add_fast_result(t1, raw)
    return s


class NormaliseTests(unittest.TestCase):
    def test_first_text_unchanged(self):
        self.assertEqual(normalise("Hello world", ""), "Hello world")

    def test_continuation_lowercases_and_spaces(self):
        self.assertEqual(normalise("The branch", "merge"), " the branch")

    def test_keeps_pronoun_i_and_acronyms(self):
        self.assertEqual(normalise("I think", "so"), " I think")
        self.assertEqual(normalise("API calls", "the"), " API calls")

    def test_after_sentence_end_keeps_capital(self):
        self.assertEqual(normalise("Next one", "Done."), " Next one")

    def test_after_newline_keeps_capital_and_adds_no_space(self):
        self.assertEqual(normalise("Next", "line\n"), "Next")

    def test_after_trailing_space_adds_no_space(self):
        self.assertEqual(normalise("next", "word "), "next")

    def test_empty_and_punctuation_only_are_dropped(self):
        self.assertEqual(normalise("", "x"), "")
        self.assertEqual(normalise(None, "x"), "")
        self.assertEqual(normalise("  . ", "x"), "")

    def test_collapses_whitespace(self):
        self.assertEqual(normalise("a   b\n c", ""), "a b c")

    def test_strip_echo_removes_repeated_context(self):
        self.assertEqual(
            strip_echo("merge the branch today please", "we should merge the branch"),
            "today please",
        )

    def test_strip_echo_ignores_short_overlap(self):
        self.assertEqual(strip_echo("the branch", "merge the branch"), "the branch")


class FastResultTests(unittest.TestCase):
    def test_fast_result_is_append_edit(self):
        s = LiveSession()
        s.add_chunk(audio(1), 0.0, 1.0)
        self.assertEqual(s.add_fast_result(1.0, "Hello there"), Edit(0, "Hello there"))
        self.assertEqual(s.typed_window, "Hello there")

    def test_empty_fast_result_is_none(self):
        s = LiveSession()
        s.add_chunk(audio(1), 0.0, 1.0)
        self.assertIsNone(s.add_fast_result(1.0, "  "))
        self.assertEqual(s.piece_count, 0)
        self.assertEqual(s.chunk_count, 1)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.s = session_with([
            (0.0, 1.0, "We should merge the"),
            (1.0, 2.0, "brunch"),
            (2.0, 3.0, "today"),
        ])
        self.assertEqual(self.s.typed_window, "We should merge the brunch today")

    def test_correction_rewrites_only_differing_tail(self):
        edit = self.s.apply_correction(3.0, "We should merge the branch today")
        self.assertEqual(edit, Edit(10, "anch today"))
        self.assertEqual(self.s.typed_window, "We should merge the branch today")
        self.assertEqual(self.s.piece_count, 1)

    def test_pieces_after_t_upto_are_retyped(self):
        edit = self.s.apply_correction(2.0, "We should merge the branch")
        self.assertEqual(edit, Edit(10, "anch today"))
        self.assertEqual(self.s.typed_window, "We should merge the branch today")
        self.assertEqual(self.s.piece_count, 2)

    def test_identical_correction_is_none(self):
        self.assertIsNone(self.s.apply_correction(3.0, "We should merge the brunch today"))

    def test_empty_correction_keeps_text(self):
        self.assertIsNone(self.s.apply_correction(3.0, " . "))
        self.assertEqual(self.s.typed_window, "We should merge the brunch today")

    def test_backspace_cap(self):
        s = session_with([(0.0, 1.0, "We should merge the"), (1.0, 2.0, "brunch")], max_backspace=3)
        self.assertIsNone(s.apply_correction(2.0, "We should merge the branch"))
        self.assertEqual(s.typed_window, "We should merge the brunch")

    def test_correction_recovers_missing_fast_result(self):
        s = session_with([(0.0, 1.0, "Hello there")])
        s.add_chunk(audio(1), 1.0, 2.0)  # its fast pass failed: no piece
        self.assertEqual(s.apply_correction(2.0, "Hello there general Kenobi"), Edit(0, " general Kenobi"))

    def test_correction_after_commit_uses_committed_spacing(self):
        s = session_with([(0.0, 1.0, "One.")])
        s.commit()
        s.add_chunk(audio(1), 1.0, 2.0)
        s.add_fast_result(2.0, "Too")
        s.add_chunk(audio(1), 2.0, 3.0)
        s.add_fast_result(3.0, "more")
        self.assertEqual(s.apply_correction(3.0, "Two more"), Edit(7, "wo more"))

    def test_window_audio_covers_all_chunks(self):
        audio_all, t_upto = self.s.window_audio()
        self.assertEqual(len(audio_all), 3 * 16000)
        self.assertEqual(t_upto, 3.0)

    def test_window_audio_empty_is_none(self):
        self.assertIsNone(LiveSession().window_audio())


class CommitTests(unittest.TestCase):
    def test_long_pause_after_sentence_commits(self):
        self.assertTrue(session_with([(0.0, 1.0, "Done.")]).should_commit(long_pause=True))

    def test_long_pause_mid_sentence_does_not_commit(self):
        self.assertFalse(session_with([(0.0, 1.0, "and then")]).should_commit(long_pause=True))

    def test_window_limit_commits(self):
        s = session_with([(0.0, 8.0, "a"), (8.0, 11.0, "b")], window_max_s=12.0)
        self.assertFalse(s.should_commit(incoming_s=1.0))
        self.assertTrue(s.should_commit(incoming_s=2.0))

    def test_focus_loss_commits(self):
        self.assertTrue(session_with([(0.0, 1.0, "a")]).should_commit(focus_ok=False))

    def test_empty_window_never_commits(self):
        self.assertFalse(LiveSession().should_commit(long_pause=True, focus_ok=False))

    def test_commit_moves_text_and_clears_window(self):
        s = session_with([(0.0, 1.0, "Hello there")], context_chars=5)
        s.commit()
        self.assertEqual(s.committed_text, "there")
        self.assertEqual(s.typed_window, "")
        self.assertEqual(s.chunk_count, 0)
        s.add_chunk(audio(1), 1.0, 2.0)
        self.assertEqual(s.add_fast_result(2.0, "General"), Edit(0, " general"))


class PromptTests(unittest.TestCase):
    def test_fast_prompt_has_base_committed_and_window(self):
        s = session_with([(0.0, 1.0, "One.")], base_prompt="Vocabulary: git")
        s.commit()
        s.add_chunk(audio(1), 1.0, 2.0)
        s.add_fast_result(2.0, "Two")
        self.assertEqual(s.fast_prompt(), "Vocabulary: git\nOne. Two")
        self.assertEqual(s.correction_prompt(), "Vocabulary: git\nOne.")

    def test_prompt_tail_is_trimmed(self):
        s = session_with([(0.0, 1.0, "abcdefgh")], context_chars=4)
        self.assertEqual(s.fast_prompt(), "efgh")

    def test_no_base_no_text_is_empty(self):
        self.assertEqual(LiveSession().fast_prompt(), "")


if __name__ == "__main__":
    unittest.main()
