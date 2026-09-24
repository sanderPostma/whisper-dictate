import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import Command, mentions_command, parse_command


class ParseCommandTests(unittest.TestCase):
    def test_plain_text_is_no_command(self):
        self.assertIsNone(parse_command("I want to test more."))

    def test_period_at_the_end_needs_a_pause_mark(self):
        self.assertEqual(parse_command("test more, period"), Command(rest="test more", end="."))
        self.assertIsNone(parse_command("test more period"))

    def test_model_punctuation_on_the_command_word(self):
        self.assertEqual(parse_command("Test more, period."), Command(rest="Test more", end="."))
        self.assertEqual(parse_command("Test more. Period."), Command(rest="Test more", end="."))

    def test_other_end_words(self):
        self.assertEqual(parse_command("is it done question mark").end, "?")
        self.assertEqual(parse_command("wow exclamation mark").end, "!")
        self.assertEqual(parse_command("wow exclamation point").end, "!")
        self.assertEqual(parse_command("done. Full stop.").end, ".")

    def test_period_alone(self):
        self.assertEqual(parse_command("Period."), Command(rest="", end="."))

    def test_period_inside_a_sentence_is_text(self):
        self.assertIsNone(parse_command("the period ends today"))

    def test_comma_at_the_start(self):
        self.assertEqual(parse_command("Comma, test more."), Command(rest="test more.", comma=True))
        self.assertIsNone(parse_command("comma test more"))

    def test_model_wrote_the_comma_itself(self):
        self.assertEqual(parse_command(", test more"), Command(rest="test more", comma=True))

    def test_comma_start_and_period_end(self):
        self.assertEqual(parse_command("Comma, test more. Period."),
                         Command(rest="test more", comma=True, end="."))

    def test_comma_inside_is_text(self):
        self.assertIsNone(parse_command("add a comma here"))

    def test_scratch_that(self):
        self.assertEqual(parse_command("Scratch that."), Command(scratch=True))
        self.assertEqual(parse_command("I want to go home, scratch that."),
                         Command(rest="I want to go home", scratch=True))

    def test_ordinary_words_are_not_commands(self):
        # Seen as false positives in review.
        for text in ("We extended the trial period.", "He came to a full stop.", "The period.",
                     "Comma separated values are fine.", "the grace period"):
            self.assertIsNone(parse_command(text), text)

    def test_comma_alone(self):
        self.assertEqual(parse_command("Comma."), Command(comma=True))

    def test_command_words_anywhere_in_a_correction(self):
        self.assertTrue(mentions_command("Hello there. Period. And more"))
        self.assertTrue(mentions_command("I want to, comma, test"))
        self.assertTrue(mentions_command("Done. Scratch that. Next"))
        self.assertFalse(mentions_command("We extended the trial period and the comma rule"))

    def test_filler_before_scratch_that_counts_as_nothing(self):
        for text in ("No, scratch that.", "Okay, scratch that.", "Oh no, scratch that", "Um, scratch that."):
            self.assertEqual(parse_command(text), Command(scratch=True), text)

    def test_press_enter_at_the_end(self):
        self.assertEqual(parse_command("/compact, press enter."), Command(rest="/compact", enter=True))
        self.assertEqual(parse_command("slash compact press enter"),
                         Command(rest="slash compact", enter=True))
        self.assertEqual(parse_command("Run it. Press return."), Command(rest="Run it.", enter=True))

    def test_enter_alone(self):
        self.assertEqual(parse_command("Enter."), Command(enter=True))
        self.assertEqual(parse_command("Press enter."), Command(enter=True))

    def test_enter_as_a_word_is_text(self):
        for text in ("Enter is not working.", "Now enter the room", "the center", "enter"[:0] + "we enter"):
            self.assertIsNone(parse_command(text), text)

    def test_period_then_press_enter(self):
        self.assertEqual(parse_command("Test more. Period. Press enter."),
                         Command(rest="Test more", end=".", enter=True))

    def test_scratch_that_mid_chunk_is_text(self):
        self.assertIsNone(parse_command("scratch that itch"))


if __name__ == "__main__":
    unittest.main()
