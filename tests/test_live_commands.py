import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import Command, parse_command


class ParseCommandTests(unittest.TestCase):
    def test_plain_text_is_no_command(self):
        self.assertIsNone(parse_command("I want to test more."))

    def test_period_at_the_end(self):
        self.assertEqual(parse_command("test more period"), Command(rest="test more", end="."))

    def test_model_punctuation_on_the_command_word(self):
        self.assertEqual(parse_command("Test more, period."), Command(rest="Test more", end="."))
        self.assertEqual(parse_command("Test more. Period."), Command(rest="Test more", end="."))

    def test_other_end_words(self):
        self.assertEqual(parse_command("is it done question mark").end, "?")
        self.assertEqual(parse_command("wow exclamation mark").end, "!")
        self.assertEqual(parse_command("wow exclamation point").end, "!")
        self.assertEqual(parse_command("done full stop").end, ".")

    def test_period_alone(self):
        self.assertEqual(parse_command("Period."), Command(rest="", end="."))

    def test_period_inside_a_sentence_is_text(self):
        self.assertIsNone(parse_command("the period ends today"))

    def test_comma_at_the_start(self):
        self.assertEqual(parse_command("Comma, test more."), Command(rest="test more.", comma=True))
        self.assertEqual(parse_command("comma test more"), Command(rest="test more", comma=True))

    def test_model_wrote_the_comma_itself(self):
        self.assertEqual(parse_command(", test more"), Command(rest="test more", comma=True))

    def test_comma_start_and_period_end(self):
        self.assertEqual(parse_command("comma test more period"),
                         Command(rest="test more", comma=True, end="."))

    def test_comma_inside_is_text(self):
        self.assertIsNone(parse_command("add a comma here"))

    def test_scratch_that(self):
        self.assertEqual(parse_command("Scratch that."), Command(scratch=True))
        self.assertEqual(parse_command("I want to go home, scratch that."), Command(scratch=True))

    def test_scratch_that_mid_chunk_is_text(self):
        self.assertIsNone(parse_command("scratch that itch"))


if __name__ == "__main__":
    unittest.main()
