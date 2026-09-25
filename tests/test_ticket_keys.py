import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_fixes import fix_ticket_keys


def fix(text):
    return fix_ticket_keys(text, ["VDX", "ACH"])


class TicketKeyTests(unittest.TestCase):
    def test_spelled_letters_and_number_words(self):
        self.assertEqual(fix("v D X dash two eight three"), "VDX-283")

    def test_a_misheard_for_eight(self):
        self.assertEqual(fix("v D X dash two A three"), "VDX-283")

    def test_inside_a_sentence(self):
        self.assertEqual(fix("Please look at v D X dash two eight three today."),
                         "Please look at VDX-283 today.")

    def test_compound_numbers(self):
        self.assertEqual(fix("VDX two eighty three"), "VDX-283")
        self.assertEqual(fix("VDX dash two hundred eighty three"), "VDX-283")
        self.assertEqual(fix("vdx twelve"), "VDX-12")
        self.assertEqual(fix("VDX one oh four"), "VDX-104")

    def test_digits_and_separators(self):
        self.assertEqual(fix("VDX 283"), "VDX-283")
        self.assertEqual(fix("V.D.X. - 283"), "VDX-283")
        self.assertEqual(fix("VDX-two-eight-three"), "VDX-283")
        self.assertEqual(fix("VDX 2 8 3."), "VDX-283.")

    def test_already_right(self):
        self.assertEqual(fix("See VDX-283 and ACH-7."), "See VDX-283 and ACH-7.")

    def test_other_keys_in_the_list(self):
        self.assertEqual(fix("a c h dash seven"), "ACH-7")

    def test_leaves_other_text_alone(self):
        for text in ("I want to go to the store.", "The VDX team is great.",
                     "vdx", "a, b, c"):
            self.assertEqual(fix(text), text, text)

    def test_number_stops_at_the_first_other_word(self):
        self.assertEqual(fix("VDX two eight three is done"), "VDX-283 is done")
        self.assertEqual(fix("VDX 283 for review"), "VDX-283 for review")

    def test_no_keys_configured(self):
        self.assertEqual(fix_ticket_keys("v D X dash two", []), "v D X dash two")


if __name__ == "__main__":
    unittest.main()
