import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teams_call_watcher import is_source_output_added, teams_is_capturing

TEAMS_LISTING = """Source Output #77
\tDriver: protocol-native
\tapplication.name = "Microsoft Teams"
\tapplication.process.binary = "teams-for-linux"
Source Output #78
\tapplication.name = "PipeWire ALSA [python3.12]"
\tapplication.process.binary = "python3.12"
"""

DICTATION_LISTING = """Source Output #3056
\tapplication.name = "PipeWire ALSA [python3.12]"
\tapplication.process.binary = "python3.12"
"""


class IsSourceOutputAddedTests(unittest.TestCase):
    def test_new_source_output_matches(self):
        self.assertTrue(is_source_output_added("Event 'new' on source-output #3056"))

    def test_removed_source_output_does_not_match(self):
        self.assertFalse(is_source_output_added("Event 'remove' on source-output #3056"))

    def test_other_events_do_not_match(self):
        self.assertFalse(is_source_output_added("Event 'new' on sink-input #12"))
        self.assertFalse(is_source_output_added("Event 'change' on source #4"))


class TeamsIsCapturingTests(unittest.TestCase):
    def test_teams_application_name_detected(self):
        self.assertTrue(teams_is_capturing(TEAMS_LISTING))

    def test_teams_binary_detected(self):
        self.assertTrue(teams_is_capturing(
            '\tapplication.process.binary = "teams-for-linux"\n'))

    def test_dictation_capture_is_not_teams(self):
        self.assertFalse(teams_is_capturing(DICTATION_LISTING))

    def test_empty_listing(self):
        self.assertFalse(teams_is_capturing(""))

    def test_teams_in_other_property_is_ignored(self):
        self.assertFalse(teams_is_capturing('\tmedia.name = "Teams meeting notes"\n'))


if __name__ == "__main__":
    unittest.main()
