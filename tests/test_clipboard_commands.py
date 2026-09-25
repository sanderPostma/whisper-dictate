import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_commands import mentions_command, parse_clipboard
from live_controller import LiveController
from live_output import WezTermTarget, XdotoolTarget, choose_target, clipboard_key
from live_segmenter import ChunkReady
from live_session import Edit, LiveSession


class ParseClipboardTests(unittest.TestCase):
    def test_commands(self):
        self.assertEqual(parse_clipboard("Command paste."), "paste")
        self.assertEqual(parse_clipboard("command copy"), "copy")
        self.assertEqual(parse_clipboard("Command, cut!"), "cut")
        self.assertEqual(parse_clipboard("Comment. Paste."), "paste")

    def test_not_commands(self):
        self.assertIsNone(parse_clipboard("paste it here"))
        self.assertIsNone(parse_clipboard("command copy the file"))
        self.assertTrue(mentions_command("Command paste."))


class ClipboardKeyTests(unittest.TestCase):
    def test_other_windows(self):
        self.assertEqual(clipboard_key("paste", terminal=False), "ctrl+v")
        self.assertEqual(clipboard_key("copy", terminal=False), "ctrl+c")
        self.assertEqual(clipboard_key("cut", terminal=False), "ctrl+x")

    def test_terminals_never_get_ctrl_c(self):
        # Ctrl+C interrupts a terminal program; Ctrl+Shift+C/V is the clipboard there.
        self.assertEqual(clipboard_key("paste", terminal=True), "ctrl+shift+v")
        self.assertEqual(clipboard_key("copy", terminal=True), "ctrl+shift+c")
        self.assertIsNone(clipboard_key("cut", terminal=True))

    def test_targets_press_the_key(self):
        run = mock.Mock(return_value=mock.Mock(returncode=0))
        self.assertTrue(WezTermTarget(3, "1", run).clipboard("paste"))
        self.assertEqual(run.call_args[0][0], ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"])
        self.assertTrue(XdotoolTarget("9", run).clipboard("cut"))
        self.assertEqual(run.call_args[0][0][-1], "ctrl+x")
        self.assertFalse(WezTermTarget(3, "1", run).clipboard("cut"))

    def test_xdotool_target_in_another_terminal(self):
        run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="42\n"))
        target = choose_target("gnome-terminal-server.Gnome-terminal", run)
        self.assertTrue(target.terminal)
        self.assertEqual(clipboard_key("copy", target.terminal), "ctrl+shift+c")


class LiveClipboardTests(unittest.TestCase):
    def test_paste_forgets_history_and_types_only_a_space(self):
        actions, edits = [], []

        class Target:
            name = "t"

            def send(self, edit):
                edits.append(edit)
                return True

            def still_focused(self):
                return True

            def clipboard(self, action):
                actions.append(action)
                return True

        results = ["Hello there.", "Command paste."]
        ctl = LiveController(LiveSession(), Target(), lambda audio, prompt: results.pop(0),
                             log=lambda *_: None)
        audio = np.zeros(16000, dtype=np.float32)
        ctl.process(ChunkReady(audio, 0.0, 1.0))
        ctl.process(ChunkReady(audio, 1.0, 2.0))
        self.assertEqual(actions, ["paste"])
        self.assertEqual(edits[1:], [Edit(0, " ")])  # a space, then the paste
        self.assertEqual(ctl.session.history, "")


class OneShotPasteTests(unittest.TestCase):
    def test_space_before_a_paste_after_a_word(self):
        import whisper_dictate
        a = object.__new__(whisper_dictate.WhisperDictate)
        a.config = {}
        a.update_status = lambda status: None
        sent, pressed = [], []
        source = mock.Mock()
        source.line_context = lambda: ("See this", "")
        source.send = lambda edit: sent.append(edit) or True
        source.clipboard = lambda action: pressed.append(action) or True
        with mock.patch("builtins.print"):
            a._oneshot_clipboard(source, "paste")
        self.assertEqual((sent, pressed), ([Edit(0, " ")], ["paste"]))


if __name__ == "__main__":
    unittest.main()
