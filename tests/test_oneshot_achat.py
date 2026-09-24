import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import whisper_dictate
from whisper_dictate import WhisperDictate


class FakeLine:
    def __init__(self, context):
        self.context = context

    def line_context(self):
        return self.context


class FakeAchatSource(whisper_dictate.AchatTarget):
    def __init__(self):
        pass


def app(**config):
    a = object.__new__(WhisperDictate)
    a.config = {"output_mode": "type", **config}
    a.get_asr_context = lambda: "Vocabulary: kubectl"
    a.update_status = lambda status: None
    a.get_focused_window_class = lambda: "org.wezfurlong.wezterm"
    return a


class OneShotPromptTests(unittest.TestCase):
    def test_text_before_cursor_joins_the_asr_context(self):
        prompt = app()._oneshot_prompt(FakeLine(("This is", " rest")))
        self.assertEqual(prompt, "Vocabulary: kubectl\nThis is")

    def test_no_achat_session_keeps_asr_context(self):
        self.assertEqual(app()._oneshot_prompt(None), "Vocabulary: kubectl")

    def test_unknown_line_keeps_asr_context(self):
        self.assertEqual(app()._oneshot_prompt(FakeLine(None)), "Vocabulary: kubectl")


class OneShotOutputTests(unittest.TestCase):
    def test_achat_line_is_used_instead_of_keystrokes(self):
        with mock.patch.object(whisper_dictate, "type_into_line", return_value=True) as typed, \
                mock.patch.object(whisper_dictate.subprocess, "run") as run:
            app().output_text("The end", line_source=FakeAchatSource())
        typed.assert_called_once()
        run.assert_not_called()

    def test_falls_back_to_keystrokes_when_line_unusable(self):
        with mock.patch.object(whisper_dictate, "type_into_line", return_value=False), \
                mock.patch.object(whisper_dictate.subprocess, "run") as run, \
                mock.patch.object(whisper_dictate.time, "sleep"):
            app().output_text("The end", line_source=FakeAchatSource())
        self.assertEqual(run.call_args[0][0][:2], ["xdotool", "type"])

    def test_single_word_lowercasing_left_to_the_line(self):
        a = app()
        a.load_replacements = lambda: {"zzqx": "q"}
        self.assertEqual(a.apply_replacements("API", lower_single_word=False), "API")
        self.assertEqual(a.apply_replacements("API"), "api")


class ScreenContextTests(unittest.TestCase):
    """A WezTerm pane without achat: case and spacing from the screen, typed with keys."""

    def typed(self, text, context):
        source = whisper_dictate.WezTermTarget(7, "42")
        source.line_context = lambda: context
        with mock.patch.object(whisper_dictate.subprocess, "run") as run, \
                mock.patch.object(whisper_dictate.time, "sleep"), mock.patch("builtins.print"):
            app().output_text(text, line_source=source)
        return run.call_args[0][0][-1] if run.called else None

    def test_continuing_a_sentence(self):
        self.assertEqual(self.typed("We are not going", ("ok", "")), " we are not going")

    def test_screen_after_text_is_ignored(self):
        # The TUI's hardware cursor can sit off the input row; text after it
        # on screen is not reliably what follows the operator's cursor.
        self.assertEqual(self.typed("Decking", ("", "restarted, check the")), "Decking")

    def test_empty_prompt_keeps_the_capital(self):
        self.assertEqual(self.typed("We are not going", ("", "")), "We are not going")

    def test_unreadable_screen_types_as_is(self):
        self.assertEqual(self.typed("We are", None), "We are")


class OneShotEnterTests(unittest.TestCase):
    def test_enter_pressed_on_the_line_source_after_typing(self):
        order = []
        source = whisper_dictate.WezTermTarget(7, "42")
        source.line_context = lambda: ("", "")
        source.press_enter = lambda: order.append("ENTER") or True
        with mock.patch.object(whisper_dictate.subprocess, "run",
                               side_effect=lambda cmd, **k: order.append(cmd[-1])) as run, \
                mock.patch.object(whisper_dictate.time, "sleep"), mock.patch("builtins.print"):
            app().output_text("/compact", line_source=source, press_enter=True)
        self.assertEqual(order, ["/compact", "ENTER"])

    def test_enter_alone_without_line_source_uses_xdotool(self):
        with mock.patch.object(whisper_dictate.subprocess, "run") as run, mock.patch("builtins.print"):
            app().output_text("", press_enter=True)
        self.assertEqual(run.call_args[0][0], ["xdotool", "key", "--clearmodifiers", "Return"])


class LivePostprocessTests(unittest.TestCase):
    def test_live_corrections_do_not_retype_the_whole_window(self):
        # One-shot tidying ("Okay." -> "okay") made the fast pass and the
        # correction differ at the first character.
        import numpy as np
        from live_controller import LiveController
        from live_segmenter import ChunkReady
        from live_session import Edit, LiveSession

        a = app()
        a.load_replacements = lambda: {"zzqx": "q"}
        edits = []

        class Target:
            name = "t"

            def send(self, edit):
                edits.append(edit)
                return True

            def still_focused(self):
                return True

        def script(*results):
            results = list(results)
            return lambda audio, prompt: results.pop(0)

        def chunk(t0, t1):
            return ChunkReady(np.zeros(int(16000 * (t1 - t0)), dtype=np.float32), t0, t1)

        postprocess = lambda text: a.apply_replacements(
            text, lower_single_word=False, strip_trailing_period=False)
        ctl = LiveController(LiveSession(), Target(), script("Okay.", "I got to go ahead."),
                             script("Okay, I got to go ahead."), postprocess=postprocess,
                             log=lambda *_: None)
        with mock.patch("builtins.print"):
            ctl.process(chunk(0.0, 1.0))
            ctl.process(chunk(1.0, 2.0))
            ctl.correct()
        self.assertEqual(edits[0], Edit(0, "Okay."))
        self.assertEqual(edits[-1], Edit(20, ", I got to go ahead."))  # keeps "Okay"


if __name__ == "__main__":
    unittest.main()
