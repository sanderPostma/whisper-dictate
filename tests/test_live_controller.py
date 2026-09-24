import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_controller import LiveController, select_transcribers
from live_segmenter import ChunkReady, LongPause
from live_session import Edit, LiveSession


def chunk(t0, t1):
    return ChunkReady(np.zeros(int(16000 * (t1 - t0)), dtype=np.float32), t0, t1)


class FakeTarget:
    name = "fake"

    def __init__(self, ok=True):
        self.edits = []
        self.focused = True
        self.ok = ok

    def send(self, edit):
        self.edits.append(edit)
        return self.ok

    def still_focused(self):
        return self.focused


class Script:
    """Transcriber returning scripted results in order; Exception items are raised."""

    def __init__(self, *results):
        self.results = list(results)
        self.prompts = []

    def __call__(self, audio, prompt):
        self.prompts.append(prompt)
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class LiveControllerTests(unittest.TestCase):
    def make(self, fast, correct=None, target=None, **kwargs):
        self.target = target or FakeTarget()
        self.errors = []
        self.done = []
        return LiveController(
            LiveSession(), self.target, fast, correct,
            on_error=self.errors.append, on_done=lambda: self.done.append(True),
            log=lambda *_: None, **kwargs,
        )

    def test_fast_result_is_typed(self):
        ctl = self.make(Script("Hello there"))
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(self.target.edits, [Edit(0, "Hello there")])

    def test_postprocess_applied_before_typing(self):
        ctl = self.make(Script("hello cube control"),
                        postprocess=lambda t: t.replace("cube control", "kubectl"))
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(self.target.edits, [Edit(0, "hello kubectl")])

    def test_fast_prompt_carries_previous_text(self):
        fast = Script("Hello", "world")
        ctl = self.make(fast)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(fast.prompts, ["", "Hello"])

    def test_correction_rewrites_tail(self):
        correct = Script("We should merge the branch today")
        ctl = self.make(Script("We should merge the", "brunch today"), correct)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.target.edits[-1], Edit(10, "anch today"))
        self.assertEqual(correct.prompts, [""])

    def test_single_typed_chunk_skips_correction(self):
        correct = Script()
        ctl = self.make(Script("Hello"), correct)
        ctl.process(chunk(0.0, 1.0))
        ctl.correct()
        self.assertEqual(correct.prompts, [])

    def test_focus_change_commits_and_retargets(self):
        new_target = FakeTarget()
        correct = Script()
        ctl = self.make(Script("Hello there", "More"), correct, make_target=lambda: new_target)
        ctl.process(chunk(0.0, 1.0))
        self.target.focused = False
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.target.edits, [Edit(0, "Hello there")])
        self.assertEqual(new_target.edits, [Edit(0, " more")])
        self.assertIs(ctl.target, new_target)
        self.assertEqual(correct.prompts, [])

    def test_focus_lost_before_correction_skips_it(self):
        correct = Script("x")
        ctl = self.make(Script("a", "b"), correct)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.target.focused = False
        ctl.correct()
        self.assertEqual(correct.prompts, [])
        self.assertEqual(ctl.session.chunk_count, 0)
        self.assertEqual(len(self.target.edits), 2)

    def test_fast_failure_recovered_by_correction(self):
        ctl = self.make(Script("Hello there", RuntimeError("timeout"), RuntimeError("again")),
                        Script("Hello there general Kenobi"))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.target.edits[-1], Edit(0, " general Kenobi"))
        ctl.process(chunk(2.0, 3.0))
        self.assertEqual(len(self.errors), 1)

    def test_correction_failure_is_silent(self):
        ctl = self.make(Script("a", "b"), Script(RuntimeError("boom")))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(self.errors, [])
        self.assertEqual(len(self.target.edits), 2)

    def test_output_failure_disables_corrections(self):
        correct = Script()
        ctl = self.make(Script("a", "b"), correct, target=FakeTarget(ok=False))
        ctl.process(chunk(0.0, 1.0))
        self.assertFalse(ctl.corrections_enabled)
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(correct.prompts, [])

    def test_recoverable_rejection_commits_but_keeps_corrections(self):
        target = FakeTarget(ok=False)
        target.recoverable_rejections = True
        ctl = self.make(Script("a"), Script(), target=target)
        ctl.process(chunk(0.0, 1.0))
        self.assertTrue(ctl.corrections_enabled)
        self.assertEqual(ctl.session.chunk_count, 0)

    def test_long_pause_after_sentence_commits(self):
        ctl = self.make(Script("Done."))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(LongPause(2.0))
        self.assertEqual(ctl.session.chunk_count, 0)
        self.assertEqual(ctl.session.committed_text, "Done.")

    def test_window_limit_commits_before_adding_chunk(self):
        ctl = self.make(Script("a", "b", "c"))
        ctl.process(chunk(0.0, 6.0))
        ctl.process(chunk(6.0, 11.0))
        ctl.process(chunk(11.0, 13.0))
        self.assertEqual(ctl.session.chunk_count, 1)
        self.assertEqual(ctl.session.committed_text, "a b")

    def test_no_correct_transcriber_means_no_corrections(self):
        ctl = self.make(Script("a", "b"))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(len(self.target.edits), 2)

    def test_loop_coalesces_corrections_and_finishes(self):
        fast = Script("We should merge the", "brunch")
        correct = Script("We should merge the branch")
        ctl = self.make(fast, correct)
        ctl.submit(chunk(0.0, 1.0))
        ctl.submit(chunk(1.0, 2.0))
        ctl.stop()
        ctl.run_until_stopped()
        self.assertEqual(len(fast.prompts), 2)
        self.assertEqual(len(correct.prompts), 1)
        self.assertEqual(self.target.edits[-1], Edit(4, "anch"))
        self.assertEqual(self.done, [True])
        self.assertEqual(ctl.session.chunk_count, 0)

    def test_worker_survives_a_crashing_target(self):
        class Boom(FakeTarget):
            def send(self, edit):
                raise RuntimeError("boom")

        ctl = self.make(Script("a"), target=Boom())
        ctl.submit(chunk(0.0, 1.0))
        ctl.stop()
        ctl.run_until_stopped()
        self.assertEqual(self.done, [True])

    def test_threaded_start_and_join(self):
        ctl = self.make(Script("Hello"))
        ctl.start()
        ctl.submit(chunk(0.0, 1.0))
        ctl.stop()
        ctl.join(timeout=5)
        self.assertFalse(ctl.is_running())
        self.assertEqual(self.target.edits, [Edit(0, "Hello")])
        self.assertEqual(self.done, [True])

    def test_target_send_raises_disables_corrections(self):
        class RaisingTarget(FakeTarget):
            def send(self, edit):
                raise RuntimeError("target send crashed")

        ctl = self.make(Script("a"), target=RaisingTarget())
        ctl.process(chunk(0.0, 1.0))
        self.assertFalse(ctl.corrections_enabled)
        self.assertEqual(ctl.session.chunk_count, 0)

    def test_on_done_called_even_if_commit_raises(self):
        class RaisingSession:
            def commit(self):
                raise RuntimeError("commit failed")
            def add_chunk(self, audio, t0, t1):
                pass
            def should_commit(self, **kwargs):
                return False
            def fast_prompt(self):
                return ""
            def add_fast_result(self, t1, text):
                return Edit(0, text)
            def window_audio(self):
                return None, 0
            def correction_prompt(self):
                return ""
            def apply_correction(self, t_upto, text):
                return None
            chunk_count = 0
            piece_count = 1

        done = []
        target = FakeTarget()
        ctl = LiveController(
            RaisingSession(), target, Script("a"),
            on_done=lambda: done.append(True),
            log=lambda *_: None,
        )
        ctl.submit(chunk(0.0, 1.0))
        ctl.stop()
        ctl.run_until_stopped()
        self.assertEqual(done, [True])


class LineTextTarget(FakeTarget):
    """A target that knows the text around its cursor (like achat)."""

    def __init__(self, line, ok=True, after=""):
        super().__init__(ok)
        self.line = line
        self.after = after
        self.dropped = False

    def line_context(self, adopt_rev=True):
        return None if self.line is None else (self.line, self.after)

    @property
    def last_send_dropped(self):
        return self.dropped


class LineContextTests(unittest.TestCase):
    def make(self, fast, target):
        self.errors = []
        return LiveController(LiveSession(), target, fast, None,
                              on_error=self.errors.append, log=lambda *_: None)

    def test_existing_line_text_is_context_for_the_first_chunk(self):
        target = LineTextTarget("fix the")
        fast = Script("Widget")
        ctl = self.make(fast, target)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(target.edits, [Edit(0, " widget")])
        self.assertEqual(fast.prompts, ["fix the"])

    def test_line_is_reread_after_a_rejection(self):
        target = LineTextTarget("", ok=False)
        target.recoverable_rejections = True
        ctl = self.make(Script("Hello", "there"), target)
        ctl.process(chunk(0.0, 1.0))
        target.ok = True
        target.line = "Hello typed "
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(target.edits[-1], Edit(0, "there"))

    def test_mid_line_insert_gets_trailing_space(self):
        target = LineTextTarget("His is a ", after="test sentence.")
        fast = Script("Quick", "brown")
        ctl = self.make(fast, target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(target.edits, [Edit(0, "quick "), Edit(0, "brown ")])
        self.assertEqual(fast.prompts, ["His is a", "His is a quick"])

    def test_after_text_cleared_when_focus_moves_to_plain_target(self):
        plain = FakeTarget()
        target = LineTextTarget("His is a ", after="test.")
        ctl = LiveController(LiveSession(), target, Script("quick", "more"), None,
                             make_target=lambda: plain, log=lambda *_: None)
        ctl.process(chunk(0.0, 1.0))
        target.focused = False
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(plain.edits, [Edit(0, "more")])

    def test_line_not_reread_mid_window(self):
        target = LineTextTarget("")
        ctl = self.make(Script("Hello", "there"), target)
        ctl.process(chunk(0.0, 1.0))
        target.line = "something else"
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(target.edits[-1], Edit(0, " there"))

    def test_unknown_line_keeps_session_context(self):
        target = LineTextTarget(None)
        ctl = self.make(Script("Hello"), target)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(target.edits, [Edit(0, "Hello")])

    def test_line_text_failure_keeps_the_chunk(self):
        class Broken(LineTextTarget):
            def line_context(self):
                raise RuntimeError("boom")
        target = Broken("")
        ctl = self.make(Script("Hello"), target)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(target.edits, [Edit(0, "Hello")])

    def test_dropped_text_is_reported_once(self):
        target = LineTextTarget("", ok=False)
        target.recoverable_rejections = True
        target.dropped = True
        ctl = self.make(Script("a", "b"), target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(len(self.errors), 1)

    def test_rejection_that_still_typed_is_not_reported(self):
        target = LineTextTarget("", ok=False)
        target.recoverable_rejections = True
        ctl = self.make(Script("a"), target)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(self.errors, [])


class CommandTests(unittest.TestCase):
    def make(self, fast, target=None, correct=None):
        self.target = target or FakeTarget()
        return LiveController(LiveSession(), self.target, fast, correct, log=lambda *_: None)

    def test_period_joins_after_a_thinking_pause(self):
        self.target = LineTextTarget("")
        ctl = self.make(Script("I want to.", "test more, period"), target=self.target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(LongPause(2.5))  # committed
        self.target.line = "I want to."
        ctl.process(chunk(6.0, 7.0))
        self.assertEqual(self.target.edits, [Edit(0, "I want to."), Edit(1, " test more.")])
        self.assertEqual(ctl.session.chunk_count, 0)

    def test_command_words_never_typed_from_a_correction(self):
        correct = Script("I want to test more. Period.")
        ctl = self.make(Script("I want to", "test more"), correct=correct)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(len(self.target.edits), 2)

    def test_scratch_that(self):
        ctl = self.make(Script("Done.", "I want to go home.", "scratch that"))
        for i in range(3):
            ctl.process(chunk(float(i), i + 1.0))
        self.assertEqual(self.target.edits[-1], Edit(19, ""))

    def test_focus_change_forgets_history(self):
        new_target = FakeTarget()
        ctl = self.make(Script("Done.", "scratch that"))
        ctl.make_target = lambda: new_target
        ctl.process(chunk(0.0, 1.0))
        self.target.focused = False
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(new_target.edits, [])

    def test_failed_send_forgets_history(self):
        target = FakeTarget(ok=False)
        target.recoverable_rejections = True
        ctl = self.make(Script("Done.", "scratch that"), target=target)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(ctl.session.history, "")

    def test_exact_line_that_changed_forgets_history(self):
        target = LineTextTarget("")
        target.exact_line = True
        ctl = self.make(Script("I want to.", "test more, period"), target=target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(LongPause(2.5))
        target.line = "I want to. And the operator typed"
        ctl.process(chunk(6.0, 7.0))
        self.assertEqual(target.edits[-1].backspace, 0)  # nothing of theirs erased

    def test_exact_line_unchanged_allows_the_repair(self):
        target = LineTextTarget("")
        target.exact_line = True
        ctl = self.make(Script("I want to.", "test more, period"), target=target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(LongPause(2.5))
        target.line = "I want to."
        ctl.process(chunk(6.0, 7.0))
        self.assertEqual(target.edits[-1], Edit(1, " test more."))


class CommandSafetyTests(unittest.TestCase):
    """Review findings: commands must never let a later edit reach operator text."""

    def make(self, fast, target, correct=None):
        return LiveController(LiveSession(), target, fast, correct, log=lambda *_: None)

    def test_failed_verification_does_not_leave_the_window_open(self):
        target = LineTextTarget("")
        target.exact_line = True
        ctl = self.make(Script("Hello there.", "Scratch that."), target)
        ctl.process(chunk(0.0, 1.0))
        target.line = "Hello there. Operator notes"
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(ctl.session.chunk_count, 0)  # window committed: no correction can reach back
        self.assertEqual(len(target.edits), 1)

    def test_correction_with_a_command_commits_the_window(self):
        target = FakeTarget()
        ctl = self.make(Script("Hello there", "and more"), target,
                        correct=Script("Hello there. Period. And more"))
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        ctl.correct()
        self.assertEqual(ctl.session.chunk_count, 0)
        self.assertNotIn("Period", "".join(e.insert for e in target.edits))

    def test_screen_target_forgets_history_when_the_row_changed(self):
        target = LineTextTarget("")
        ctl = self.make(Script("Fix the bug.", "Scratch that."), target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(LongPause(2.5))
        target.line = "yes"  # operator pressed Enter and typed
        ctl.process(chunk(5.0, 6.0))
        self.assertEqual(len(target.edits), 1)

    def test_screen_target_keeps_history_when_the_row_matches(self):
        target = LineTextTarget("")
        ctl = self.make(Script("Done.", "Oops.", "Scratch that."), target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        target.line = "Done. Oops."
        ctl.process(chunk(2.0, 3.0))
        self.assertEqual(target.edits[-1], Edit(6, ""))

    def test_keystroke_target_limits_commands_to_the_window(self):
        target = FakeTarget()  # no line_context: nothing to check against
        ctl = self.make(Script("Fix the bug.", "Scratch that."), target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(LongPause(2.5))  # committed; the operator may have typed since
        ctl.process(chunk(5.0, 6.0))
        self.assertEqual(len(target.edits), 1)


class KeystrokeLongPauseTests(unittest.TestCase):
    def test_long_pause_closes_the_window_on_a_target_without_line(self):
        # Re-review: "hello world" (no full stop), long pause, "scratch that"
        # backspaced at wherever the operator's cursor had moved to.
        target = FakeTarget()
        ctl = LiveController(LiveSession(), target, Script("hello world", "Scratch that."), None,
                             log=lambda *_: None)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(LongPause(2.5))
        self.assertEqual(ctl.session.chunk_count, 0)
        ctl.process(chunk(5.0, 6.0))
        self.assertEqual(len(target.edits), 1)


class EnterTarget(FakeTarget):
    def __init__(self):
        super().__init__()
        self.log = []

    def send(self, edit):
        self.log.append(edit)
        return super().send(edit)

    def press_enter(self):
        self.log.append("ENTER")
        return True


class EnterTests(unittest.TestCase):
    def make(self, fast, target):
        return LiveController(LiveSession(), target, fast, None, log=lambda *_: None)

    def test_press_enter_types_then_submits(self):
        target = EnterTarget()
        ctl = self.make(Script("/compact, press enter."), target)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(target.log, [Edit(0, "/compact"), "ENTER"])

    def test_after_enter_the_line_starts_fresh(self):
        target = EnterTarget()
        ctl = self.make(Script("Do it, press enter.", "Next one"), target)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(target.log[-1], Edit(0, "Next one"))
        self.assertEqual(ctl.session.history, "Next one")

    def test_nothing_crosses_a_submitted_prompt(self):
        target = EnterTarget()
        ctl = self.make(Script("Do it.", "Press enter.", "Scratch that."), target)
        for i in range(3):
            ctl.process(chunk(float(i), i + 1.0))
        self.assertEqual(target.log, [Edit(0, "Do it."), "ENTER"])

    def test_enter_on_a_target_that_cannot_press_it(self):
        target = FakeTarget()
        ctl = self.make(Script("Do it, press enter."), target)
        ctl.process(chunk(0.0, 1.0))
        self.assertEqual(target.edits, [Edit(0, "Do it")])


class CursorMoveTests(unittest.TestCase):
    def test_move_commits_and_forgets_then_dictation_goes_on(self):
        from live_commands import CursorMove
        moves = []
        target = FakeTarget()
        target.move_cursor = lambda move: moves.append(move) or True
        ctl = LiveController(LiveSession(), target, Script("Fix the fox.", "Cursor back two words.", "brown"),
                             None, log=lambda *_: None)
        ctl.process(chunk(0.0, 1.0))
        ctl.process(chunk(1.0, 2.0))
        self.assertEqual(moves, [CursorMove("word", -1, 2)])
        self.assertEqual(ctl.session.history, "")
        self.assertEqual(ctl.session.chunk_count, 0)
        ctl.process(chunk(2.0, 3.0))
        self.assertEqual(target.edits[-1].backspace, 0)
        self.assertEqual(len(target.edits), 2)  # the command itself typed nothing


class SelectTranscribersTests(unittest.TestCase):
    def setUp(self):
        self.down = False
        self.results = []
        self.remote = Script()
        self.local = Script()
        self.fallback = Script()

    def select(self, remote_enabled=True):
        return select_transcribers(
            remote_enabled, lambda: self.down, self.remote, self.local, self.fallback,
            lambda ok, reason: self.results.append(ok),
        )

    def test_remote_disabled_uses_local_without_corrections(self):
        fast, correct = self.select(remote_enabled=False)
        self.assertIs(fast, self.local)
        self.assertIsNone(correct)

    def test_remote_up_uses_remote_for_both(self):
        self.remote.results = ["fast", "corrected"]
        fast, correct = self.select()
        self.assertEqual(fast(None, "p"), "fast")
        self.assertEqual(correct(None, "p"), "corrected")
        self.assertEqual(self.results, [True])

    def test_remote_error_falls_back_and_reports(self):
        self.remote.results = [RuntimeError("refused")]
        self.fallback.results = ["local"]
        fast, _ = self.select()
        self.assertEqual(fast(None, "p"), "local")
        self.assertEqual(self.results, [False])

    def test_remote_down_skips_remote_and_corrections_raise(self):
        self.down = True
        self.fallback.results = ["local"]
        fast, correct = self.select()
        self.assertEqual(fast(None, "p"), "local")
        self.assertEqual(self.remote.prompts, [])
        with self.assertRaises(RuntimeError):
            correct(None, "p")


if __name__ == "__main__":
    unittest.main()
