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
