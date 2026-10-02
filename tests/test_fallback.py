import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_dictate import WhisperDictate


def app():
    a = object.__new__(WhisperDictate)
    a.config = {
        "model": "mai-transcribe-2",
        "language": "en",
        "fallback_models": ["gpt-transcribe", "qwen3-asr-1.7b", "base"],
        "remote_server": {"enabled": True, "model": "medium"},
    }
    return a


class FallbackWalkTests(unittest.TestCase):
    def test_chain_is_gpt_then_qwen_then_local(self):
        self.assertEqual(
            app().get_fallback_models(),
            ["gpt-transcribe", "qwen3-asr-1.7b", "base"],
        )
        # Preload can only warm a local model. The first of those is Qwen.
        self.assertEqual(app().get_cpu_fallback_model(), "qwen3-asr-1.7b")

    def test_legacy_single_fallback_still_works(self):
        a = app()
        del a.config["fallback_models"]
        a.config["cpu_fallback_model"] = "base"
        self.assertEqual(a.get_fallback_models(), ["base"])

    def test_walk_skips_the_model_that_just_failed(self):
        a = app()
        a.config["fallback_models"] = ["mai-transcribe-2", "gpt-transcribe", "base"]
        calls = []

        def one(_audio, model, _prompt, _timeout):
            calls.append(model)
            if model != "base":
                raise RuntimeError("down")
            return "heard"

        a._transcribe_one = one
        text = a._transcribe_fallbacks(object(), None, skip="mai-transcribe-2")
        self.assertEqual(text, "heard")
        self.assertEqual(calls, ["gpt-transcribe", "base"])


if __name__ == "__main__":
    unittest.main()
