import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_dictate import WhisperDictate


def app(remote_enabled):
    a = object.__new__(WhisperDictate)
    a.config = {"model": "qwen3-asr-1.7b", "cpu_fallback_model": "base", "language": "en",
                "remote_server": {"enabled": remote_enabled}}
    return a


class PreloadModelTests(unittest.TestCase):
    def test_remote_on_preloads_only_the_local_fallback(self):
        # The main model runs on the server; locally only the fallback is ever used.
        self.assertEqual(app(True).preload_model_name(), "base")

    def test_remote_off_preloads_the_main_model(self):
        self.assertEqual(app(False).preload_model_name(), "qwen3-asr-1.7b")


if __name__ == "__main__":
    unittest.main()
