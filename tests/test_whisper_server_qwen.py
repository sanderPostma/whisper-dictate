import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_server import WhisperTCPServer


class WhisperServerQwenTests(unittest.TestCase):
    def test_load_model_dispatches_to_qwen_loader(self):
        server = WhisperTCPServer("127.0.0.1", 0, None, "cuda")
        fake_model = MagicMock()
        fake_processor = MagicMock()
        with patch("whisper_server.load_qwen", return_value=(fake_model, fake_processor)) as load:
            obj, backend = server._load_model("qwen3-asr-1.7b")
        self.assertEqual(backend, "qwen")
        self.assertEqual(obj, (fake_model, fake_processor))
        load.assert_called_once_with("qwen3-asr-1.7b", "cuda")

    def test_transcribe_passes_prompt_to_qwen(self):
        server = WhisperTCPServer("127.0.0.1", 0, None, "cpu")
        model = MagicMock()
        processor = MagicMock()
        server.model_cache["qwen3-asr-1.7b"] = ((model, processor), "qwen")
        server.model_name = "qwen3-asr-1.7b"
        audio = np.zeros(16, dtype=np.float32)
        with patch("whisper_server.transcribe_qwen", return_value="hello git") as transcribe:
            result = server._transcribe(
                audio, "en", "qwen3-asr-1.7b", prompt="Vocabulary: git"
            )
        self.assertEqual(result["text"], "hello git")
        transcribe.assert_called_once()
        args, kwargs = transcribe.call_args
        self.assertIs(args[0], model)
        self.assertIs(args[1], processor)
        self.assertEqual(kwargs.get("language") or args[3], "en")
        self.assertEqual(kwargs.get("prompt") or args[4], "Vocabulary: git")

    def test_whisper_transcribe_gets_initial_prompt(self):
        server = WhisperTCPServer("127.0.0.1", 0, None, "cpu")
        model = MagicMock()
        model.transcribe.return_value = {"text": "rebase"}
        server.model_cache["base"] = (model, "whisper")
        server.model_name = "base"
        audio = np.zeros(8, dtype=np.float32)
        result = server._transcribe(audio, "en", "base", prompt="Vocabulary: rebase")
        self.assertEqual(result["text"], "rebase")
        _, kwargs = model.transcribe.call_args
        self.assertEqual(kwargs.get("initial_prompt"), "Vocabulary: rebase")


if __name__ == "__main__":
    unittest.main()
