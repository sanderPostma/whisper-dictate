import base64
import io
import json
import sys
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asr_openrouter
from asr_models import effective_remote_model, is_english_only_model, is_openrouter_model, multilingual_model
from asr_openrouter import build_request_body, encode_wav, openrouter_api_key, transcribe_openrouter


class OpenRouterModelTests(unittest.TestCase):
    def test_gpt_transcribe_is_hosted_and_multilingual(self):
        self.assertTrue(is_openrouter_model("gpt-transcribe"))
        self.assertFalse(is_openrouter_model("qwen3-asr-1.7b"))
        self.assertFalse(is_english_only_model("gpt-transcribe"))
        self.assertEqual(multilingual_model("gpt-transcribe"), "gpt-transcribe")

    def test_locally_selected_hosted_model_wins_over_server_model(self):
        self.assertEqual(effective_remote_model("gpt-transcribe", "medium.en", "nl"), "gpt-transcribe")


class RequestTests(unittest.TestCase):
    def test_wav_is_16bit_mono_at_sample_rate(self):
        data = encode_wav(np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32), 16000)
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 16000))
            samples = np.frombuffer(w.readframes(4), dtype="<i2")
        self.assertEqual(list(samples), [0, 16383, -32767, 32767])

    def test_body_has_model_audio_language_and_provider_prompt(self):
        body = build_request_body("gpt-transcribe", b"RIFF", language="en", prompt="Vocabulary: VDX.")
        self.assertEqual(body["model"], "openai/gpt-transcribe")
        self.assertEqual(body["input_audio"], {"data": base64.b64encode(b"RIFF").decode(), "format": "wav"})
        self.assertEqual(body["language"], "en")
        self.assertEqual(body["provider"], {"options": {"openai": {"prompt": "Vocabulary: VDX."}}})

    def test_blank_language_and_prompt_are_left_out(self):
        body = build_request_body("gpt-transcribe", b"x", language="", prompt="  ")
        self.assertNotIn("language", body)
        self.assertNotIn("provider", body)

    def test_env_key_wins_over_config(self):
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "env"}):
            self.assertEqual(openrouter_api_key({"openrouter_api_key": "cfg"}), "env")
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(openrouter_api_key({"openrouter_api_key": "cfg"}), "cfg")
            self.assertEqual(openrouter_api_key({}), "")

    def test_missing_key_raises_before_any_request(self):
        with mock.patch.object(asr_openrouter.urllib.request, "urlopen") as urlopen:
            with self.assertRaises(RuntimeError):
                transcribe_openrouter(np.zeros(10, dtype=np.float32), "gpt-transcribe", "")
            urlopen.assert_not_called()

    def test_returns_stripped_text(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"text": " Hello there. ", "usage": {"seconds": 1, "cost": 0.0001}}
        ).encode()
        with mock.patch.object(asr_openrouter.urllib.request, "urlopen", return_value=response) as urlopen:
            text = transcribe_openrouter(np.zeros(10, dtype=np.float32), "gpt-transcribe", "k")
        self.assertEqual(text, "Hello there.")
        request = urlopen.call_args[0][0]
        self.assertEqual(request.get_header("Authorization"), "Bearer k")


if __name__ == "__main__":
    unittest.main()


class AppRoutingTests(unittest.TestCase):
    def app(self, remote_enabled):
        from whisper_dictate import WhisperDictate
        a = object.__new__(WhisperDictate)
        a.model = None
        a.config = {"model": "gpt-transcribe", "cpu_fallback_model": "base", "language": "en",
                    "sample_rate": 16000,
                    "remote_server": {"enabled": remote_enabled, "model": "medium.en"}}
        a.get_asr_context = lambda: "ctx"
        return a

    def test_remote_on_goes_to_openrouter_not_the_server(self):
        a = self.app(True)
        with mock.patch("whisper_dictate.transcribe_openrouter", return_value="hi") as t, \
                mock.patch("whisper_dictate.socket.create_connection") as conn:
            self.assertEqual(a.transcribe_remote(np.zeros(4, dtype=np.float32), prompt="p"), "hi")
        conn.assert_not_called()
        self.assertEqual(t.call_args.kwargs["prompt"], "p")

    def test_remote_off_goes_to_openrouter_without_loading(self):
        a = self.app(False)
        with mock.patch("whisper_dictate.transcribe_openrouter", return_value="hi"), \
                mock.patch("whisper_dictate.whisper.load_model") as load:
            a.load_model("gpt-transcribe")
            self.assertEqual(a._transcribe_local(np.zeros(4, dtype=np.float32)), "hi")
        load.assert_not_called()

    def test_probe_needs_only_an_api_key(self):
        a = self.app(True)
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}):
            self.assertTrue(a.probe_remote_service())
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(a.probe_remote_service())
