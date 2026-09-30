import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_rate import resample


def tone(freq, rate, seconds=1.0):
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class ResampleTests(unittest.TestCase):
    def test_same_rate_is_untouched(self):
        a = tone(440, 16000)
        self.assertIs(resample(a, 16000, 16000), a)

    def test_24k_to_16k_keeps_length_and_speech_band(self):
        out = resample(tone(1000, 24000), 24000, 16000)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 16000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(out[1000:-1000] ** 2))), 0.707, places=2)

    def test_above_new_nyquist_is_filtered_not_aliased(self):
        # 10 kHz fits at 24 kHz but not at 16 kHz; it must vanish, not fold to 6 kHz.
        out = resample(tone(10000, 24000), 24000, 16000)
        self.assertLess(float(np.sqrt(np.mean(out[1000:-1000] ** 2))), 0.05)


if __name__ == "__main__":
    unittest.main()


class AppRateTests(unittest.TestCase):
    def app(self):
        from whisper_dictate import WhisperDictate
        a = object.__new__(WhisperDictate)
        a.model = None
        a.config = {"model": "gpt-transcribe", "language": "en", "sample_rate": 48000,
                    "api_sample_rate": 24000,
                    "remote_server": {"enabled": True, "host": "h", "port": 1, "model": "medium.en"}}
        a.get_asr_context = lambda: ""
        return a

    def test_hosted_model_gets_api_rate(self):
        from unittest import mock
        a = self.app()
        with mock.patch("whisper_dictate.transcribe_openrouter", return_value="x") as t:
            a.transcribe_openrouter(tone(440, 48000), "gpt-transcribe")
        self.assertEqual(len(t.call_args.args[0]), 24000)
        self.assertEqual(t.call_args.kwargs["sample_rate"], 24000)

    def test_local_whisper_gets_16k(self):
        from unittest import mock
        a = self.app()
        a.model = mock.MagicMock()
        a.model.transcribe.return_value = {"text": "x"}
        a._model_backend = "whisper"
        with mock.patch.object(a, "load_model"):
            a._transcribe_local(tone(440, 48000), model_name="base", prompt="")
        self.assertEqual(len(a.model.transcribe.call_args.args[0]), 16000)
