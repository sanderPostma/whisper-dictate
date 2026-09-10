import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_qwen import load_qwen, transcribe_qwen


class TranscribeQwenTests(unittest.TestCase):
    def test_passes_audio_language_and_prompt_then_returns_text(self):
        processor = MagicMock()
        model = MagicMock()
        model.device = "cpu"
        model.dtype = "float32"

        input_ids = MagicMock()
        input_ids.shape = [1, 4]
        inputs = MagicMock()
        inputs.__getitem__.side_effect = lambda key: input_ids if key == "input_ids" else None
        inputs.to.return_value = inputs
        processor.apply_transcription_request.return_value = inputs

        generated = object()
        output_ids = MagicMock()
        output_ids.__getitem__.return_value = generated
        model.generate.return_value = output_ids
        processor.decode.return_value = ["  kubectl apply  "]

        audio = np.zeros(1600, dtype=np.float32)
        text = transcribe_qwen(
            model,
            processor,
            audio,
            language="en",
            prompt="Vocabulary: kubectl",
        )

        self.assertEqual(text, "kubectl apply")
        call_kwargs = processor.apply_transcription_request.call_args.kwargs
        self.assertIs(call_kwargs["audio"], audio)
        self.assertEqual(call_kwargs["language"], "en")
        self.assertEqual(call_kwargs["prompt"], "Vocabulary: kubectl")
        processor.decode.assert_called_once()
        _, decode_kwargs = processor.decode.call_args
        self.assertEqual(decode_kwargs.get("return_format"), "transcription_only")
        generate_kwargs = model.generate.call_args.kwargs
        self.assertGreaterEqual(generate_kwargs.get("max_new_tokens", 0), 256)

    def test_omits_blank_prompt_and_language(self):
        processor = MagicMock()
        model = MagicMock()
        model.device = "cpu"
        model.dtype = None
        inputs = MagicMock()
        input_ids = MagicMock()
        input_ids.shape = [1, 1]
        inputs.__getitem__.return_value = input_ids
        inputs.to.return_value = inputs
        processor.apply_transcription_request.return_value = inputs
        output_ids = MagicMock()
        output_ids.__getitem__.return_value = MagicMock()
        model.generate.return_value = output_ids
        processor.decode.return_value = "ok"

        transcribe_qwen(model, processor, np.zeros(8, dtype=np.float32), language="", prompt="  ")
        call_kwargs = processor.apply_transcription_request.call_args.kwargs
        self.assertNotIn("prompt", call_kwargs)
        self.assertNotIn("language", call_kwargs)


class LoadQwenTests(unittest.TestCase):
    def test_loads_hf_native_checkpoint_on_requested_device(self):
        fake_processor = object()
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model

        with patch("asr_qwen.AutoProcessor") as auto_proc, patch(
            "asr_qwen.AutoModelForMultimodalLM"
        ) as auto_model, patch("asr_qwen.torch") as torch_mod:
            torch_mod.bfloat16 = "bf16"
            torch_mod.float32 = "fp32"
            auto_proc.from_pretrained.return_value = fake_processor
            auto_model.from_pretrained.return_value = fake_model

            model, processor = load_qwen("qwen3-asr-1.7b", device="cuda")

        self.assertIs(model, fake_model)
        self.assertIs(processor, fake_processor)
        auto_proc.from_pretrained.assert_called_once_with("Qwen/Qwen3-ASR-1.7B-hf")
        auto_model.from_pretrained.assert_called_once()
        _, kwargs = auto_model.from_pretrained.call_args
        self.assertEqual(kwargs.get("dtype"), "bf16")
        fake_model.to.assert_called_once_with("cuda")
        fake_model.eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
