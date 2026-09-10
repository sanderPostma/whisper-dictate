import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_models import (
    QWEN_MODELS,
    build_remote_header,
    effective_remote_model,
    is_distil_model,
    is_english_only_model,
    is_qwen_model,
    multilingual_model,
    qwen_hf_id,
)


class QwenModelMappingTests(unittest.TestCase):
    def test_qwen_1_7b_maps_to_hf_checkpoint(self):
        self.assertEqual(qwen_hf_id("qwen3-asr-1.7b"), "Qwen/Qwen3-ASR-1.7B-hf")

    def test_is_qwen_model_accepts_known_ids(self):
        self.assertTrue(is_qwen_model("qwen3-asr-1.7b"))
        self.assertIn("qwen3-asr-1.7b", QWEN_MODELS)
        self.assertFalse(is_qwen_model("large"))
        self.assertFalse(is_qwen_model("distil-large-v3"))

    def test_qwen_is_not_english_only_or_distil(self):
        self.assertFalse(is_english_only_model("qwen3-asr-1.7b"))
        self.assertFalse(is_distil_model("qwen3-asr-1.7b"))

    def test_multilingual_rewrite_leaves_qwen_unchanged(self):
        self.assertEqual(multilingual_model("qwen3-asr-1.7b"), "qwen3-asr-1.7b")
        self.assertEqual(multilingual_model("base.en"), "base")
        self.assertEqual(multilingual_model("distil-large-v3"), "large")


class RemoteHeaderTests(unittest.TestCase):
    def test_header_includes_prompt_when_set(self):
        header = build_remote_header(
            language="en",
            model="qwen3-asr-1.7b",
            sample_rate=16000,
            audio_size=32,
            prompt="Vocabulary: git, kubectl",
        )
        self.assertEqual(header["model"], "qwen3-asr-1.7b")
        self.assertEqual(header["prompt"], "Vocabulary: git, kubectl")
        self.assertEqual(header["language"], "en")
        self.assertEqual(header["audio_size"], 32)

    def test_header_omits_empty_prompt(self):
        header = build_remote_header(
            language="nl",
            model="large",
            sample_rate=16000,
            audio_size=8,
            prompt="",
        )
        self.assertNotIn("prompt", header)

    def test_effective_remote_model_prefers_local_qwen(self):
        self.assertEqual(
            effective_remote_model("qwen3-asr-1.7b", "distil-large-v3", "en"),
            "qwen3-asr-1.7b",
        )
        self.assertEqual(
            effective_remote_model("base", "distil-large-v3", "en"),
            "distil-large-v3",
        )
        self.assertEqual(
            effective_remote_model("base", "medium.en", "nl"),
            "medium",
        )


if __name__ == "__main__":
    unittest.main()
