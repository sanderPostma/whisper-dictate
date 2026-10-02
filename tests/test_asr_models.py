import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_models import (
    OPENROUTER_MODELS,
    QWEN_MODELS,
    build_remote_header,
    effective_remote_model,
    fallback_chain,
    is_distil_model,
    is_english_only_model,
    is_openrouter_model,
    is_qwen_model,
    multilingual_model,
    openrouter_model_id,
    openrouter_provider_tag,
    qwen_hf_id,
)


class OpenRouterModelMappingTests(unittest.TestCase):
    def test_openrouter_model_ids(self):
        self.assertEqual(openrouter_model_id("gpt-transcribe"), "openai/gpt-transcribe")
        self.assertEqual(openrouter_model_id("openai/gpt-transcribe"), "openai/gpt-transcribe")
        self.assertEqual(openrouter_model_id("mai-transcribe-2"), "microsoft/mai-transcribe-2")
        self.assertEqual(openrouter_model_id("microsoft/mai-transcribe-2"), "microsoft/mai-transcribe-2")
        with self.assertRaises(ValueError):
            openrouter_model_id("non-existent-model")

    def test_openrouter_provider_tags(self):
        self.assertEqual(openrouter_provider_tag("openai/gpt-transcribe"), "openai")
        self.assertEqual(openrouter_provider_tag("microsoft/mai-transcribe-2"), "azure")
        self.assertEqual(openrouter_provider_tag("custom/model"), "custom")


class FallbackChainTests(unittest.TestCase):
    def test_keeps_order_and_drops_blanks_and_duplicates(self):
        self.assertEqual(
            fallback_chain(["gpt-transcribe", "", "qwen3-asr-1.7b", "gpt-transcribe", "base"]),
            ["gpt-transcribe", "qwen3-asr-1.7b", "base"],
        )

    def test_non_english_rewrites_english_only_ids(self):
        self.assertEqual(
            fallback_chain(["base.en", "distil-large-v3", "qwen3-asr-1.7b"], language="nl"),
            ["base", "large", "qwen3-asr-1.7b"],
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
