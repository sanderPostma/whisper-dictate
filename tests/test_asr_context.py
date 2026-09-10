import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_context import CONTEXT_PACKS, context_from_config, resolve_asr_context


class ResolveAsrContextTests(unittest.TestCase):
    def test_none_pack_returns_empty_without_extra(self):
        self.assertEqual(resolve_asr_context("none"), "")

    def test_developer_pack_includes_core_jargon(self):
        text = resolve_asr_context("developer")
        self.assertTrue(text.startswith("Vocabulary:"))
        for term in ("git", "kubectl", "YAML", "pytest", "Whisper", "Qwen"):
            self.assertIn(term, text)

    def test_unknown_pack_is_empty(self):
        self.assertEqual(resolve_asr_context("not-a-pack"), "")

    def test_extra_prompt_appends_to_pack(self):
        text = resolve_asr_context("developer", extra="Names: Sphereon, KTOR")
        self.assertIn("git", text)
        self.assertIn("Names: Sphereon, KTOR", text)

    def test_custom_pack_reads_context_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context.txt"
            path.write_text("Vocabulary: beads, dolt\n", encoding="utf-8")
            text = resolve_asr_context("custom", context_file=path)
            self.assertIn("beads", text)
            self.assertIn("dolt", text)

    def test_none_pack_still_uses_extra_prompt(self):
        self.assertEqual(
            resolve_asr_context("none", extra="Vocabulary: NixOS"),
            "Vocabulary: NixOS",
        )

    def test_developer_pack_is_registered(self):
        self.assertIn("developer", CONTEXT_PACKS)
        self.assertIn("none", CONTEXT_PACKS)
        self.assertIn("custom", CONTEXT_PACKS)

    def test_context_from_config_uses_pack_extra_and_custom_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context.txt"
            path.write_text("Vocabulary: NixOS", encoding="utf-8")
            text = context_from_config(
                {"context_pack": "custom", "context_prompt": "Names: Sander"},
                tmp,
            )
            self.assertIn("NixOS", text)
            self.assertIn("Names: Sander", text)
        developer = context_from_config({"context_pack": "developer"}, "/nonexistent")
        self.assertIn("git", developer)


if __name__ == "__main__":
    unittest.main()
