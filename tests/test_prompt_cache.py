"""Exact CPU tensors only; no model loading or GPU generation."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio"))
from prompt_cache import PromptCache


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "cache"
        self.embeds = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 8)
        self.mask = torch.tensor([[True, True, False]])
        self.encoder = Mock(return_value=(self.embeds, self.mask))

    def test_exact_roundtrip_across_instances(self):
        cache = PromptCache(self.root, {"version": 1})
        cache.encode("portrait", self.encoder, "cpu")
        self.assertEqual(cache.status, "miss")
        cache = PromptCache(self.root, {"version": 1})
        actual = cache.encode("portrait", self.encoder, "cpu")
        self.assertEqual(cache.status, "hit")
        self.encoder.assert_called_once()
        for expected, value in zip((self.embeds, self.mask), actual):
            self.assertEqual(value.dtype, expected.dtype)
            self.assertTrue(torch.equal(value, expected))

    def test_prompt_and_identity_invalidate(self):
        cache = PromptCache(self.root, {"version": 1})
        cache.encode("one", self.encoder, "cpu")
        cache.encode("two", self.encoder, "cpu")
        PromptCache(self.root, {"version": 2}).encode("one", self.encoder, "cpu")
        self.assertEqual(self.encoder.call_count, 3)

    def test_corrupt_cache_falls_back(self):
        cache = PromptCache(self.root, {})
        self.root.mkdir()
        (self.root / (cache.key("one") + ".safetensors")).write_bytes(b"broken")
        cache.encode("one", self.encoder, "cpu")
        self.assertEqual(cache.status, "miss")
        cache.encode("one", self.encoder, "cpu")
        self.assertEqual(cache.status, "hit")

    def test_private_never_accesses_cache(self):
        # Use an identity that cannot be serialized: even key construction is bypassed.
        cache = PromptCache(self.root, object(), private=True)
        cache.encode("secret", self.encoder, "cpu")
        self.assertEqual(cache.status, "private")
        self.assertFalse(self.root.exists())

    def test_entry_and_byte_bounds(self):
        cache = PromptCache(self.root, {}, max_entries=2)
        for prompt in ("one", "two", "three"):
            cache.encode(prompt, self.encoder, "cpu")
        self.assertEqual(len(list(self.root.glob("*.safetensors"))), 2)
        cache = PromptCache(self.root, {}, max_bytes=1)
        cache.encode("four", self.encoder, "cpu")
        self.assertEqual(list(self.root.glob("*.safetensors")), [])
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_unwritable_cache_falls_back(self):
        self.root.write_text("not a directory")
        cache = PromptCache(self.root, {})
        actual = cache.encode("one", self.encoder, "cpu")
        self.assertIs(actual[0], self.embeds)
        self.assertEqual(cache.status, "miss (cache unavailable)")


if __name__ == "__main__":
    unittest.main()
