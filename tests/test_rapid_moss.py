"""CPU-only synthetic tensor checks. No image model, CUDA, server or generation."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio"))
from rapid_moss import RapidMossAdapter, sample_latents
import server


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(latent_channels=4)
        layer = torch.nn.Module()
        layer.proj = torch.nn.ModuleDict({"q": torch.nn.Linear(4, 4, bias=False)})
        self.layers = torch.nn.ModuleList([layer])
        self.time_embedding = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.SiLU(), torch.nn.Linear(4, 4))
        self.shared_block_modulation = torch.nn.Linear(4, 24)
        self.video_head_modulation = torch.nn.Linear(4, 8)


class RapidTests(unittest.TestCase):
    def setUp(self):
        # These tests cover a full BF16 install; adaptive defaults are tested separately.
        for target, value in ((server.engine, 'default_model'), (server.engine, 'bf16_available')):
            patcher = patch.object(target, value, return_value=server.engine.SFT_MODEL if value == 'default_model' else True)
            patcher.start(); self.addCleanup(patcher.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.manual_seed(17)
        self.model = TinyModel().bfloat16()
        self.down = torch.randn(2, 4) * .01
        self.up = torch.randn(4, 2) * .01
        prefix = "diffusion_model.layers.0.proj.q"
        self.lora = {prefix + ".lora_down.weight": self.down,
                     prefix + ".lora_up.weight": self.up,
                     prefix + ".alpha": torch.tensor(2.)}
        self.time = {"time_embedding." + k: torch.randn_like(v.float())
                     for k, v in self.model.time_embedding.state_dict().items()}
        for key in ("shared_block_modulation.bias", "video_head_modulation.bias"):
            self.time[key] = torch.randn_like(self.model.get_parameter(key).float())
        save_file(self.lora, self.root / "lora.safetensors")
        save_file(self.time, self.root / "time_embedding.safetensors")
        save_file({"latent_mix.Q": torch.eye(4), "step_codes": torch.tensor([1000, 968, 926, 871])},
                  self.root / "dataspace.safetensors")

    def test_fp32_unmerged_lora_survives_meta_reload_and_chunks_tokens(self):
        from accelerate.utils import set_module_tensor_to_device
        linear = self.model.layers[0].proj["q"]
        weight = linear.weight.detach().clone()
        # Mimic WeeLLM's pre-load and post-eviction surrounding each block.
        before = linear.register_forward_pre_hook(lambda module, args:
            set_module_tensor_to_device(module, "weight", "cpu", value=weight, dtype=torch.bfloat16))
        adapter = RapidMossAdapter(self.model, self.root)
        self.addCleanup(adapter.close)
        after = linear.register_forward_hook(lambda module, args, output:
            set_module_tensor_to_device(module, "weight", "meta"))
        self.addCleanup(before.remove)
        self.addCleanup(after.remove)
        x = torch.randn(2, 301, 4).bfloat16()
        expected = torch.nn.functional.linear(x, weight)
        delta = ((x.float() @ self.down.T) @ self.up.T).to(expected.dtype)
        expected = expected + delta
        with torch.no_grad(), patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            for _ in range(2):
                actual = linear(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(linear.weight.is_meta)
        self.assertFalse(any("lora" in key for key in self.model.state_dict()))

    def test_time_path_is_fp32_and_reapplied_after_resident_reload(self):
        adapter = RapidMossAdapter(self.model, self.root)
        self.addCleanup(adapter.close)
        for _ in range(2):
            self.model.bfloat16()
            adapter._restore_time_path(self.model, (torch.zeros(1, 4, 1, 1),))
            for name in ("time_embedding", "shared_block_modulation", "video_head_modulation"):
                self.assertTrue(all(p.dtype == torch.float32 for p in getattr(self.model, name).parameters()))
            for key, value in self.time.items():
                torch.testing.assert_close(self.model.get_parameter(key), value, rtol=0, atol=0)
        adapter.close()
        self.assertEqual(len(self.model._forward_pre_hooks), 0)

    def test_sampler_matches_release_recurrence_and_uses_exactly_four_calls(self):
        calls = []
        def tr(x, code, embeds, mask):
            calls.append(int(code))
            return types.SimpleNamespace(sample=x * .125 + code / 10000)
        codes = torch.tensor([1000, 968, 926, 871])
        Q = torch.eye(4) * .9
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            actual = sample_latents(tr, Q, codes, torch.zeros(1), torch.ones(1), 32, 32, 42, iter)
        z = torch.randn(1, 4, 2, 2, generator=torch.Generator("cpu").manual_seed(42))
        expected = torch.zeros_like(z)
        for code in codes:
            x = z + torch.einsum("oc,bchw->bohw", Q, expected)
            expected = z - (x * .125 + code / 10000)
        self.assertEqual(calls, codes.tolist())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cached_larger_chunks_preserve_fp32_math_and_release_cache(self):
        linear = self.model.layers[0].proj['q']
        x = torch.randn(1, 2053, 4).bfloat16()
        outputs = []
        for chunk, cache_mb in ((128, 0), (1024, 1)):
            adapter = RapidMossAdapter(self.model, self.root, chunk_tokens=chunk, cache_mb=cache_mb)
            try:
                with torch.no_grad():
                    first, second = linear(x), linear(x)
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                outputs.append(first)
                if cache_mb:
                    self.assertEqual(adapter.stats['cache_hits'], 1)
                    self.assertEqual(adapter.stats['cache_misses'], 1)
                    self.assertLessEqual(adapter.stats['cache_bytes'], cache_mb * 2**20)
                    self.assertTrue(all(t.dtype == torch.float32 for pair in adapter.cache.values() for t in pair))
                else:
                    self.assertEqual(adapter.stats['cache_hits'], 0)
                    self.assertEqual(adapter.cache, {})
            finally:
                adapter.close()
                self.assertEqual(adapter.cache, {})
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_unchanged_time_path_is_not_reloaded(self):
        adapter = RapidMossAdapter(self.model, self.root)
        self.addCleanup(adapter.close)
        args = (torch.zeros(1, 4, 1, 1),)
        adapter._restore_time_path(self.model, args)
        adapter._restore_time_path(self.model, args)
        self.assertEqual(adapter.stats['time_path_loads'], 1)
        self.model.bfloat16()  # Simulate WeeLLM restoring resident weights.
        adapter._restore_time_path(self.model, args)
        self.assertEqual(adapter.stats['time_path_loads'], 2)

    def test_stage_timings_synchronize_at_boundaries(self):
        from timing import StageTimings
        with patch('timing.time.perf_counter', side_effect=[10., 11., 13., 15.]):
            timings = StageTimings()
            from unittest.mock import Mock
            synchronize = Mock()
            timings.synchronize = synchronize
            with timings.stage('sampling'):
                pass
            self.assertEqual(timings.finish(), {'sampling': 2., 'worker_total': 5.})
            self.assertEqual(synchronize.call_count, 3)

    def test_missing_lora_tensor_fails_before_sampling(self):
        self.lora.pop("diffusion_model.layers.0.proj.q.alpha")
        save_file(self.lora, self.root / "lora.safetensors")
        with self.assertRaises(Exception):
            RapidMossAdapter(self.model, self.root)
        self.assertEqual(len(self.model.layers[0].proj["q"]._forward_hooks), 0)

    def test_actual_weellm_moss_class_with_tiny_cpu_weights_and_block_eviction(self):
        from accelerate.utils import set_module_tensor_to_device
        # Optional TorchAO/Triton probes GPU properties during import. Suppress
        # those probes so this compatibility check stays entirely on the CPU.
        with patch("torch.utils._triton.has_triton", return_value=False), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            from weellm.models.transformers.moss_image_transformer_2d_model import MOSSImageTransformer
        model = MOSSImageTransformer(dim=16, ffn_dim=32, num_heads=2,
            num_layers=2, text_dim=8, freq_dim=8, latent_channels=4).bfloat16()
        lora = {}
        for i, block in enumerate(model.layers):
            for name, linear in block.proj.items():
                key = f"diffusion_model.layers.{i}.proj.{name}"
                lora[key + ".lora_down.weight"] = torch.randn(2, linear.in_features) * .001
                lora[key + ".lora_up.weight"] = torch.randn(linear.out_features, 2) * .001
                lora[key + ".alpha"] = torch.tensor(2.)
            original = {k: v.detach().clone() for k, v in block.state_dict().items()}
            def reload(module, args, weights=original):
                module.load_state_dict(weights, assign=True)
            def evict(module, args, output):
                for key, _ in list(module.named_parameters()):
                    set_module_tensor_to_device(module, key, "meta")
            self.addCleanup(block.register_forward_pre_hook(reload).remove)
            self.addCleanup(block.register_forward_hook(evict).remove)
        time = {"time_embedding." + k: v.float() + .0001
                for k, v in model.time_embedding.state_dict().items()}
        for key in ("shared_block_modulation.bias", "video_head_modulation.bias"):
            time[key] = model.get_parameter(key).detach().float() + .0001
        save_file(lora, self.root / "lora.safetensors")
        save_file(time, self.root / "time_embedding.safetensors")
        adapter = RapidMossAdapter(model, self.root)
        self.addCleanup(adapter.close)
        x, emb, mask = torch.randn(1, 4, 2, 2), torch.randn(1, 3, 8).bfloat16(), torch.ones(1, 3)
        with torch.no_grad(), patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            first = model(x, torch.tensor(1000), emb, mask).sample
            second = model(x, torch.tensor(1000), emb, mask).sample
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(all(p.is_meta for block in model.layers for p in block.parameters()))
        self.assertEqual(model.shared_block_modulation.weight.dtype, torch.float32)

    def test_api_default_turbo_and_explicit_normal_override(self):
        server.JOBS.clear()
        server.ORDER.clear()
        try:
            with patch.object(server.engine, "turbo_available", return_value=True):
                server.submit({"prompt": "CPU validation only"})
                server.submit({"prompt": "CPU validation only", "turbo": False, "steps": 30})
            jobs = list(server.JOBS.values())
            self.assertEqual([(j['turbo'], j['steps']) for j in jobs], [(True, 4), (False, 30)])
        finally:
            server.JOBS.clear()
            server.ORDER.clear()

    def test_legacy_auto_cooldown_migrates_to_off(self):
        with patch.object(server, 'read_json', return_value={'cooldown': 'auto'}):
            self.assertEqual(server.load_settings()['cooldown'], 'off')
            with patch.object(server.time, 'time', side_effect=AssertionError('no delay expected')):
                server.plan_cooldown({})

    def test_default_download_selects_both_sft_and_adapter_without_network(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import download
        with patch.object(sys, 'argv', ['download.py']), \
                patch.object(download, 'download_model') as model, \
                patch.object(download, 'download_turbo') as adapter:
            download.main()
        model.assert_called_once_with('SFT', 32, 4)
        adapter.assert_called_once_with(32)

    def test_api_enforces_turbo_settings_and_rejects_missing_adapter_or_base(self):
        server.JOBS.clear()
        server.ORDER.clear()
        spec = {"prompt": "synthetic validation only", "turbo": True, "steps": 50, "cfg": 7, "negative": "unused"}
        with patch.object(server.engine, "turbo_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "download.ps1"):
                server.submit(spec)
        with patch.object(server.engine, "turbo_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "SFT"):
                server.submit({**spec, "model": "OpenMOSS-Team/MOSS-Image-Base"})
            server.submit(spec)
        job = next(iter(server.JOBS.values()))
        self.assertEqual((job["steps"], job["cfg"], job["negative"], job["sampler"]), (4, 1., "", "rapid-moss"))
        self.assertEqual(server.calib_key(job), "gen_rapid4")
        server.JOBS.clear()
        server.ORDER.clear()


if __name__ == "__main__":
    unittest.main()
