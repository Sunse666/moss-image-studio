"""CPU checks of profile isolation, quant decoding and flow schedule."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio'))
import int8_config
from int8_backend import simple_sigmas, sample_latents, checkpoint_seeker
import server


class Int8Tests(unittest.TestCase):
    def test_flow_euler_recurrence_and_eight_calls(self):
        calls = []
        def forward(latents, time, embeds, mask):
            calls.append(float(time))
            return types.SimpleNamespace(sample=torch.ones_like(latents))
        transformer = types.SimpleNamespace(config=types.SimpleNamespace(latent_channels=4), __call__=forward)
        class Model:
            config = transformer.config
            __call__ = staticmethod(forward)
        result = sample_latents(Model(), torch.zeros(1, 2, 8), torch.ones(1, 2), 32, 32, 42, iter)
        noise = torch.randn((1, 4, 2, 2), generator=torch.Generator('cpu').manual_seed(42))
        torch.testing.assert_close(result, noise - 1, rtol=1e-5, atol=1e-6)
        self.assertEqual(len(calls), 8)
        self.assertEqual(calls[0], 1000)
        self.assertAlmostEqual(calls[-1], 416.66666, places=3)
        sigmas = simple_sigmas()
        self.assertEqual(sigmas[-1], 0)
        self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]))

    def test_api_profile_settings_default_and_turbo_rejection(self):
        server.JOBS.clear(); server.ORDER.clear()
        self.addCleanup(server.JOBS.clear); self.addCleanup(server.ORDER.clear)
        with patch.object(int8_config, 'available', return_value=True):
            server.submit({'model': int8_config.MODEL, 'prompt': 'CPU test', 'steps': 4, 'cfg': 7, 'negative': 'unused'})
            job = next(iter(server.JOBS.values()))
            self.assertEqual((job['steps'], job['cfg'], job['negative'], job['turbo']), (8, 1, '', False))
            self.assertEqual((job['sampler'], job['scheduler'], job['flow_shift']), ('euler', 'simple', 5))
            self.assertEqual(server.calib_key(job), 'gen_int8flow8')
            with self.assertRaisesRegex(ValueError, 'Turbo 4-step'):
                server.submit({'model': int8_config.MODEL, 'prompt': 'CPU test', 'turbo': True})
        with patch.object(int8_config, 'available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'download.ps1 -Model INT8'):
                server.submit({'model': int8_config.MODEL, 'prompt': 'CPU test'})
        self.assertEqual(server.engine.DEFAULT_MODEL, 'OpenMOSS-Team/MOSS-Image-SFT')

    def test_real_weellm_convrot_decode_on_cpu(self):
        with patch('torch.utils._triton.has_triton', return_value=False), \
                patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
            from weellm.io.safetensors.disk_seek import SafetensorsDiskSeeker
            from weellm.io.safetensors.comfy_dequant import generate_hadamard
        # A full 256-channel rotation group, synthetic quantized weights only.
        q = torch.arange(512, dtype=torch.int32).remainder(127).to(torch.int8).reshape(2, 256)
        scale = torch.tensor([.25, .5])
        metadata = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.safetensors'
            save_file({'layers.0.proj.q.weight': q, 'layers.0.proj.q.weight_scale': scale,
                       'layers.0.proj.q.comfy_quant': torch.tensor(list(metadata), dtype=torch.uint8)}, str(path))
            seeker = SafetensorsDiskSeeker(path)
            actual = seeker.get_tensors(['layers.0.proj.q.weight'], device='cpu')['layers.0.proj.q.weight']
            expected = ((q.float() * scale[:, None]) @ generate_hadamard(256)).bfloat16()
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Actual new checkpoint layout: recipe in header, scale shaped [out,1].
            native = Path(directory) / 'native.safetensors'
            recipe = json.loads(metadata)
            save_file({'layers.0.proj.q.weight': q,
                       'layers.0.proj.q.weight_scale': scale[:, None]}, str(native),
                      metadata={'_quantization_metadata': json.dumps({'format_version': '1.0',
                                'layers': {'layers.0.proj.q': recipe}})})
            reader = checkpoint_seeker(native)
            raw = reader.get_tensors(['layers.0.proj.q.weight'], process_comfy=False)
            self.assertEqual(raw['layers.0.proj.q.weight'].dtype, torch.int8)
            self.assertIn('layers.0.proj.q.comfy_quant', raw)
            decoded = reader.get_tensors(['layers.0.proj.q.weight'])['layers.0.proj.q.weight']
            torch.testing.assert_close(decoded, expected, rtol=0, atol=0)

    def test_checkpoint_validator_rejects_wrong_shapes_and_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original'; original.mkdir()
            save_file({'layers.0.proj.q.weight': torch.ones(2, 256)}, str(original / 'model.safetensors'))
            quant = root / 'quant.safetensors'
            save_file({'layers.0.proj.q.weight': torch.ones(2, 128, dtype=torch.int8)}, str(quant))
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                int8_config.validate_checkpoint(quant, original)
            metadata = json.dumps({'format': 'unsupported'}).encode()
            save_file({'layers.0.proj.q.weight': torch.ones(2, 256, dtype=torch.int8),
                       'layers.0.proj.q.weight_scale': torch.ones(2),
                       'layers.0.proj.q.comfy_quant': torch.tensor(list(metadata), dtype=torch.uint8)}, str(quant))
            with self.assertRaisesRegex(ValueError, 'Unsupported INT8 recipe'):
                int8_config.validate_checkpoint(quant, original)


if __name__ == '__main__':
    unittest.main()
