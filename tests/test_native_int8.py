"""CPU validation of native INT8 storage, streaming and profile selection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio'))
from native_int8 import NativeInt8Linear, native_transformer
import int8_config
import server


class NativeTests(unittest.TestCase):
    def test_native_profile_selection_and_missing_runtime(self):
        server.JOBS.clear(); server.ORDER.clear()
        self.addCleanup(server.JOBS.clear); self.addCleanup(server.ORDER.clear)
        spec = {'model': int8_config.MODEL, 'prompt': 'CPU only', 'int8_compute': 'native'}
        with patch.object(int8_config, 'available', return_value=True):
            with patch.object(int8_config, 'native_available', return_value=False):
                with self.assertRaisesRegex(ValueError, 'optional'):
                    server.submit(spec)
            with patch.object(int8_config, 'native_available', return_value=True):
                server.submit(spec)
                job = next(iter(server.JOBS.values()))
                self.assertEqual((job['steps'], job['turbo'], server.calib_key(job)), (8, False, 'gen_int8native8'))
            with self.assertRaisesRegex(ValueError, 'native or bf16'):
                server.submit({**spec, 'int8_compute': 'unknown'})
        self.assertEqual(server.engine.DEFAULT_MODEL, 'OpenMOSS-Team/MOSS-Image-SFT')

    def test_actual_streamer_native_load_evict_reload(self):
        with patch('torch.utils._triton.has_triton', return_value=False), \
                patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
            from weellm.models.transformers.moss_image_transformer_2d_model import MOSSImageTransformer, MOSSImageTransformerStreamer
            from weellm.io.memory import evict_module
        model = MOSSImageTransformer(dim=256, ffn_dim=256, num_heads=2,
                                     num_layers=1, text_dim=8, freq_dim=8, latent_channels=4).bfloat16()
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        recipes = {}
        for name, linear in model.layers[0].proj.items():
            prefix = 'layers.0.proj.' + name
            weights[prefix + '.weight'] = torch.ones(linear.out_features, linear.in_features, dtype=torch.int8)
            weights[prefix + '.weight_scale'] = torch.full((linear.out_features, 1), .001, dtype=torch.float32)
            recipes[prefix] = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}
        def kernel(x, weight, scale, **kwargs):
            self.assertEqual(weight.dtype, torch.int8)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(x.dtype, torch.bfloat16)
            return torch.nn.functional.linear(x.float(), weight.float() * scale).to(torch.bfloat16)
        stats = {'linear_calls': 0, 'projections': 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model.save_config(root)
            path = root / 'model.safetensors'
            save_file(weights, str(path), metadata={'_quantization_metadata': json.dumps({'layers': recipes})})
            with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')), \
                    patch.object(torch.cuda, 'is_available', return_value=False), \
                    native_transformer(path, stats, kernel=kernel):
                streamer = MOSSImageTransformerStreamer.from_pretrained(root, device='cpu', dtype=torch.bfloat16, prefetch=False)
                self.assertEqual(stats['projections'], 6)
                keys = streamer._get_layer_keys('layers.0')
                raw = streamer.seeker.get_tensors(keys, dtype=torch.bfloat16)
                self.assertIn('layers.0.proj.q.qweight', raw)
                self.assertNotIn('layers.0.proj.q.weight', raw)
                self.assertFalse(any(k.endswith('.comfy_quant') for k in raw))
                outputs = []
                for _ in range(2):
                    streamer.apply_state_dict(raw)
                    linear = streamer.model.layers[0].proj['q']
                    self.assertIsInstance(linear, NativeInt8Linear)
                    self.assertEqual(linear.weight.dtype, torch.bfloat16)
                    outputs.append(linear(torch.ones(1, 2, 256, dtype=torch.bfloat16)))
                    evict_module(streamer.model.layers[0])
                    self.assertTrue(linear.qweight.is_meta and linear.weight_scale.is_meta)
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                self.assertEqual(stats['linear_calls'], 2)


if __name__ == '__main__':
    unittest.main()
