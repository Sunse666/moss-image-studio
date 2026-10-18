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
import native_int8
from native_int8 import Int8Kernels, NativeInt8Linear, native_transformer, read_raw
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
        def quantize(x):
            self.assertEqual(x.dtype, torch.bfloat16)
            return x.reshape(-1, x.shape[-1]), torch.ones(x.numel() // x.shape[-1], 1)
        def matmul(x, x_scale, weight, scale):
            self.assertEqual(weight.dtype, torch.int8)
            self.assertEqual(scale.dtype, torch.float32)
            return torch.nn.functional.linear(x.float() * x_scale, weight.float() * scale).to(torch.bfloat16)
        kernel = Int8Kernels(quantize, matmul)
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
                proj = streamer.model.layers[0].proj
                self.assertIs(proj['q'].shared, proj['v'].shared)
                self.assertIsNone(proj['o'].shared)
                self.assertEqual((proj['q'].last, proj['k'].last, proj['v'].last), (False, False, True))
                keys = streamer._get_layer_keys('layers.0')
                raw = streamer.seeker.get_tensors(keys, dtype=torch.bfloat16)
                self.assertIn('layers.0.proj.q.qweight', raw)
                self.assertNotIn('layers.0.proj.q.weight', raw)
                # Raw reader returns the exact checkpoint bytes for every dtype.
                for name, value in raw.items():
                    source = weights[name.replace('.qweight', '.weight')]
                    self.assertEqual(value.dtype, source.dtype)
                    torch.testing.assert_close(value, source, rtol=0, atol=0)
                self.assertGreater(stats['pageable_reads'], 0)
                self.assertEqual(stats['pinned_reads'], 0)  # no CUDA: never pins
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
                # A real MOSS block passes one normalized tensor to q, k and v.
                streamer.apply_state_dict(raw)
                rope = streamer.model.rope(2, 2, 1, 'cpu').to(torch.bfloat16)
                before = dict(stats)
                block = streamer.model.layers[0]  # forward() bypasses CUDA-syncing stream hooks
                block.forward(torch.randn(1, 5, 256, dtype=torch.bfloat16), rope,
                              torch.zeros(1, 6, 256, dtype=torch.bfloat16))
                self.assertEqual(stats['linear_calls'] - before['linear_calls'], 6)
                self.assertEqual(stats['activation_reuses'] - before['activation_reuses'], 2)
                self.assertEqual(stats['activation_quantizations'] - before['activation_quantizations'], 4)
                self.assertEqual(streamer.model.layers[0].proj['q'].shared, {})


    def test_shared_qkv_input_quantized_once_and_identical(self):
        calls = []
        def quantize(x):
            calls.append(x)
            scale = x.float().abs().amax(-1, keepdim=True).reshape(-1, 1) / 127
            return torch.round(x.float().reshape(-1, x.shape[-1]) / scale).to(torch.int8), scale
        def matmul(x, x_scale, weight, scale):
            return ((x.float() @ weight.float().T) * x_scale * scale.reshape(1, -1)).to(torch.bfloat16)
        kernel = Int8Kernels(quantize, matmul)
        stats = {'linear_calls': 0, 'activation_quantizations': 0, 'activation_reuses': 0}
        shared = {}
        torch.manual_seed(0)
        layers = [NativeInt8Linear(16, 8, kernel, stats, shared=shared, last=name == 'v') for name in 'qkv']
        alone = NativeInt8Linear(16, 8, kernel, stats)
        for layer in layers:
            layer.qweight = torch.nn.Parameter(torch.randint(-127, 128, (8, 16), dtype=torch.int8), requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(torch.rand(8, 1), requires_grad=False)
        y = torch.randn(1, 5, 16, dtype=torch.bfloat16)
        outputs = [layer(y) for layer in layers]
        self.assertEqual(len(calls), 1)
        self.assertEqual((stats['activation_quantizations'], stats['activation_reuses']), (1, 2))
        self.assertEqual(shared, {})
        for layer, output in zip(layers, outputs):
            alone.qweight, alone.weight_scale = layer.qweight, layer.weight_scale
            torch.testing.assert_close(output, alone(y), rtol=0, atol=0)
            self.assertEqual(output.shape, (1, 5, 8))
        layers[0](y); y.mul_(2); layers[1](y)  # in-place change invalidates reuse
        self.assertEqual(stats['activation_quantizations'], 6)
        layers[2](y)
        self.assertEqual(shared, {})

    def test_prefetch_depth_override_is_scoped(self):
        from weellm.models.transformers import moss_image_transformer_2d_model as module
        streamer = module.MOSSImageTransformerStreamer
        original = streamer._compute_prefetch_depth
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            save_file({'a': torch.zeros(1)}, str(path))
            with native_transformer(path, {'projections': 0}, kernel=Int8Kernels(None, None), prefetch_depth=9):
                self.assertEqual(streamer._compute_prefetch_depth(object()), 6)
        self.assertIs(streamer._compute_prefetch_depth, original)


    def test_raw_reader_pins_cpu_staging_and_falls_back(self):
        tensors = {'a': torch.arange(-5, 5, dtype=torch.int8).reshape(2, 5),
                   'b': torch.randn(3, 1, dtype=torch.float32),
                   'c': torch.randn(7, dtype=torch.bfloat16)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            save_file(tensors, str(path))
            header, base = int8_config.read_header(path)
            requests = []
            def allocate(nbytes, pin):
                requests.append(pin)
                if pin and len(requests) == 1:
                    raise RuntimeError('pinned allocation refused')
                return torch.empty(nbytes, dtype=torch.uint8)
            with patch.object(native_int8, '_host_buffer', allocate):
                values, pinned = read_raw(path, header, base, ['c', 'a', 'b'], True)
            self.assertEqual((requests, pinned), ([True, False], False))
            for key, value in values.items():
                torch.testing.assert_close(value, tensors[key], rtol=0, atol=0)
                self.assertEqual(value.storage_offset() * value.element_size() % 64, 0)
            with patch.object(native_int8, '_host_buffer', lambda n, pin: torch.empty(n, dtype=torch.uint8)):
                _, pinned = read_raw(path, header, base, ['a'], True)
            self.assertTrue(pinned)
            with self.assertRaises(KeyError):
                read_raw(path, header, base, ['missing'], False)
            with patch('builtins.open', side_effect=lambda *a, **k: __import__('io').BytesIO(b'\x00' * (base + 3))):
                with self.assertRaisesRegex(ValueError, 'Short read'):
                    read_raw(path, header, base, ['a'], False)

    def test_seeker_pins_only_cpu_reads_with_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            save_file({'layers.0.proj.q.weight': torch.ones(4, 4, dtype=torch.int8),
                       'layers.0.proj.q.weight_scale': torch.ones(4, 1)}, str(path))
            seen = []
            def fake_read(path_, header, base, keys, pin):
                seen.append(pin)
                return read_raw(path_, header, base, keys, False)[0], pin
            stats = {'projections': 0}
            with patch.object(native_int8, 'read_raw', fake_read),                     patch.object(torch.cuda, 'is_available', return_value=True),                     native_transformer(path, stats, kernel=Int8Kernels(None, None), pinned=True):
                from weellm.models.transformers import moss_image_transformer_2d_model as module
                seeker = module.get_seeker(path)
                result = seeker.get_tensors(['layers.0.proj.q.weight'], 'cpu', torch.bfloat16, False)
                self.assertEqual(set(result), {'layers.0.proj.q.qweight', 'layers.0.proj.q.weight_scale'})
                self.assertEqual(result['layers.0.proj.q.weight_scale'].dtype, torch.float32)
                with patch.object(torch.Tensor, 'to', lambda self, *a, **k: self):
                    seeker.get_tensors(['layers.0.proj.q.weight'], 'cuda', torch.bfloat16)
            self.assertEqual(seen, [True, False])
            self.assertEqual((stats['pinned_reads'], stats['pageable_reads']), (1, 1))
            with native_transformer(path, {'projections': 0}, kernel=Int8Kernels(None, None), pinned=False),                     patch.object(native_int8, 'read_raw', fake_read),                     patch.object(torch.cuda, 'is_available', return_value=True):
                module.get_seeker(path).get_tensors(['layers.0.proj.q.weight'], 'cpu')
            self.assertEqual(seen[-1], False)


if __name__ == '__main__':
    unittest.main()
