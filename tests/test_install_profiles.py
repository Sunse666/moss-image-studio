"""CPU validation of adaptive defaults for Lite (INT8-only) and full BF16 installs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio'))
import int8_config
import server

ROOT = Path(__file__).resolve().parents[1]


class InstallProfileTests(unittest.TestCase):
    def setUp(self):
        server.JOBS.clear(); server.ORDER.clear()
        self.addCleanup(server.JOBS.clear); self.addCleanup(server.ORDER.clear)

    def test_default_prefers_installed_int8_else_bf16(self):
        with patch.object(int8_config, 'available', return_value=True):
            self.assertEqual(server.engine.default_model(), int8_config.MODEL)
        with patch.object(int8_config, 'available', return_value=False):
            self.assertEqual(server.engine.default_model(), server.engine.SFT_MODEL)

    def test_bf16_available_follows_transformer_index(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'models' / 'MOSS-Image-SFT' / 'transformer'
            folder.mkdir(parents=True)
            (folder / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'a': 'shard-1.safetensors', 'b': 'shard-2.safetensors'}}))
            (folder / 'shard-1.safetensors').write_bytes(b'x')
            with patch.object(server.engine, 'ROOT', directory):
                self.assertFalse(server.engine.bf16_available())  # Lite: shard missing
                (folder / 'shard-2.safetensors').write_bytes(b'x')
                self.assertTrue(server.engine.bf16_available())
            with patch.object(server.engine, 'ROOT', str(Path(directory) / 'none')):
                self.assertFalse(server.engine.bf16_available())

    def test_lite_install_defaults_to_native_int8_and_rejects_bf16(self):
        with patch.object(int8_config, 'available', return_value=True), \
                patch.object(int8_config, 'native_available', return_value=True), \
                patch.object(server.engine, 'bf16_available', return_value=False), \
                patch.object(server.engine, 'turbo_available', return_value=False):
            server.submit({'prompt': 'CPU only'})
            job = next(iter(server.JOBS.values()))
            self.assertEqual((job['model'], job['int8_compute'], job['steps'], job['turbo']),
                             (int8_config.MODEL, 'native', 8, False))
            for model in (server.engine.SFT_MODEL, 'OpenMOSS-Team/MOSS-Image-Base'):
                with self.assertRaisesRegex(ValueError, 'not downloaded.*select INT8'):
                    server.submit({'prompt': 'CPU only', 'model': model})
            with self.assertRaisesRegex(ValueError, 'not downloaded'):
                server.submit({'prompt': 'CPU only', 'turbo': True})  # Turbo implies BF16 SFT

    def test_full_install_without_int8_keeps_bf16_turbo_default(self):
        with patch.object(int8_config, 'available', return_value=False), \
                patch.object(server.engine, 'bf16_available', return_value=True), \
                patch.object(server.engine, 'turbo_available', return_value=True):
            server.submit({'prompt': 'CPU only'})
            job = next(iter(server.JOBS.values()))
            self.assertEqual((job['model'], job['turbo'], job['steps']), (server.engine.SFT_MODEL, True, 4))

    def test_lite_download_skips_bf16_transformer_and_adds_int8(self):
        sys.path.insert(0, str(ROOT))
        import download
        with patch.object(sys, 'argv', ['download.py', 'Lite']), \
                patch.object(download, 'download_model') as model, \
                patch.object(download, 'download_int8') as int8, \
                patch.object(download, 'download_turbo') as turbo:
            download.main()
        model.assert_called_once_with('SFT', 32, 4, transformer=False)
        int8.assert_called_once_with(32)
        turbo.assert_not_called()
        self.assertIn("'Lite'", (ROOT / 'download.ps1').read_text())

    @staticmethod
    def fail_fetch():
        raise RuntimeError('stop after recording')  # no network, no files written

    def test_lite_file_filter_keeps_shared_components_only(self):
        sys.path.insert(0, str(ROOT))
        import download
        names = ['text_encoder/model-00001-of-00004.safetensors', 'vae/diffusion_pytorch_model.safetensors',
                 'transformer/diffusion_pytorch_model-00001-of-00003.safetensors', 'README.md']
        siblings = [type('F', (), {'rfilename': n, 'lfs': None})() for n in names]
        info = type('I', (), {'siblings': siblings, 'sha': 'abc'})()
        fetched = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(download, '__file__', str(Path(directory) / 'download.py')), \
                patch.object(download.HfApi, 'model_info', return_value=info), \
                patch.object(download, 'snapshot_download'), \
                patch.object(download, 'fetch', side_effect=lambda url, dest, workers: fetched.append(url) or self.fail_fetch()):
            with self.assertRaisesRegex(RuntimeError, 'stop after recording'):
                download.download_model('SFT', 1, 1, transformer=False)
        self.assertTrue(fetched)
        self.assertFalse(any('/transformer/' in url for url in fetched))

    def test_lite_compatibility_uses_original_index_and_weellm_shapes(self):
        import torch
        with patch('torch.utils._triton.has_triton', return_value=False):
            from weellm.models.transformers.moss_image_transformer_2d_model import MOSSImageTransformer
        model = MOSSImageTransformer(dim=256, ffn_dim=256, num_heads=2, num_layers=1,
                                     text_dim=8, freq_dim=8, latent_channels=4)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model.save_config(folder)
            names = {k: 'missing-shard.safetensors' for k in model.state_dict()}
            index = folder / 'diffusion_pytorch_model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': names}))
            with self.assertRaises(int8_config.OriginalHeadersMissing):
                int8_config.validate_checkpoint(folder / 'none.safetensors', folder)
            expected = int8_config.architecture_shapes(folder)
            self.assertEqual({k: v['shape'] for k, v in expected.items()},
                             {k: list(v.shape) for k, v in model.state_dict().items()})
            index.write_text(json.dumps({'weight_map': {**names, 'layers.0.proj.qkv.weight': 'x'}}))
            with self.assertRaisesRegex(ValueError, 'unknown to WeeLLM'):
                int8_config.architecture_shapes(folder)


if __name__ == '__main__':
    unittest.main()
