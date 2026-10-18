import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio'))
import server
from PIL import Image

class StudioTests(unittest.TestCase):
    def setUp(self):
        server.JOBS.clear()
        server.ORDER.clear()
        server.PRIVATE_MODE['on'] = False

    def test_moss_model_validation_and_limits(self):
        for mode in ('edit', 'assist', 'upscale'):
            with self.assertRaises(ValueError):
                server.submit({'mode': mode, 'prompt': 'test'})
        with self.assertRaises(ValueError):
            server.submit({'prompt': 'test', 'model': 'unknown'})
        result = server.submit({'prompt': 'test', 'width': 2048, 'height': 2048,
                                'model': 'OpenMOSS-Team/MOSS-Image-Base', 'batch': 2, 'seed': 4})
        jobs = list(server.JOBS.values())
        self.assertEqual([j['seed'] for j in jobs], [4, 5])
        self.assertTrue(all(j['width'] == 1024 and j['model'].endswith('Base') for j in jobs))

    def test_private_completion_has_no_sidecar_or_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(server, 'PRIVATE', directory):
                server.submit({'prompt': 'private scene', 'private': True})
                job = next(iter(server.JOBS.values()))
                def generate(job, output, log):
                    self.assertIsNone(log)
                    Image.new('RGB', (32, 32)).save(output)
                with patch.object(server.ENGINE, 'generate', side_effect=generate):
                    server.run_image_job(job)
                self.assertEqual(job['state'], 'done')
                self.assertEqual([p.suffix for p in Path(directory).iterdir()], ['.png'])
                self.assertEqual(job['meta']['model'], server.engine.DEFAULT_MODEL)

    def test_cancelled_worker_leaves_no_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(server, 'PRIVATE', directory):
                server.submit({'prompt': 'cancelled scene', 'private': True})
                job = next(iter(server.JOBS.values()))
                def cancelled(job, output, log):
                    Path(output).write_bytes(b'partial')
                    Path(output + '.part').write_bytes(b'partial')
                    job['state'] = 'cancelled'
                    raise server.Cancelled()
                with patch.object(server.ENGINE, 'generate', side_effect=cancelled):
                    with self.assertRaises(server.Cancelled):
                        server.run_image_job(job)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cancel_before_worker_launch_never_starts_process(self):
        job = {'state': 'running', '_cancel_requested': True}
        with patch.object(server.engine.subprocess, 'Popen') as spawn:
            with self.assertRaises(server.Cancelled):
                server.ENGINE.generate(job, 'unused.png', None)
            spawn.assert_not_called()

    def test_png_conversion_without_platform_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / 'source.jpg')
            output = str(Path(directory) / 'output.png')
            Image.new('RGB', (80, 40)).save(source)
            self.assertTrue(server.to_png(source, output, max_side=40))
            self.assertEqual(server.image_size(output), (40, 20))

if __name__ == '__main__':
    unittest.main()
