"""Exercise range resumption and corrupted-response recovery against a real HTTP server."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio'))
import downloads

class DownloadTests(unittest.TestCase):
    def test_resume_and_retry_preserve_exact_bytes(self):
        payload = bytes(range(251)) * 400
        requests = []
        failed = [False]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                start, end = map(int, self.headers['Range'][6:].split('-'))
                requests.append((start, end))
                if start == 32768 and not failed[0]:
                    failed[0] = True
                    self.send_error(503)
                    return
                data = payload[start:end+1]
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(payload)}')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / 'weights.bin'
                parts = Path(folder) / 'weights.bin.parts-32768'
                parts.mkdir()
                (parts/'0').write_bytes(payload[:32768])
                (parts/'1').write_bytes(b'incomplete')
                with patch.object(downloads, 'CHUNK', 32768), patch.object(downloads.time, 'sleep'):
                    downloads.fetch(f'http://127.0.0.1:{server.server_port}/weights', target, workers=2)
                self.assertEqual(hashlib.sha256(target.read_bytes()).digest(), hashlib.sha256(payload).digest())
                self.assertNotIn((0,32767), requests)
                self.assertEqual(requests.count((32768,65535)), 2)
                self.assertFalse(parts.exists())
        finally:
            server.shutdown()
            server.server_close()

if __name__ == '__main__':
    unittest.main()
