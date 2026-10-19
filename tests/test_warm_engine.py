"""CPU validation of warm worker reuse, queue clearing and private-session exit."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio'))
import int8_config
import server
import moss_worker

# Speaks the worker protocol without torch: one image per spec line.
FAKE_WORKER = r'''
import json, os, sys
while True:
    line = sys.stdin.readline()
    if not line:
        break
    spec = json.loads(line)
    if spec["prompt"] == "fail":
        raise SystemExit("synthetic failure")
    open(spec["output"], "wb").write(b"png")
    print("MOSS_EVENT " + json.dumps({"worker_pid": os.getpid(), "stage": "Done"}), flush=True)
    if not spec.get("keep_warm"):
        break
    print("MOSS_EVENT " + json.dumps({"job_done": True}), flush=True)
'''


def job(prompt='scene', model=int8_config.MODEL, compute='native'):
    return {'model': model, 'prompt': prompt, 'negative': '', 'width': 64, 'height': 64, 'steps': 8,
            'cfg': 1, 'seed': 1, 'turbo': False, 'private': False, 'int8_compute': compute, 'state': 'running'}


class WarmEngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        script = Path(self.directory.name) / 'fake_worker.py'
        script.write_text(FAKE_WORKER)
        self.engine = server.engine.Engine()
        self.engine.worker_script = str(script)
        self.addCleanup(self.engine.stop)

    def run_job(self, spec):
        out = os.path.join(self.directory.name, spec['prompt'] + str(time.monotonic_ns()) + '.png')
        self.engine.generate(spec, out, None)
        return spec

    def test_native_int8_reuses_warm_worker_then_reaps(self):
        self.engine.warm_seconds = 60
        first = self.run_job(job('a'))
        self.assertEqual(first['engine'], 'cold')
        self.assertEqual(self.engine.status()['state'], 'warm')
        self.assertTrue(self.engine.will_reuse(job('b')))
        second = self.run_job(job('b'))
        self.assertEqual((second['engine'], second['worker_pid']), ('warm', first['worker_pid']))
        self.engine.warm_seconds = 0  # reaper stops an idle warm worker once its time is up
        deadline = time.time() + 10
        while self.engine.running and time.time() < deadline:
            time.sleep(0.2)
        self.assertEqual(self.engine.status()['state'], 'cold')

    def test_other_profiles_and_off_setting_stay_one_shot(self):
        self.engine.warm_seconds = 60
        compat = self.run_job(job('c', compute='bf16'))
        self.assertFalse(self.engine.running)
        self.assertEqual(compat['engine'], 'cold')
        self.engine.warm_seconds = 0
        self.run_job(job('d'))
        self.assertFalse(self.engine.running)

    def test_profile_change_and_failure_restart_cold(self):
        self.engine.warm_seconds = 60
        first = self.run_job(job('e'))
        self.assertFalse(self.engine.will_reuse(job('f', compute='bf16')))
        with self.assertRaisesRegex(RuntimeError, 'MOSS worker failed'):
            self.run_job(job('fail'))  # same profile reuses, then the error drops the worker
        self.assertFalse(self.engine.running)
        after = self.run_job(job('g'))
        self.assertEqual(after['engine'], 'cold')
        self.assertNotEqual(after['worker_pid'], first['worker_pid'])

    def test_settings_validate_warm_minutes(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(server, 'SETTINGS', os.path.join(directory, 'settings.json')):
            self.assertEqual(server.load_settings()['warm_minutes'], 5)
            server.write_json(server.SETTINGS, {'warm_minutes': 7})
            self.assertEqual(server.load_settings()['warm_minutes'], 5)
            server.write_json(server.SETTINGS, {'warm_minutes': 0})
            with patch.object(server, 'ENGINE', self.engine):
                server.apply_settings(server.load_settings())
            self.assertEqual(self.engine.warm_seconds, 0)


class QueueAndPrivateTests(unittest.TestCase):
    def setUp(self):
        server.JOBS.clear(); server.ORDER.clear()
        self.addCleanup(server.JOBS.clear); self.addCleanup(server.ORDER.clear)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.private = os.path.join(self.directory.name, 'private')
        os.makedirs(self.private)
        patcher = patch.object(server, 'PRIVATE', self.private)
        patcher.start(); self.addCleanup(patcher.stop)
        server.PRIVATE_MODE['on'] = False
        self.addCleanup(server.PRIVATE_MODE.update, on=False)

    def add(self, state, private, output=None):
        new = server.new_job('gen', private, state=state, output=output)
        server.JOBS[new['id']] = new; server.ORDER.append(new['id'])
        if output:
            Path(self.private, output).write_bytes(b'png')
        return new

    def test_clear_finished_hides_private_but_keeps_private_tray(self):
        public_done = self.add('done', False)
        private_done = self.add('done', True, 'p.png')
        running = self.add('running', True)
        server.clear_finished()
        visible = [server.JOBS[i]['id'] for i in server.ORDER if not server.JOBS[i].get('queue_hidden')]
        self.assertEqual(visible, [running['id']])
        self.assertNotIn(public_done['id'], server.JOBS)
        self.assertEqual([item['name'] for item in server.private_items()], ['p.png'])
        self.assertIn(private_done['id'], server.JOBS)

    def test_leaving_private_mode_erases_session_and_cancels_pending(self):
        self.add('done', True, 'p.png')
        queued = self.add('queued', True)
        kept = self.add('done', False)
        with patch.object(server, 'ENGINE'):
            server.end_private_session()
        self.assertEqual(os.listdir(self.private), [])
        self.assertEqual(list(server.JOBS), [kept['id']])
        self.assertEqual(queued['state'], 'cancelled')
        self.assertEqual(server.private_items(), [])

    def test_wipe_private_tolerates_leftover_directory(self):
        with patch.object(server.shutil, 'rmtree'):  # simulate a locked file surviving rmtree
            server.wipe_private()
        self.assertTrue(os.path.isdir(self.private))


class WorkerResetTests(unittest.TestCase):
    def test_reset_for_next_job_recalibrates_streamer(self):
        import threading
        from concurrent.futures import Future
        future = Future()
        streamer = type('Streamer', (), {})()
        streamer._lock = threading.Lock()
        streamer._h2d_futures, streamer._disk_futures = {'layers.0': future}, {}
        streamer._pinned_blocks, streamer._cache_budget_bytes, streamer._calibration_done = {'layers.1'}, 5, True
        transformer = type('T', (), {})()
        transformer._weellm_streamer, transformer._weellm_te_evicted = streamer, True
        pipe = type('P', (), {})(); pipe._pipeline = type('Q', (), {})(); pipe._pipeline.transformer = transformer
        stats = {'linear_calls': 9, 'projections': 252, 'backend': 'x', 'pinned_reads': 3}
        moss_worker.reset_for_next_job(pipe, stats)
        self.assertEqual((streamer._pinned_blocks, streamer._cache_budget_bytes, streamer._calibration_done),
                         (set(), 0, False))
        self.assertEqual(streamer._h2d_futures, {})
        self.assertTrue(future.cancelled())
        self.assertTrue(transformer._weellm_te_evicted)  # warm workers keep the text encoder loaded
        self.assertEqual(stats, {'linear_calls': 0, 'projections': 252, 'backend': 'x', 'pinned_reads': 0})

    def test_profile_and_job_checks(self):
        spec = {'model': int8_config.MODEL, 'int8_compute': 'native', 'turbo': False,
                'steps': 8, 'cfg': 1, 'negative': ''}
        self.assertEqual(moss_worker.profile(spec), (int8_config.MODEL, 'native', False))
        moss_worker.check_job(spec)
        with self.assertRaisesRegex(ValueError, 'eight-step'):
            moss_worker.check_job({**spec, 'steps': 4})


if __name__ == '__main__':
    unittest.main()
