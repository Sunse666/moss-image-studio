"""CPU validation of the opt-in WeeLLM per-block timing collector."""
import importlib.util
import logging
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio'))
from stream_timings import collect

HIT = "    [Streamer] Block %s (%d/%d) [VRAM HIT — skipping disk]"
LOAD = "    [Streamer] Block %s (%d/%d) [loading from %s]"
WAIT = "    [Streamer] %s (%d/%d): Disk/Wait=%.3fs | H2D+Apply=%.3fs"
COMPUTE = "    [Streamer] %s: GPU Compute=%.3fs (Offloading...)"


class StreamTimingTests(unittest.TestCase):
    def test_templates_match_installed_weellm(self):
        spec = importlib.util.find_spec('weellm')
        source = (Path(spec.origin).parent / 'models' / 'transformers' / 'base_transformer_streamer.py').read_text(encoding='utf-8')
        for template in (HIT, LOAD, WAIT, COMPUTE):
            self.assertIn(template.strip(), source)

    def test_aggregates_passes_and_restores_logger(self):
        logger = logging.getLogger('weellm')
        saved = logger.level, logger.propagate, list(logger.handlers)
        with collect(True) as log:
            for _ in range(2):
                for position, source in ((1, 'prefetch'), (2, None)):
                    name = f'layers.{position - 1}'
                    if source:
                        logger.info(LOAD, name, position, 2, source)
                    else:
                        logger.info(HIT, name, position, 2)
                    logger.info(WAIT, name, position, 2, 0.25 if source else 0.0, 0.5)
                    logger.info(COMPUTE, name, 0.1)
        self.assertEqual((logger.level, logger.propagate, logger.handlers), (saved[0], saved[1], saved[2]))
        summary = log.summary(type('S', (), {'_pinned_blocks': {'layers.1'}, '_double_buffering_enabled': True,
                                             '_prefetch_depth': 3, '_cache_budget_bytes': 2**20})())
        self.assertEqual(len(summary['passes']), 2)
        first = summary['passes'][0]
        self.assertEqual((first['blocks'], first['sources']), (2, {'prefetch': 1, 'vram': 1}))
        self.assertAlmostEqual(first['wait_s'], 0.25)
        self.assertAlmostEqual(first['compute_s'], 0.2)
        self.assertEqual(summary['steady_per_block_ms'], {'wait_s': 125.0, 'h2d_apply_s': 500.0, 'compute_s': 100.0})
        self.assertEqual(summary['streamer'], {'pinned_blocks': 1, 'double_buffering': True,
                                               'prefetch_depth': 3, 'unused_pin_budget_mb': 1})

    def test_disabled_is_noop(self):
        logger = logging.getLogger('weellm')
        level = logger.level
        with collect(False) as log:
            self.assertIsNone(log)
            self.assertEqual(logger.level, level)


if __name__ == '__main__':
    unittest.main()
