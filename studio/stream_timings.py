"""Opt-in aggregation of WeeLLM's per-block streaming timings (MOSS_STREAM_TIMINGS=1).

WeeLLM logs, at INFO level, each block's wait for its weights (disk read and/or
H2D future), H2D+apply time and synchronized compute time. This collector turns
those records into one compact per-pass summary instead of ~1,000 log lines.
"""
from contextlib import contextmanager
import logging
import sys


class StreamTimings(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.passes = []

    def _pass(self, position):
        if position == 1 or not self.passes:
            self.passes.append({'blocks': 0, 'wait_s': 0.0, 'h2d_apply_s': 0.0, 'compute_s': 0.0,
                                'max_wait_s': 0.0, 'sources': {}})
        return self.passes[-1]

    def emit(self, record):
        message, args = str(record.msg), record.args or ()
        try:
            if record.levelno >= logging.WARNING:
                sys.stderr.write(self.format(record) + '\n')  # keep warnings visible
            elif '] Block ' in message and len(args) >= 3:
                source = 'vram' if 'VRAM HIT' in message else str(args[3]) if len(args) > 3 else 'unknown'
                current = self._pass(int(args[1]))
                current['sources'][source] = current['sources'].get(source, 0) + 1
            elif 'Disk/Wait=' in message and len(args) == 5 and self.passes:
                current = self.passes[-1]
                current['blocks'] += 1
                current['wait_s'] += float(args[3])
                current['max_wait_s'] = max(current['max_wait_s'], float(args[3]))
                current['h2d_apply_s'] += float(args[4])
            elif 'GPU Compute=' in message and len(args) == 2 and self.passes:
                self.passes[-1]['compute_s'] += float(args[1])
        except (TypeError, ValueError):
            pass  # Diagnostics must never break generation.

    def summary(self, streamer=None):
        passes = [{k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()} for p in self.passes]
        result = {'passes': passes}
        steady = self.passes[1:]  # pass 1 includes calibration and cold reads
        if steady:
            blocks = sum(p['blocks'] for p in steady) or 1
            result['steady_per_block_ms'] = {
                key: round(sum(p[key] for p in steady) / blocks * 1000, 2)
                for key in ('wait_s', 'h2d_apply_s', 'compute_s')}
        if streamer is not None:
            result['streamer'] = {
                'pinned_blocks': len(getattr(streamer, '_pinned_blocks', ())),
                'double_buffering': bool(getattr(streamer, '_double_buffering_enabled', False)),
                'prefetch_depth': getattr(streamer, '_prefetch_depth', None),
                'unused_pin_budget_mb': round(getattr(streamer, '_cache_budget_bytes', 0) / 2**20)}
        return result


@contextmanager
def collect(enabled):
    if not enabled:
        yield None
        return
    logger = logging.getLogger('weellm')
    handler = StreamTimings()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    saved = logger.level, logger.propagate
    # INFO only reaches this handler; other handlers keep their old view.
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
