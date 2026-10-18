"""Wall-clock stage timings, optionally synchronized with the execution device."""
import time
from contextlib import contextmanager


class StageTimings:
    def __init__(self):
        self.seconds = {}
        self.synchronize = None
        self.started = time.perf_counter()

    @contextmanager
    def stage(self, name):
        if self.synchronize:
            self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.synchronize:
                self.synchronize()
            self.seconds[name] = round(self.seconds.get(name, 0) + time.perf_counter() - started, 4)

    def finish(self):
        if self.synchronize:
            self.synchronize()
        return {**self.seconds, "worker_total": round(time.perf_counter() - self.started, 4)}
