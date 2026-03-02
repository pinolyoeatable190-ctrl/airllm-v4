"""AirLLM v4 — Profiler with __slots__ for minimal overhead."""

import torch
from collections import defaultdict


class LayeredProfiler:
    """Lightweight profiler using defaultdict — no per-key init check."""
    __slots__ = ('_times', '_print_mem', '_min_free')

    def __init__(self, print_memory: bool = False):
        self._times = defaultdict(list)
        self._print_mem = print_memory
        self._min_free = float('inf')

    def add(self, item: str, elapsed: float):
        self._times[item].append(elapsed)
        if self._print_mem and torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0]
            self._min_free = min(self._min_free, free)

    def clear(self):
        self._times.clear()

    def print(self):
        for k, v in self._times.items():
            print(f"  {k}: {sum(v):.4f}s ({len(v)} calls)")
