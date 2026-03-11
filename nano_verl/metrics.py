"""Timing and throughput tracking for the nano_verl pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class StageMetric:
    """Aggregated measurements for one pipeline stage."""

    name: str
    duration_s: float = 0.0
    items: int = 0
    calls: int = 0

    @property
    def throughput(self) -> float:
        """Items processed per second for the stage."""

        if self.duration_s == 0:
            return 0.0
        return self.items / self.duration_s


class MetricsTracker:
    """Collect wall-clock timings and lightweight counters."""

    def __init__(self) -> None:
        self.started_at = perf_counter()
        self._stages: dict[str, StageMetric] = {}
        self._counters: dict[str, int] = {}

    @contextmanager
    def timed(self, stage_name: str, items: int = 0):
        """Measure a block of work and attach item counts to it."""

        start = perf_counter()
        try:
            yield
        finally:
            duration = perf_counter() - start
            metric = self._stages.setdefault(stage_name, StageMetric(name=stage_name))
            metric.duration_s += duration
            metric.items += items
            metric.calls += 1

    def increment(self, counter_name: str, value: int = 1) -> None:
        """Increment a named counter."""

        self._counters[counter_name] = self._counters.get(counter_name, 0) + value

    def stage_metrics(self) -> list[StageMetric]:
        """Return stage metrics in insertion order."""

        return list(self._stages.values())

    def counters(self) -> dict[str, int]:
        """Return a copy of all counters."""

        return dict(self._counters)

    def total_duration_s(self) -> float:
        """Total wall-clock duration since tracker creation."""

        return perf_counter() - self.started_at

