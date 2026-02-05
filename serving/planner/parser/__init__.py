"""Parser module for profiler data.

This module provides raw profiler data (components, pipelines, latency, metric)
that can be loaded into ProfileData for use by schedulers.
"""

from .profiler import components, pipelines, latency, metric

__all__ = ['components', 'pipelines', 'latency', 'metric']
