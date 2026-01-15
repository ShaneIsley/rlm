"""
RLM Benchmarks Framework

A minimal, extensible framework for evaluating Recursive Language Models
against the benchmarks from the original RLM paper (arXiv:2512.24601).

Benchmark tasks:
- S-NIAH: Single-Needle-in-a-Haystack (context-insensitive retrieval)
- BrowseComp-Plus: Compositional multi-hop QA over document corpora
- OOLONG: Semantic aggregation over long contexts
- OOLONG-Pairs: Pairwise combinatorial aggregation

Usage:
    from benchmarks import BenchmarkRunner, OolongBenchmark

    runner = BenchmarkRunner(backend="openai", model="gpt-5-mini")
    results = runner.run(OolongBenchmark(subset="toy_dnd", num_samples=10))
"""

from benchmarks.base import Benchmark, BenchmarkResult, BenchmarkSample
from benchmarks.metrics import Metrics
from benchmarks.runner import BenchmarkRunner, compare_methods
from benchmarks.tasks import (
    BrowseCompPlusBenchmark,
    NIAHBenchmark,
    OolongBenchmark,
    OolongPairsBenchmark,
)

__all__ = [
    # Base classes
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkSample",
    # Runner
    "BenchmarkRunner",
    "compare_methods",
    # Metrics
    "Metrics",
    # Benchmark tasks
    "NIAHBenchmark",
    "OolongBenchmark",
    "OolongPairsBenchmark",
    "BrowseCompPlusBenchmark",
]
