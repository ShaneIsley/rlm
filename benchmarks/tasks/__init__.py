"""
Benchmark task implementations.

Contains implementations for the four benchmark tasks from the RLM paper:
- S-NIAH: Single-Needle-in-a-Haystack
- BrowseComp-Plus: Multi-hop QA over document corpora
- OOLONG: Semantic aggregation
- OOLONG-Pairs: Pairwise combinatorial aggregation
"""

from benchmarks.tasks.browsecomp import BrowseCompPlusBenchmark
from benchmarks.tasks.niah import NIAHBenchmark
from benchmarks.tasks.oolong import OolongBenchmark, OolongPairsBenchmark

__all__ = [
    "NIAHBenchmark",
    "OolongBenchmark",
    "OolongPairsBenchmark",
    "BrowseCompPlusBenchmark",
]
