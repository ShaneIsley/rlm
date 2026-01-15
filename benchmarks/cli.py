#!/usr/bin/env python3
"""
CLI for running RLM benchmarks.

Usage:
    python -m benchmarks.cli --benchmark oolong --subset toy_dnd --num-samples 10
    python -m benchmarks.cli --benchmark niah --context-length 100000 --methods rlm direct
    python -m benchmarks.cli --benchmark all --num-samples 5 --output results.json
"""

import argparse
import json
import sys
from datetime import datetime

from benchmarks.base import BenchmarkResult
from benchmarks.runner import BenchmarkRunner, compare_methods
from benchmarks.tasks.browsecomp import BrowseCompPlusBenchmark
from benchmarks.tasks.niah import NIAHBenchmark
from benchmarks.tasks.oolong import OolongBenchmark, OolongPairsBenchmark


def get_benchmark(args: argparse.Namespace):
    """Instantiate benchmark from CLI arguments."""
    name = args.benchmark.lower()

    if name == "niah":
        return NIAHBenchmark(
            context_length=args.context_length,
            needle_depth=args.needle_depth,
        )
    elif name == "oolong":
        return OolongBenchmark(subset=args.subset)
    elif name == "oolong-pairs":
        return OolongPairsBenchmark(
            num_items=args.num_items,
            num_pairs=args.num_pairs,
        )
    elif name == "browsecomp":
        return BrowseCompPlusBenchmark(
            num_documents=args.num_documents,
            num_hops=args.num_hops,
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def get_all_benchmarks(args: argparse.Namespace):
    """Get all benchmarks for comprehensive evaluation."""
    return [
        NIAHBenchmark(context_length=args.context_length),
        OolongBenchmark(subset="toy_dnd"),
        OolongPairsBenchmark(num_items=30, num_pairs=15),
        BrowseCompPlusBenchmark(num_documents=50, num_hops=2),
    ]


def save_results(results: dict[str, BenchmarkResult], output_path: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": {
            name: {
                "summary": result.summary(),
                "sample_results": [
                    {
                        "id": sr.sample_id,
                        "prediction": sr.prediction,
                        "expected": sr.expected,
                        "is_correct": sr.is_correct,
                        "metrics": sr.metrics,
                        "iterations": sr.iterations,
                        "total_tokens": sr.total_tokens,
                        "execution_time_ms": sr.execution_time_ms,
                        "error": sr.error,
                    }
                    for sr in result.sample_results
                ],
            }
            for name, result in results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: dict[str, BenchmarkResult]):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    headers = ["Method", "Accuracy", "Mean F1", "Tokens", "Iterations", "Time (ms)"]
    col_widths = [15, 10, 10, 12, 12, 12]

    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True))
    print(header_line)
    print("-" * len(header_line))

    # Print results
    for name, result in results.items():
        row = [
            name[:15],
            f"{result.accuracy:.1%}",
            f"{result.mean_f1:.3f}",
            str(result.total_tokens),
            f"{result.mean_iterations:.1f}",
            f"{result.mean_execution_time_ms:.0f}",
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(row, col_widths, strict=True)))

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run RLM benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run OOLONG benchmark with RLM
  python -m benchmarks.cli --benchmark oolong --num-samples 10

  # Compare RLM vs direct LLM on NIAH
  python -m benchmarks.cli --benchmark niah --methods rlm direct --num-samples 20

  # Run all benchmarks and save results
  python -m benchmarks.cli --benchmark all --output results.json
        """,
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        required=True,
        choices=["niah", "oolong", "oolong-pairs", "browsecomp", "all"],
        help="Benchmark to run",
    )

    # Common options
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=["rlm"],
        choices=["rlm", "direct", "summarize"],
        help="Inference methods to compare (default: rlm)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for results (JSON)",
    )

    # Model configuration
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="LLM backend (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="subprocess",
        help="REPL environment (default: subprocess)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Max RLM iterations (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for trajectory logs",
    )

    # Benchmark-specific options
    parser.add_argument(
        "--context-length",
        type=int,
        default=100_000,
        help="NIAH: context length in chars (default: 100000)",
    )
    parser.add_argument(
        "--needle-depth",
        type=float,
        default=None,
        help="NIAH: needle position 0.0-1.0 (default: random)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="toy_dnd",
        help="OOLONG: dataset subset (default: toy_dnd)",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=50,
        help="OOLONG-Pairs: number of items (default: 50)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=25,
        help="OOLONG-Pairs: number of pairs (default: 25)",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=100,
        help="BrowseComp: number of documents (default: 100)",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=2,
        help="BrowseComp: reasoning hops (default: 2)",
    )

    args = parser.parse_args()

    # Create runner
    runner = BenchmarkRunner(
        backend=args.backend,
        model=args.model,
        environment=args.environment,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        log_dir=args.log_dir,
    )

    # Get benchmarks
    if args.benchmark == "all":
        benchmarks = get_all_benchmarks(args)
    else:
        benchmarks = [get_benchmark(args)]

    # Run benchmarks
    all_results = {}

    for benchmark in benchmarks:
        print(f"\n{'=' * 60}")
        print(f"Running: {benchmark.name}")
        print(f"Description: {benchmark.description}")
        print(f"{'=' * 60}")

        results = compare_methods(
            benchmark=benchmark,
            runner=runner,
            methods=args.methods,
            num_samples=args.num_samples,
            seed=args.seed,
        )

        for method, result in results.items():
            key = f"{benchmark.name}/{method}"
            all_results[key] = result

    # Print summary
    print_summary(all_results)

    # Save results if output specified
    if args.output:
        save_results(all_results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
