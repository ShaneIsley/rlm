#!/usr/bin/env python3
"""
CLI for running RLM benchmarks.

Usage:
    # Run benchmarks
    python -m benchmarks.cli run --benchmark oolong --num-samples 10
    python -m benchmarks.cli run --benchmark niah --methods rlm direct

    # Query stored results
    python -m benchmarks.cli history --benchmark oolong-toy_dnd --limit 10
    python -m benchmarks.cli compare --benchmark niah-100k --group-by method

    # Legacy (no subcommand defaults to 'run')
    python -m benchmarks.cli --benchmark oolong --num-samples 10
"""

import argparse
import json
import sys
from datetime import datetime

from benchmarks.base import BenchmarkResult
from benchmarks.results import ExperimentConfig, ResultsStore
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


def cmd_history(args: argparse.Namespace) -> int:
    """Show historical results for a benchmark."""
    store = ResultsStore(args.results_dir)
    records = store.get_history(
        benchmark=args.benchmark,
        model=args.model,
        method=args.method,
        limit=args.limit,
    )

    if not records:
        print(f"No results found for benchmark: {args.benchmark}")
        return 1

    print(f"\nHistory for {args.benchmark} ({len(records)} results)")
    print("=" * 80)

    headers = ["Timestamp", "Model", "Method", "Accuracy", "F1", "Tokens"]
    col_widths = [20, 20, 10, 10, 8, 10]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True))
    print(header_line)
    print("-" * len(header_line))

    for r in records:
        row = [
            r.timestamp[:19],
            r.config.model[:20],
            r.config.method[:10],
            f"{r.accuracy:.1%}",
            f"{r.mean_f1:.3f}",
            str(r.total_tokens),
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(row, col_widths, strict=True)))

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare results grouped by a dimension."""
    store = ResultsStore(args.results_dir)
    comparison = store.compare(
        benchmark=args.benchmark,
        group_by=args.group_by,
        filter_model=args.model,
    )

    if not comparison:
        print(f"No results found for benchmark: {args.benchmark}")
        return 1

    print(f"\nComparison for {args.benchmark} (grouped by {args.group_by})")
    print("=" * 70)

    headers = ["Group", "Count", "Accuracy", "Max Acc", "F1", "Tokens"]
    col_widths = [15, 8, 10, 10, 8, 12]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True))
    print(header_line)
    print("-" * len(header_line))

    for group, metrics in sorted(comparison.items()):
        row = [
            group[:15],
            str(metrics["count"]),
            f"{metrics['mean_accuracy']:.1%}",
            f"{metrics['max_accuracy']:.1%}",
            f"{metrics['mean_f1']:.3f}",
            f"{metrics['mean_tokens']:.0f}",
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(row, col_widths, strict=True)))

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available benchmarks and summary."""
    store = ResultsStore(args.results_dir)
    summary = store.summary()

    print("\nBenchmark Results Summary")
    print("=" * 50)
    print(f"Results directory: {summary['results_dir']}")
    print(f"Total experiments: {summary['total_experiments']}")
    print("\nBenchmarks:")

    for name, count in summary["benchmarks"].items():
        print(f"  - {name}: {count} runs")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export results to CSV."""
    store = ResultsStore(args.results_dir)
    store.export_csv(args.benchmark, args.output)
    print(f"Exported {args.benchmark} results to {args.output}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RLM Benchmarks CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./benchmark_results",
        help="Directory for storing results (default: ./benchmark_results)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========== RUN subcommand ==========
    run_parser = subparsers.add_parser("run", help="Run benchmarks")

    run_parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        required=True,
        choices=["niah", "oolong", "oolong-pairs", "browsecomp", "all"],
        help="Benchmark to run",
    )
    run_parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)",
    )
    run_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    run_parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=["rlm"],
        choices=["rlm", "direct", "summarize"],
        help="Inference methods to compare (default: rlm)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for results (JSON)",
    )
    run_parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="LLM backend (default: openai)",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name (default: gpt-5-mini)",
    )
    run_parser.add_argument(
        "--environment",
        type=str,
        default="subprocess",
        help="REPL environment (default: subprocess)",
    )
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Max RLM iterations (default: 30)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    run_parser.add_argument("--log-dir", type=str, default=None, help="Trajectory logs directory")
    run_parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=1,
        help="Parallel workers (1=sequential, >1=parallel threads, default: 1)",
    )
    run_parser.add_argument(
        "--progress",
        "-p",
        type=str,
        default="auto",
        choices=["auto", "tqdm", "simple", "none"],
        help="Progress display mode: auto (uses tqdm if available), tqdm (progress bar), "
        "simple (periodic status), none (quiet). Default: auto",
    )

    # Benchmark-specific options for run
    run_parser.add_argument("--context-length", type=int, default=100_000, help="NIAH context len")
    run_parser.add_argument("--needle-depth", type=float, default=None, help="NIAH needle position")
    run_parser.add_argument("--subset", type=str, default="toy_dnd", help="OOLONG subset")
    run_parser.add_argument("--num-items", type=int, default=50, help="OOLONG-Pairs items")
    run_parser.add_argument("--num-pairs", type=int, default=25, help="OOLONG-Pairs pairs")
    run_parser.add_argument("--num-documents", type=int, default=100, help="BrowseComp docs")
    run_parser.add_argument("--num-hops", type=int, default=2, help="BrowseComp hops")

    # ========== HISTORY subcommand ==========
    history_parser = subparsers.add_parser("history", help="Show historical results")
    history_parser.add_argument("--benchmark", "-b", type=str, required=True, help="Benchmark name")
    history_parser.add_argument("--model", type=str, default=None, help="Filter by model")
    history_parser.add_argument("--method", type=str, default=None, help="Filter by method")
    history_parser.add_argument("--limit", type=int, default=20, help="Max results to show")

    # ========== COMPARE subcommand ==========
    compare_parser = subparsers.add_parser("compare", help="Compare results")
    compare_parser.add_argument("--benchmark", "-b", type=str, required=True, help="Benchmark name")
    compare_parser.add_argument(
        "--group-by",
        type=str,
        default="method",
        choices=["method", "model", "environment"],
        help="Group by dimension",
    )
    compare_parser.add_argument("--model", type=str, default=None, help="Filter by model")

    # ========== LIST subcommand ==========
    subparsers.add_parser("list", help="List benchmarks and summary")

    # ========== EXPORT subcommand ==========
    export_parser = subparsers.add_parser("export", help="Export results to CSV")
    export_parser.add_argument("--benchmark", "-b", type=str, required=True, help="Benchmark name")
    export_parser.add_argument("--output", "-o", type=str, required=True, help="Output CSV path")

    args = parser.parse_args()

    # Handle legacy usage (no subcommand)
    if args.command is None:
        # Check if --benchmark was passed (legacy mode)
        if hasattr(args, "benchmark") and args.benchmark:
            args.command = "run"
        else:
            parser.print_help()
            return 1

    # Dispatch to appropriate command
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "history":
        return cmd_history(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "export":
        return cmd_export(args)
    else:
        parser.print_help()
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run benchmarks (main command)."""
    runner = BenchmarkRunner(
        backend=args.backend,
        model=args.model,
        environment=args.environment,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        log_dir=args.log_dir,
        max_workers=args.max_workers,
        progress=args.progress,
    )

    if args.benchmark == "all":
        benchmarks = get_all_benchmarks(args)
    else:
        benchmarks = [get_benchmark(args)]

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

    # Save to results store
    store = ResultsStore(args.results_dir)
    config = ExperimentConfig(
        backend=args.backend,
        model=args.model,
        environment=args.environment,
        max_iterations=args.max_iterations,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    for key, result in all_results.items():
        exp_id = store.save(result, config)
        print(f"Saved {key} as experiment {exp_id}")

    if args.output:
        save_results(all_results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
