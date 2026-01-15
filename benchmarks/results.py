"""
Results storage and comparison for RLM benchmarks.

Provides persistent storage of benchmark results with:
- Structured experiment metadata (model, environment, config)
- Historical comparison across runs
- Statistical analysis utilities
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.base import BenchmarkResult


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment run."""

    # Model configuration
    backend: str
    model: str

    # Environment configuration
    environment: str
    max_iterations: int = 30

    # Benchmark configuration
    benchmark_name: str = ""
    method: str = "rlm"
    num_samples: int | None = None
    seed: int | None = None

    # Additional kwargs
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    environment_kwargs: dict[str, Any] = field(default_factory=dict)

    def config_hash(self) -> str:
        """Generate a hash of the configuration for deduplication."""
        config_str = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class ExperimentRecord:
    """A single experiment run with full metadata."""

    # Identifiers
    experiment_id: str
    timestamp: str

    # Configuration
    config: ExperimentConfig

    # Results summary
    accuracy: float
    mean_f1: float
    total_tokens: int
    mean_iterations: float
    mean_execution_time_ms: float
    error_rate: float
    num_samples: int

    # Optional: full sample results
    sample_results: list[dict[str, Any]] | None = None

    # Git/version info for reproducibility
    git_commit: str | None = None
    rlm_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "results": {
                "accuracy": self.accuracy,
                "mean_f1": self.mean_f1,
                "total_tokens": self.total_tokens,
                "mean_iterations": self.mean_iterations,
                "mean_execution_time_ms": self.mean_execution_time_ms,
                "error_rate": self.error_rate,
                "num_samples": self.num_samples,
            },
            "sample_results": self.sample_results,
            "metadata": {
                "git_commit": self.git_commit,
                "rlm_version": self.rlm_version,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        """Create from dictionary."""
        config = ExperimentConfig(**data["config"])
        results = data["results"]
        return cls(
            experiment_id=data["experiment_id"],
            timestamp=data["timestamp"],
            config=config,
            accuracy=results["accuracy"],
            mean_f1=results["mean_f1"],
            total_tokens=results["total_tokens"],
            mean_iterations=results["mean_iterations"],
            mean_execution_time_ms=results["mean_execution_time_ms"],
            error_rate=results["error_rate"],
            num_samples=results["num_samples"],
            sample_results=data.get("sample_results"),
            git_commit=data.get("metadata", {}).get("git_commit"),
            rlm_version=data.get("metadata", {}).get("rlm_version"),
        )


class ResultsStore:
    """Persistent storage for benchmark results.

    Stores results in JSON-lines format for efficient append and query.
    Each benchmark gets its own file for easy management.

    Directory structure:
        results_dir/
            niah-100k.jsonl
            oolong-toy_dnd.jsonl
            oolong-pairs-50x25.jsonl
            index.json  # Quick lookup index

    Usage:
        store = ResultsStore("./benchmark_results")

        # Save a result
        store.save(benchmark_result, config)

        # Query historical results
        history = store.get_history("oolong-toy_dnd", model="gpt-5")

        # Compare methods
        comparison = store.compare(
            benchmark="niah-100k",
            group_by="method",
            filter_model="gpt-5"
        )
    """

    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.results_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load or create the index file."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                self._index = json.load(f)
        else:
            self._index = {"benchmarks": {}, "experiments": []}

    def _save_index(self):
        """Save the index file."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.results_dir.parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def _get_rlm_version(self) -> str | None:
        """Get RLM package version."""
        try:
            import importlib.metadata

            return importlib.metadata.version("rlm")
        except Exception:
            return None

    def save(
        self,
        result: BenchmarkResult,
        config: ExperimentConfig | None = None,
        include_samples: bool = True,
    ) -> str:
        """Save a benchmark result.

        Args:
            result: BenchmarkResult to save.
            config: ExperimentConfig with full configuration.
                    If None, creates from result.config.
            include_samples: Whether to store individual sample results.

        Returns:
            Experiment ID for reference.
        """
        # Create config if not provided
        if config is None:
            config = ExperimentConfig(
                backend=result.config.get("backend", "unknown"),
                model=result.model,
                environment=result.config.get("environment", "unknown"),
                max_iterations=result.config.get("max_iterations", 30),
                benchmark_name=result.benchmark_name,
                method=result.method,
                num_samples=result.config.get("num_samples"),
                seed=result.config.get("seed"),
            )
        else:
            config.benchmark_name = result.benchmark_name
            config.method = result.method

        # Generate experiment ID
        timestamp = datetime.now().isoformat()
        exp_id = f"{config.config_hash()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create record
        sample_results = None
        if include_samples:
            sample_results = [
                {
                    "sample_id": sr.sample_id,
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
            ]

        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=timestamp,
            config=config,
            accuracy=result.accuracy,
            mean_f1=result.mean_f1,
            total_tokens=result.total_tokens,
            mean_iterations=result.mean_iterations,
            mean_execution_time_ms=result.mean_execution_time_ms,
            error_rate=result.error_rate,
            num_samples=len(result.sample_results),
            sample_results=sample_results,
            git_commit=self._get_git_commit(),
            rlm_version=self._get_rlm_version(),
        )

        # Write to benchmark-specific file
        benchmark_file = self.results_dir / f"{result.benchmark_name}.jsonl"
        with open(benchmark_file, "a") as f:
            json.dump(record.to_dict(), f)
            f.write("\n")

        # Update index
        if result.benchmark_name not in self._index["benchmarks"]:
            self._index["benchmarks"][result.benchmark_name] = {
                "file": f"{result.benchmark_name}.jsonl",
                "count": 0,
            }
        self._index["benchmarks"][result.benchmark_name]["count"] += 1
        self._index["experiments"].append(
            {
                "id": exp_id,
                "benchmark": result.benchmark_name,
                "model": config.model,
                "method": config.method,
                "accuracy": result.accuracy,
                "timestamp": timestamp,
            }
        )
        self._save_index()

        return exp_id

    def get_history(
        self,
        benchmark: str,
        model: str | None = None,
        method: str | None = None,
        limit: int | None = None,
    ) -> list[ExperimentRecord]:
        """Get historical results for a benchmark.

        Args:
            benchmark: Benchmark name to query.
            model: Filter by model name (substring match).
            method: Filter by method (rlm, direct, etc.).
            limit: Maximum number of results to return.

        Returns:
            List of ExperimentRecord objects, newest first.
        """
        benchmark_file = self.results_dir / f"{benchmark}.jsonl"
        if not benchmark_file.exists():
            return []

        records = []
        with open(benchmark_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    record = ExperimentRecord.from_dict(data)

                    # Apply filters
                    if model and model.lower() not in record.config.model.lower():
                        continue
                    if method and record.config.method != method:
                        continue

                    records.append(record)

        # Sort by timestamp descending
        records.sort(key=lambda r: r.timestamp, reverse=True)

        if limit:
            records = records[:limit]

        return records

    def compare(
        self,
        benchmark: str,
        group_by: str = "method",
        filter_model: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare results grouped by a dimension.

        Args:
            benchmark: Benchmark to compare.
            group_by: Dimension to group by ("method", "model", "environment").
            filter_model: Optional model filter.

        Returns:
            Dictionary mapping group key to aggregated metrics.
        """
        records = self.get_history(benchmark, model=filter_model)

        groups: dict[str, list[ExperimentRecord]] = {}
        for record in records:
            if group_by == "method":
                key = record.config.method
            elif group_by == "model":
                key = record.config.model
            elif group_by == "environment":
                key = record.config.environment
            else:
                key = getattr(record.config, group_by, "unknown")

            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        # Aggregate metrics for each group
        comparison = {}
        for key, group_records in groups.items():
            comparison[key] = {
                "count": len(group_records),
                "mean_accuracy": sum(r.accuracy for r in group_records) / len(group_records),
                "max_accuracy": max(r.accuracy for r in group_records),
                "mean_f1": sum(r.mean_f1 for r in group_records) / len(group_records),
                "mean_tokens": sum(r.total_tokens for r in group_records) / len(group_records),
                "mean_iterations": sum(r.mean_iterations for r in group_records)
                / len(group_records),
            }

        return comparison

    def list_benchmarks(self) -> list[str]:
        """List all benchmarks with stored results."""
        return list(self._index["benchmarks"].keys())

    def summary(self) -> dict[str, Any]:
        """Get summary of all stored results."""
        return {
            "total_experiments": len(self._index["experiments"]),
            "benchmarks": {name: info["count"] for name, info in self._index["benchmarks"].items()},
            "results_dir": str(self.results_dir),
        }

    def export_csv(self, benchmark: str, output_path: str):
        """Export benchmark results to CSV for external analysis."""
        import csv

        records = self.get_history(benchmark)

        if not records:
            return

        fieldnames = [
            "experiment_id",
            "timestamp",
            "model",
            "method",
            "environment",
            "accuracy",
            "mean_f1",
            "total_tokens",
            "mean_iterations",
            "num_samples",
            "git_commit",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(
                    {
                        "experiment_id": r.experiment_id,
                        "timestamp": r.timestamp,
                        "model": r.config.model,
                        "method": r.config.method,
                        "environment": r.config.environment,
                        "accuracy": r.accuracy,
                        "mean_f1": r.mean_f1,
                        "total_tokens": r.total_tokens,
                        "mean_iterations": r.mean_iterations,
                        "num_samples": r.num_samples,
                        "git_commit": r.git_commit,
                    }
                )
