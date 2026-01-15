"""
Base classes for RLM benchmarks.

Provides abstract interfaces that all benchmark implementations must follow,
enabling consistent evaluation and extensibility.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkSample:
    """A single benchmark sample with context, question, and expected answer."""

    id: str
    context: str
    question: str
    expected_answer: str | list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_tokens_approx(self) -> int:
        """Approximate token count (rough estimate: ~4 chars per token)."""
        return len(self.context) // 4


@dataclass
class SampleResult:
    """Result for a single benchmark sample."""

    sample_id: str
    prediction: str
    expected: str | list[str]
    is_correct: bool
    metrics: dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    total_tokens: int = 0
    execution_time_ms: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregated results for a complete benchmark run."""

    benchmark_name: str
    method: str  # "rlm", "direct", "summarize", "rag", etc.
    model: str
    sample_results: list[SampleResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Fraction of correct predictions."""
        if not self.sample_results:
            return 0.0
        correct = sum(1 for r in self.sample_results if r.is_correct)
        return correct / len(self.sample_results)

    @property
    def mean_f1(self) -> float:
        """Mean F1 score across samples."""
        f1_scores = [r.metrics.get("f1", 0.0) for r in self.sample_results]
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all samples."""
        return sum(r.total_tokens for r in self.sample_results)

    @property
    def mean_iterations(self) -> float:
        """Mean number of RLM iterations."""
        iters = [r.iterations for r in self.sample_results]
        return sum(iters) / len(iters) if iters else 0.0

    @property
    def mean_execution_time_ms(self) -> float:
        """Mean execution time per sample in milliseconds."""
        times = [r.execution_time_ms for r in self.sample_results]
        return sum(times) / len(times) if times else 0.0

    @property
    def error_rate(self) -> float:
        """Fraction of samples that resulted in errors."""
        errors = sum(1 for r in self.sample_results if r.error is not None)
        return errors / len(self.sample_results) if self.sample_results else 0.0

    def summary(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "benchmark": self.benchmark_name,
            "method": self.method,
            "model": self.model,
            "num_samples": len(self.sample_results),
            "accuracy": self.accuracy,
            "mean_f1": self.mean_f1,
            "total_tokens": self.total_tokens,
            "mean_iterations": self.mean_iterations,
            "mean_execution_time_ms": self.mean_execution_time_ms,
            "error_rate": self.error_rate,
        }


class Benchmark(ABC):
    """Abstract base class for all benchmarks.

    To create a new benchmark:
    1. Subclass Benchmark
    2. Implement name, description, load_samples(), and evaluate()
    3. Optionally override default_metrics() for custom evaluation

    Example:
        class MyBenchmark(Benchmark):
            @property
            def name(self) -> str:
                return "my-benchmark"

            def load_samples(self, num_samples: int | None = None) -> Iterator[BenchmarkSample]:
                # Load from dataset, file, or generate samples
                yield BenchmarkSample(...)

            def evaluate(self, prediction: str, expected: str | list[str]) -> dict[str, float]:
                # Return metrics dict with at least "correct" and "f1"
                return {"correct": 1.0 if prediction == expected else 0.0, "f1": ...}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this benchmark."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the benchmark."""
        return ""

    @abstractmethod
    def load_samples(
        self, num_samples: int | None = None, seed: int | None = None
    ) -> Iterator[BenchmarkSample]:
        """Load benchmark samples.

        Args:
            num_samples: Maximum number of samples to load. None for all.
            seed: Random seed for reproducible sampling.

        Yields:
            BenchmarkSample instances.
        """
        pass

    @abstractmethod
    def evaluate(self, prediction: str, expected: str | list[str]) -> dict[str, float]:
        """Evaluate a prediction against expected answer(s).

        Args:
            prediction: Model's prediction string.
            expected: Expected answer(s). Can be a single string or list of valid answers.

        Returns:
            Dictionary with at least:
                - "correct": 1.0 if correct, 0.0 otherwise
                - "f1": F1 score between prediction and expected
        """
        pass

    def default_metrics(self) -> list[str]:
        """List of metric names this benchmark produces."""
        return ["correct", "f1"]
