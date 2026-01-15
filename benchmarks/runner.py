"""
Benchmark runner for evaluating RLM and baseline methods.

Orchestrates running benchmarks with different inference methods:
- RLM (recursive language model)
- Direct LLM call
- Summarization-based
- RAG (retrieval-augmented)
- CodeAct (code-generation agents)
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from benchmarks.base import Benchmark, BenchmarkResult, BenchmarkSample, SampleResult


@dataclass
class RunnerConfig:
    """Configuration for benchmark runner."""

    backend: str = "openai"
    model: str = "gpt-5-mini"
    environment: str = "subprocess"
    max_iterations: int = 30
    verbose: bool = False
    log_dir: str | None = None
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    environment_kwargs: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Runs benchmarks with configurable inference methods.

    Supports multiple methods for comparison:
    - "rlm": Full RLM with REPL environment
    - "direct": Direct LLM call (context + question)
    - "summarize": Iterative summarization baseline
    - "custom": User-provided inference function

    Example:
        runner = BenchmarkRunner(backend="openai", model="gpt-5-mini")

        # Run with RLM
        rlm_results = runner.run(benchmark, method="rlm", num_samples=100)

        # Run with direct LLM for comparison
        direct_results = runner.run(benchmark, method="direct", num_samples=100)

        # Compare results
        print(f"RLM accuracy: {rlm_results.accuracy:.2%}")
        print(f"Direct accuracy: {direct_results.accuracy:.2%}")
    """

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-5-mini",
        environment: str = "subprocess",
        max_iterations: int = 30,
        verbose: bool = False,
        log_dir: str | None = None,
        **kwargs,
    ):
        """Initialize runner with configuration.

        Args:
            backend: LLM backend (openai, anthropic, etc.)
            model: Model name to use.
            environment: REPL environment for RLM (local, subprocess, docker, modal).
            max_iterations: Max iterations for RLM.
            verbose: Enable verbose output.
            log_dir: Directory for logging trajectories.
            **kwargs: Additional backend or environment kwargs.
        """
        self.config = RunnerConfig(
            backend=backend,
            model=model,
            environment=environment,
            max_iterations=max_iterations,
            verbose=verbose,
            log_dir=log_dir,
            backend_kwargs={"model_name": model, **kwargs.get("backend_kwargs", {})},
            environment_kwargs=kwargs.get("environment_kwargs", {}),
        )

    def run(
        self,
        benchmark: Benchmark,
        method: str = "rlm",
        num_samples: int | None = None,
        seed: int | None = None,
        custom_fn: Callable[[BenchmarkSample], str] | None = None,
    ) -> BenchmarkResult:
        """Run a benchmark with the specified method.

        Args:
            benchmark: Benchmark instance to run.
            method: Inference method ("rlm", "direct", "summarize", "custom").
            num_samples: Number of samples to evaluate. None for all.
            seed: Random seed for reproducible sampling.
            custom_fn: Custom inference function for method="custom".
                       Takes BenchmarkSample, returns prediction string.

        Returns:
            BenchmarkResult with all sample results and aggregate metrics.
        """
        result = BenchmarkResult(
            benchmark_name=benchmark.name,
            method=method,
            model=self.config.model,
            config={
                "backend": self.config.backend,
                "environment": self.config.environment,
                "max_iterations": self.config.max_iterations,
                "num_samples": num_samples,
                "seed": seed,
            },
        )

        inference_fn = self._get_inference_fn(method, custom_fn)

        for sample in benchmark.load_samples(num_samples=num_samples, seed=seed):
            sample_result = self._run_sample(sample, inference_fn, benchmark)
            result.sample_results.append(sample_result)

            if self.config.verbose:
                status = "✓" if sample_result.is_correct else "✗"
                print(f"  [{status}] Sample {sample.id}: {sample_result.metrics}")

        return result

    def _get_inference_fn(
        self,
        method: str,
        custom_fn: Callable[[BenchmarkSample], str] | None = None,
    ) -> Callable[[BenchmarkSample], tuple[str, dict[str, Any]]]:
        """Get inference function for the specified method.

        Returns:
            Function that takes BenchmarkSample and returns (prediction, metadata).
        """
        if method == "rlm":
            return self._inference_rlm
        elif method == "direct":
            return self._inference_direct
        elif method == "summarize":
            return self._inference_summarize
        elif method == "custom":
            if custom_fn is None:
                raise ValueError("custom_fn required for method='custom'")
            return lambda s: (custom_fn(s), {})
        else:
            raise ValueError(f"Unknown method: {method}")

    def _inference_rlm(self, sample: BenchmarkSample) -> tuple[str, dict[str, Any]]:
        """Run inference using RLM."""
        from rlm import RLM
        from rlm.logger import RLMLogger

        logger = None
        if self.config.log_dir:
            logger = RLMLogger(log_dir=self.config.log_dir)

        rlm = RLM(
            backend=self.config.backend,
            backend_kwargs=self.config.backend_kwargs,
            environment=self.config.environment,
            environment_kwargs=self.config.environment_kwargs,
            max_iterations=self.config.max_iterations,
            logger=logger,
            verbose=self.config.verbose,
        )

        result = rlm.completion(prompt=sample.context, root_prompt=sample.question)

        metadata = {
            "iterations": result.iterations if hasattr(result, "iterations") else 0,
            "total_tokens": (
                result.usage.total_tokens if hasattr(result, "usage") and result.usage else 0
            ),
        }

        return result.response, metadata

    def _inference_direct(self, sample: BenchmarkSample) -> tuple[str, dict[str, Any]]:
        """Run inference using direct LLM call."""
        from rlm.clients import get_client

        client = get_client(self.config.backend, **self.config.backend_kwargs)

        prompt = f"""Context:
{sample.context}

Question: {sample.question}

Answer the question based on the context above. Provide only the answer, nothing else."""

        response = client.completion(prompt)

        usage = client.get_last_usage()
        total_tokens = 0
        if usage and usage.model_usage_summaries:
            for model_usage in usage.model_usage_summaries.values():
                total_tokens += model_usage.total_input_tokens + model_usage.total_output_tokens

        return response, {"iterations": 1, "total_tokens": total_tokens}

    def _inference_summarize(self, sample: BenchmarkSample) -> tuple[str, dict[str, Any]]:
        """Run inference using iterative summarization.

        Splits context into chunks, summarizes each, then answers from summaries.
        """
        from rlm.clients import get_client

        client = get_client(self.config.backend, **self.config.backend_kwargs)

        # Chunk the context (simple split by paragraphs, ~4k chars each)
        chunk_size = 4000
        context = sample.context
        chunks = []

        while context:
            if len(context) <= chunk_size:
                chunks.append(context)
                break
            # Find a good break point
            break_point = context.rfind("\n\n", 0, chunk_size)
            if break_point == -1:
                break_point = context.rfind(". ", 0, chunk_size)
            if break_point == -1:
                break_point = chunk_size
            chunks.append(context[:break_point])
            context = context[break_point:].lstrip()

        # Summarize each chunk
        summaries = []
        iterations = 0
        for chunk in chunks:
            iterations += 1
            summary_prompt = f"""Summarize the following text, keeping all important facts and details relevant to answering questions:

{chunk}

Summary:"""
            summary = client.completion(summary_prompt)
            summaries.append(summary)

        # Combine summaries and answer
        iterations += 1
        combined = "\n\n".join(summaries)
        answer_prompt = f"""Based on these summaries:

{combined}

Question: {sample.question}

Answer:"""
        response = client.completion(answer_prompt)

        usage = client.get_usage_summary()
        total_tokens = 0
        if usage and usage.model_usage_summaries:
            for model_usage in usage.model_usage_summaries.values():
                total_tokens += model_usage.total_input_tokens + model_usage.total_output_tokens

        return response, {"iterations": iterations, "total_tokens": total_tokens}

    def _run_sample(
        self,
        sample: BenchmarkSample,
        inference_fn: Callable[[BenchmarkSample], tuple[str, dict[str, Any]]],
        benchmark: Benchmark,
    ) -> SampleResult:
        """Run a single sample and evaluate."""
        start_time = time.time()
        error = None
        prediction = ""
        metadata: dict[str, Any] = {}

        try:
            prediction, metadata = inference_fn(sample)
        except Exception as e:
            error = str(e)
            prediction = ""

        execution_time_ms = (time.time() - start_time) * 1000

        if error:
            metrics = {m: 0.0 for m in benchmark.default_metrics()}
            is_correct = False
        else:
            metrics = benchmark.evaluate(prediction, sample.expected_answer)
            is_correct = metrics.get("correct", 0.0) > 0.5

        return SampleResult(
            sample_id=sample.id,
            prediction=prediction,
            expected=sample.expected_answer,
            is_correct=is_correct,
            metrics=metrics,
            iterations=metadata.get("iterations", 0),
            total_tokens=metadata.get("total_tokens", 0),
            execution_time_ms=execution_time_ms,
            error=error,
        )


def compare_methods(
    benchmark: Benchmark,
    runner: BenchmarkRunner,
    methods: list[str] | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, BenchmarkResult]:
    """Run benchmark with multiple methods for comparison.

    Args:
        benchmark: Benchmark to run.
        runner: Configured BenchmarkRunner.
        methods: List of methods to compare. Default: ["rlm", "direct"].
        num_samples: Number of samples per method.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping method name to BenchmarkResult.
    """
    if methods is None:
        methods = ["rlm", "direct"]

    results = {}
    for method in methods:
        print(f"\nRunning {benchmark.name} with method={method}...")
        results[method] = runner.run(benchmark, method=method, num_samples=num_samples, seed=seed)
        print(f"  Accuracy: {results[method].accuracy:.2%}")
        print(f"  Mean F1: {results[method].mean_f1:.3f}")

    return results
