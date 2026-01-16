"""
Benchmark runner for evaluating RLM and baseline methods.

Orchestrates running benchmarks with different inference methods:
- RLM (recursive language model)
- Direct LLM call
- Summarization-based
- RAG (retrieval-augmented)
- CodeAct (code-generation agents)

Supports parallel execution for faster evaluation.
Includes progress tracking with ETA via tqdm or custom callbacks.
"""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from benchmarks.base import Benchmark, BenchmarkResult, BenchmarkSample, SampleResult

# Type alias for progress callback
ProgressCallback = Callable[[int, int, "SampleResult | None", "ProgressStats"], None]


@dataclass
class ProgressStats:
    """Running statistics for progress reporting."""

    completed: int = 0
    total: int = 0
    correct: int = 0
    errors: int = 0
    total_time_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        """Current running accuracy."""
        if self.completed == 0:
            return 0.0
        return self.correct / self.completed

    @property
    def error_rate(self) -> float:
        """Current error rate."""
        if self.completed == 0:
            return 0.0
        return self.errors / self.completed

    @property
    def avg_time_ms(self) -> float:
        """Average time per sample in milliseconds."""
        if self.completed == 0:
            return 0.0
        return self.total_time_ms / self.completed

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        remaining = self.total - self.completed
        if remaining <= 0 or self.avg_time_ms == 0:
            return 0.0
        return (remaining * self.avg_time_ms) / 1000.0


@dataclass
class RunnerConfig:
    """Configuration for benchmark runner."""

    backend: str = "openai"
    model: str = "gpt-5-mini"
    environment: str = "subprocess"
    max_iterations: int = 30
    verbose: bool = False
    log_dir: str | None = None
    max_workers: int = 1  # Number of parallel workers (1 = sequential)
    progress: str = "auto"  # Progress display: "auto", "tqdm", "simple", "none"
    progress_callback: ProgressCallback | None = None  # Custom progress callback
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
        max_workers: int = 1,
        progress: str = "auto",
        progress_callback: ProgressCallback | None = None,
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
            max_workers: Number of parallel workers (1 = sequential, >1 = parallel).
            progress: Progress display mode:
                - "auto": Use tqdm if available, else simple
                - "tqdm": Force tqdm progress bar
                - "simple": Print periodic status updates
                - "none": No progress output
            progress_callback: Custom callback for progress updates.
                Signature: (completed, total, sample_result, stats) -> None
            **kwargs: Additional backend or environment kwargs.
        """
        self.config = RunnerConfig(
            backend=backend,
            model=model,
            environment=environment,
            max_iterations=max_iterations,
            verbose=verbose,
            log_dir=log_dir,
            max_workers=max_workers,
            progress=progress,
            progress_callback=progress_callback,
            backend_kwargs={"model_name": model, **kwargs.get("backend_kwargs", {})},
            environment_kwargs=kwargs.get("environment_kwargs", {}),
        )
        self._tqdm_available: bool | None = None

    def _check_tqdm(self) -> bool:
        """Check if tqdm is available."""
        if self._tqdm_available is None:
            try:
                import tqdm  # noqa: F401

                self._tqdm_available = True
            except ImportError:
                self._tqdm_available = False
        return self._tqdm_available

    def _get_progress_mode(self) -> str:
        """Determine effective progress mode."""
        mode = self.config.progress
        if mode == "auto":
            return "tqdm" if self._check_tqdm() else "simple"
        if mode == "tqdm" and not self._check_tqdm():
            return "simple"
        return mode

    def _format_eta(self, seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds <= 0:
            return "--:--"
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}:{secs:02d}"
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins:02d}m"

    def run(
        self,
        benchmark: Benchmark,
        method: str = "rlm",
        num_samples: int | None = None,
        seed: int | None = None,
        custom_fn: Callable[[BenchmarkSample], str] | None = None,
        max_workers: int | None = None,
    ) -> BenchmarkResult:
        """Run a benchmark with the specified method.

        Args:
            benchmark: Benchmark instance to run.
            method: Inference method ("rlm", "direct", "summarize", "custom").
            num_samples: Number of samples to evaluate. None for all.
            seed: Random seed for reproducible sampling.
            custom_fn: Custom inference function for method="custom".
                       Takes BenchmarkSample, returns prediction string.
            max_workers: Override default max_workers for this run.
                         1 = sequential, >1 = parallel threads.

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
        workers = max_workers if max_workers is not None else self.config.max_workers

        # Collect samples first (needed for parallel execution)
        samples = list(benchmark.load_samples(num_samples=num_samples, seed=seed))
        total = len(samples)

        # Initialize progress tracking
        stats = ProgressStats(total=total)
        progress_mode = self._get_progress_mode()

        if workers <= 1:
            # Sequential execution with progress
            result.sample_results = self._run_sequential(
                samples, inference_fn, benchmark, stats, progress_mode
            )
        else:
            # Parallel execution with progress
            result.sample_results = self._run_parallel(
                samples, inference_fn, benchmark, workers, stats, progress_mode
            )

        return result

    def _update_progress(
        self,
        sample_result: SampleResult,
        stats: ProgressStats,
        progress_mode: str,
        pbar: Any = None,
    ) -> None:
        """Update progress statistics and display."""
        stats.completed += 1
        if sample_result.is_correct:
            stats.correct += 1
        if sample_result.error:
            stats.errors += 1
        stats.total_time_ms += sample_result.execution_time_ms

        # Call custom callback if provided
        if self.config.progress_callback:
            self.config.progress_callback(stats.completed, stats.total, sample_result, stats)

        # Update display based on mode
        if progress_mode == "tqdm" and pbar is not None:
            pbar.set_postfix(
                acc=f"{stats.accuracy:.1%}",
                err=stats.errors,
                eta=self._format_eta(stats.eta_seconds),
                refresh=False,
            )
            pbar.update(1)
        elif progress_mode == "simple":
            # Print every 10% or every sample for small runs
            interval = max(1, stats.total // 10)
            if stats.completed % interval == 0 or stats.completed == stats.total:
                print(
                    f"  Progress: {stats.completed}/{stats.total} "
                    f"({stats.completed / stats.total:.0%}) | "
                    f"Acc: {stats.accuracy:.1%} | "
                    f"Errors: {stats.errors} | "
                    f"ETA: {self._format_eta(stats.eta_seconds)}"
                )

        # Verbose per-sample output
        if self.config.verbose:
            status = "✓" if sample_result.is_correct else "✗"
            print(f"  [{status}] Sample {sample_result.sample_id}: {sample_result.metrics}")
            if sample_result.error:
                print(f"      Error: {sample_result.error}")

    def _run_sequential(
        self,
        samples: list[BenchmarkSample],
        inference_fn: Callable[[BenchmarkSample], tuple[str, dict[str, Any]]],
        benchmark: Benchmark,
        stats: ProgressStats,
        progress_mode: str,
    ) -> list[SampleResult]:
        """Run samples sequentially with progress tracking."""
        results: list[SampleResult] = []
        pbar = None

        try:
            if progress_mode == "tqdm":
                from tqdm import tqdm

                pbar = tqdm(
                    total=stats.total,
                    desc=f"{benchmark.name}",
                    unit="sample",
                    ncols=100,
                )

            for sample in samples:
                sample_result = self._run_sample(sample, inference_fn, benchmark)
                results.append(sample_result)
                self._update_progress(sample_result, stats, progress_mode, pbar)

        finally:
            if pbar is not None:
                pbar.close()

        return results

    def _run_parallel(
        self,
        samples: list[BenchmarkSample],
        inference_fn: Callable[[BenchmarkSample], tuple[str, dict[str, Any]]],
        benchmark: Benchmark,
        max_workers: int,
        stats: ProgressStats,
        progress_mode: str,
    ) -> list[SampleResult]:
        """Run samples in parallel using thread pool with progress tracking.

        Args:
            samples: List of samples to process.
            inference_fn: Inference function to apply.
            benchmark: Benchmark for evaluation.
            max_workers: Number of parallel threads.
            stats: Progress statistics to update.
            progress_mode: Progress display mode.

        Returns:
            List of SampleResult in original sample order.
        """
        import threading

        results: dict[str, SampleResult] = {}
        lock = threading.Lock()
        pbar = None

        try:
            if progress_mode == "tqdm":
                from tqdm import tqdm

                pbar = tqdm(
                    total=stats.total,
                    desc=f"{benchmark.name}",
                    unit="sample",
                    ncols=100,
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(self._run_sample, sample, inference_fn, benchmark): sample
                    for sample in samples
                }

                # Collect results as they complete
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        sample_result = future.result()
                    except Exception as e:
                        # Handle unexpected errors
                        sample_result = SampleResult(
                            sample_id=sample.id,
                            prediction="",
                            expected=sample.expected_answer,
                            is_correct=False,
                            metrics={m: 0.0 for m in benchmark.default_metrics()},
                            error=f"Parallel execution error: {e}",
                        )

                    with lock:
                        results[sample.id] = sample_result
                        self._update_progress(sample_result, stats, progress_mode, pbar)

        finally:
            if pbar is not None:
                pbar.close()

        # Return results in original sample order
        return [results[sample.id] for sample in samples]

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

        # Note: We always set RLM verbose=False to avoid its output interleaving
        # with the benchmark progress bar. The runner's verbose flag controls
        # per-sample result output instead.
        rlm = RLM(
            backend=self.config.backend,
            backend_kwargs=self.config.backend_kwargs,
            environment=self.config.environment,
            environment_kwargs=self.config.environment_kwargs,
            max_iterations=self.config.max_iterations,
            logger=logger,
            verbose=False,
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
