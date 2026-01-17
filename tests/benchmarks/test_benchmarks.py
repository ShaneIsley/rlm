"""Tests for benchmark framework components."""

from benchmarks.base import BenchmarkResult, BenchmarkSample, SampleResult
from benchmarks.metrics import Metrics
from benchmarks.tasks.niah import NIAHBenchmark
from benchmarks.tasks.oolong import OolongPairsBenchmark


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_normalize(self):
        """Test text normalization."""
        assert Metrics.normalize("Hello, World!") == "hello world"
        assert Metrics.normalize("  Multiple   Spaces  ") == "multiple spaces"
        assert Metrics.normalize("UPPERCASE") == "uppercase"

    def test_exact_match_single(self):
        """Test exact match with single expected."""
        assert Metrics.exact_match("hello", "hello")
        assert Metrics.exact_match("Hello!", "hello")
        assert not Metrics.exact_match("hello world", "hello")

    def test_exact_match_multiple(self):
        """Test exact match with multiple expected answers."""
        assert Metrics.exact_match("hello", ["hello", "hi", "hey"])
        assert Metrics.exact_match("hi", ["hello", "hi", "hey"])
        assert not Metrics.exact_match("goodbye", ["hello", "hi", "hey"])

    def test_containment(self):
        """Test containment matching."""
        assert Metrics.containment("The answer is 42", "42")
        assert Metrics.containment("42", "The answer is 42")
        assert Metrics.containment("hello world", ["hello", "goodbye"])
        assert not Metrics.containment("hello", "world")

    def test_token_f1(self):
        """Test token-level F1 score."""
        # Perfect match
        assert Metrics.token_f1("hello world", "hello world") == 1.0

        # Partial overlap
        f1 = Metrics.token_f1("hello world today", "hello world")
        assert 0.5 < f1 < 1.0

        # No overlap
        assert Metrics.token_f1("hello", "goodbye") == 0.0

        # Empty prediction
        assert Metrics.token_f1("", "hello") == 0.0

    def test_pairwise_f1(self):
        """Test pairwise F1 for OOLONG-Pairs."""
        pred = {("A", "B"), ("C", "D")}
        exp = {("A", "B"), ("C", "D")}
        result = Metrics.pairwise_f1(pred, exp)
        assert result["f1"] == 1.0

        # Partial overlap
        pred = {("A", "B"), ("E", "F")}
        exp = {("A", "B"), ("C", "D")}
        result = Metrics.pairwise_f1(pred, exp)
        assert result["f1"] == 0.5

        # Order independence
        pred = {("B", "A")}
        exp = {("A", "B")}
        result = Metrics.pairwise_f1(pred, exp)
        assert result["f1"] == 1.0

    def test_evaluate_standard(self):
        """Test standard evaluation combining metrics."""
        result = Metrics.evaluate_standard("The answer is 42", "42")
        assert result["correct"] == 1.0
        assert result["containment"] == 1.0
        assert result["f1"] > 0


class TestBenchmarkSample:
    """Tests for BenchmarkSample."""

    def test_create_sample(self):
        """Test creating a benchmark sample."""
        sample = BenchmarkSample(
            id="test-001",
            context="This is the context.",
            question="What is this?",
            expected_answer="context",
        )
        assert sample.id == "test-001"
        assert sample.context_tokens_approx > 0

    def test_sample_with_multiple_answers(self):
        """Test sample with multiple valid answers."""
        sample = BenchmarkSample(
            id="test-002",
            context="Context",
            question="Question?",
            expected_answer=["answer1", "answer2"],
        )
        assert len(sample.expected_answer) == 2


class TestBenchmarkResult:
    """Tests for BenchmarkResult aggregation."""

    def test_empty_result(self):
        """Test empty result defaults."""
        result = BenchmarkResult(
            benchmark_name="test",
            method="rlm",
            model="test-model",
        )
        assert result.accuracy == 0.0
        assert result.mean_f1 == 0.0
        assert result.error_rate == 0.0

    def test_result_aggregation(self):
        """Test metric aggregation."""
        result = BenchmarkResult(
            benchmark_name="test",
            method="rlm",
            model="test-model",
            sample_results=[
                SampleResult(
                    sample_id="1",
                    prediction="correct",
                    expected="correct",
                    is_correct=True,
                    metrics={"f1": 1.0},
                    iterations=5,
                    total_tokens=100,
                ),
                SampleResult(
                    sample_id="2",
                    prediction="wrong",
                    expected="right",
                    is_correct=False,
                    metrics={"f1": 0.0},
                    iterations=3,
                    total_tokens=50,
                ),
            ],
        )
        assert result.accuracy == 0.5
        assert result.mean_f1 == 0.5
        assert result.total_tokens == 150
        assert result.mean_iterations == 4.0

    def test_result_summary(self):
        """Test summary dict generation."""
        result = BenchmarkResult(
            benchmark_name="test",
            method="rlm",
            model="test-model",
        )
        summary = result.summary()
        assert "benchmark" in summary
        assert "accuracy" in summary
        assert "mean_f1" in summary


class TestNIAHBenchmark:
    """Tests for NIAH benchmark."""

    def test_load_samples(self):
        """Test loading NIAH samples."""
        benchmark = NIAHBenchmark(context_length=10_000)
        samples = list(benchmark.load_samples(num_samples=5, seed=42))

        assert len(samples) == 5
        for sample in samples:
            assert len(sample.context) > 0
            assert len(sample.question) > 0
            assert len(sample.expected_answer) > 0

    def test_needle_in_context(self):
        """Test that needle is present in context."""
        benchmark = NIAHBenchmark(context_length=5_000)
        sample = next(benchmark.load_samples(num_samples=1, seed=42))

        # The expected answer should be findable in context
        needle = sample.metadata["needle"]
        assert needle in sample.context

    def test_evaluate(self):
        """Test NIAH evaluation."""
        benchmark = NIAHBenchmark()

        # Exact match
        result = benchmark.evaluate("abc123", "abc123")
        assert result["correct"] == 1.0

        # Containment
        result = benchmark.evaluate("The code is abc123.", "abc123")
        assert result["correct"] == 1.0

        # Miss
        result = benchmark.evaluate("xyz789", "abc123")
        assert result["correct"] == 0.0


class TestOolongPairsBenchmark:
    """Tests for OOLONG-Pairs benchmark."""

    def test_load_samples(self):
        """Test loading OOLONG-Pairs samples."""
        benchmark = OolongPairsBenchmark(num_items=20, num_pairs=10)
        samples = list(benchmark.load_samples(num_samples=3, seed=42))

        assert len(samples) == 3
        for sample in samples:
            assert len(sample.expected_answer) == 10
            assert all(isinstance(p, tuple) for p in sample.expected_answer)

    def test_parse_pairs(self):
        """Test parsing pairs from prediction."""
        benchmark = OolongPairsBenchmark()

        # Test various formats
        pred = "(Alice, Bob) and (Charlie, Diana)"
        pairs = benchmark._parse_pairs(pred)
        assert len(pairs) >= 2

        pred = "Alice and Bob collaborated. Charlie with Diana worked."
        pairs = benchmark._parse_pairs(pred)
        assert len(pairs) >= 2

    def test_evaluate(self):
        """Test OOLONG-Pairs evaluation."""
        benchmark = OolongPairsBenchmark()

        # Perfect match
        expected = [("Alice", "Bob"), ("Charlie", "Diana")]
        prediction = "(Alice, Bob), (Charlie, Diana)"
        result = benchmark.evaluate(prediction, expected)
        assert result["f1"] == 1.0

        # Partial match
        prediction = "(Alice, Bob), (Eve, Frank)"
        result = benchmark.evaluate(prediction, expected)
        assert 0 < result["f1"] < 1.0


class TestParallelExecution:
    """Tests for parallel execution safety."""

    def test_parallel_custom_fn_isolation(self):
        """Test that parallel execution with custom functions don't interfere."""
        import tempfile
        import threading
        import time
        from pathlib import Path

        from benchmarks.runner import BenchmarkRunner
        from benchmarks.tasks.niah import NIAHBenchmark

        # Track which thread processed which sample
        thread_ids = {}
        lock = threading.Lock()

        def custom_inference(sample):
            """Custom inference that tests isolation by writing temp files."""
            # Create a temp file unique to this invocation
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                temp_path = f.name
                f.write(sample.id)

            # Small delay to increase chance of race conditions
            time.sleep(0.01)

            # Read back and verify it's still our data
            with open(temp_path) as f:
                read_id = f.read()

            # Clean up
            Path(temp_path).unlink()

            # Record thread info
            with lock:
                thread_ids[sample.id] = threading.current_thread().ident

            # Return expected answer if our temp file wasn't clobbered
            if read_id == sample.id:
                return sample.expected_answer
            else:
                return f"CLOBBERED: expected {sample.id}, got {read_id}"

        benchmark = NIAHBenchmark(context_length=1000)
        runner = BenchmarkRunner(max_workers=4)

        result = runner.run(
            benchmark,
            method="custom",
            custom_fn=custom_inference,
            num_samples=8,
            seed=42,
            max_workers=4,
        )

        # All samples should be correct (temp files weren't clobbered)
        assert result.accuracy == 1.0, (
            f"Some samples were clobbered: {[sr for sr in result.sample_results if not sr.is_correct]}"
        )

        # Verify multiple threads were used
        unique_threads = set(thread_ids.values())
        assert len(unique_threads) > 1, "Expected multiple threads to be used"

    def test_parallel_preserves_order(self):
        """Test that parallel results maintain original sample order."""
        from benchmarks.runner import BenchmarkRunner
        from benchmarks.tasks.niah import NIAHBenchmark

        def custom_inference(sample):
            return sample.expected_answer

        benchmark = NIAHBenchmark(context_length=1000)
        runner = BenchmarkRunner(max_workers=4)

        result = runner.run(
            benchmark,
            method="custom",
            custom_fn=custom_inference,
            num_samples=10,
            seed=42,
            max_workers=4,
        )

        # Verify results are in expected order
        expected_ids = [f"niah-{i:04d}" for i in range(10)]
        actual_ids = [sr.sample_id for sr in result.sample_results]
        assert actual_ids == expected_ids, f"Order mismatch: {actual_ids}"

    def test_parallel_error_handling(self):
        """Test that errors in parallel execution are handled gracefully."""
        from benchmarks.runner import BenchmarkRunner
        from benchmarks.tasks.niah import NIAHBenchmark

        call_count = {"count": 0}

        def flaky_inference(sample):
            call_count["count"] += 1
            if call_count["count"] % 3 == 0:
                raise ValueError(f"Simulated error for sample {sample.id}")
            return sample.expected_answer

        benchmark = NIAHBenchmark(context_length=1000)
        runner = BenchmarkRunner(max_workers=2)

        result = runner.run(
            benchmark,
            method="custom",
            custom_fn=flaky_inference,
            num_samples=6,
            seed=42,
            max_workers=2,
        )

        # Should complete without crashing
        assert len(result.sample_results) == 6

        # Some should have errors
        errors = [sr for sr in result.sample_results if sr.error]
        assert len(errors) > 0, "Expected some errors"

        # Errors should have zero metrics
        for sr in errors:
            assert sr.is_correct is False


class TestProgressTracking:
    """Tests for progress tracking functionality."""

    def test_progress_stats_calculation(self):
        """Test ProgressStats computes metrics correctly."""
        from benchmarks.runner import ProgressStats

        stats = ProgressStats(total=100)
        assert stats.accuracy == 0.0
        assert stats.eta_seconds == 0.0

        # Simulate some progress
        stats.completed = 25
        stats.correct = 20
        stats.errors = 2
        stats.total_time_ms = 5000.0  # 5 seconds

        assert stats.accuracy == 0.8  # 20/25
        assert stats.error_rate == 0.08  # 2/25
        assert stats.avg_time_ms == 200.0  # 5000/25
        # ETA: 75 remaining * 200ms / 1000 = 15 seconds
        assert stats.eta_seconds == 15.0

    def test_progress_callback_invoked(self):
        """Test that custom progress callbacks are called."""
        from benchmarks.runner import BenchmarkRunner
        from benchmarks.tasks.niah import NIAHBenchmark

        callback_invocations = []

        def track_callback(completed, total, sample_result, stats):
            callback_invocations.append(
                {
                    "completed": completed,
                    "total": total,
                    "accuracy": stats.accuracy,
                }
            )

        benchmark = NIAHBenchmark(context_length=1000)
        runner = BenchmarkRunner(
            progress="none",
            progress_callback=track_callback,
        )

        runner.run(
            benchmark,
            method="custom",
            custom_fn=lambda s: s.expected_answer,
            num_samples=5,
            seed=42,
        )

        assert len(callback_invocations) == 5
        assert callback_invocations[-1]["completed"] == 5
        assert callback_invocations[-1]["total"] == 5
        assert callback_invocations[-1]["accuracy"] == 1.0

    def test_simple_progress_mode(self, capsys):
        """Test simple progress mode outputs status."""
        from benchmarks.runner import BenchmarkRunner
        from benchmarks.tasks.niah import NIAHBenchmark

        benchmark = NIAHBenchmark(context_length=1000)
        runner = BenchmarkRunner(progress="simple")

        runner.run(
            benchmark,
            method="custom",
            custom_fn=lambda s: s.expected_answer,
            num_samples=10,
            seed=42,
        )

        captured = capsys.readouterr()
        # Simple mode should print progress at completion
        assert "Progress:" in captured.out
        assert "10/10" in captured.out
        assert "Acc:" in captured.out


class TestModelSpec:
    """Tests for model specification parsing."""

    def test_parse_model_spec_with_backend(self):
        """Test parsing backend:model format."""
        from benchmarks.cli import parse_model_spec

        spec = parse_model_spec("openai:gpt-4o")
        assert spec.backend == "openai"
        assert spec.model == "gpt-4o"

        spec = parse_model_spec("anthropic:claude-sonnet-4-20250514")
        assert spec.backend == "anthropic"
        assert spec.model == "claude-sonnet-4-20250514"

    def test_parse_model_spec_model_only(self):
        """Test parsing model-only format defaults to openai."""
        from benchmarks.cli import parse_model_spec

        spec = parse_model_spec("gpt-4o")
        assert spec.backend == "openai"
        assert spec.model == "gpt-4o"

    def test_model_spec_str(self):
        """Test ModelSpec string representation."""
        from benchmarks.cli import ModelSpec

        spec = ModelSpec(backend="anthropic", model="claude-sonnet-4-20250514")
        assert str(spec) == "anthropic:claude-sonnet-4-20250514"


class TestModelPropagation:
    """Tests that model configuration is correctly passed through the system."""

    def test_runner_config_model_name(self):
        """Test that BenchmarkRunner correctly sets model_name in backend_kwargs."""
        from benchmarks.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            backend="openai",
            model="my-test-model",
            progress="none",
        )

        # Verify the config has the correct model_name in backend_kwargs
        assert runner.config.backend == "openai"
        assert runner.config.model == "my-test-model"
        assert runner.config.backend_kwargs.get("model_name") == "my-test-model"

    def test_runner_config_preserves_extra_kwargs(self):
        """Test that extra backend_kwargs are preserved."""
        from benchmarks.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            backend="openai",
            model="my-model",
            progress="none",
            backend_kwargs={"extra_param": "extra_value", "api_key": "test-key"},
        )

        # model_name from model param should be present
        assert runner.config.backend_kwargs.get("model_name") == "my-model"
        # Extra kwargs should be preserved
        assert runner.config.backend_kwargs.get("extra_param") == "extra_value"
        assert runner.config.backend_kwargs.get("api_key") == "test-key"

    def test_model_spec_to_runner(self):
        """Test end-to-end from model spec parsing to runner config."""
        from benchmarks.cli import parse_model_spec
        from benchmarks.runner import BenchmarkRunner

        spec = parse_model_spec("anthropic:claude-3-opus")
        runner = BenchmarkRunner(
            backend=spec.backend,
            model=spec.model,
            progress="none",
        )

        assert runner.config.backend == "anthropic"
        assert runner.config.model == "claude-3-opus"
        assert runner.config.backend_kwargs.get("model_name") == "claude-3-opus"

    def test_rlm_receives_correct_config(self):
        """Test that RLM is initialized with correct backend_kwargs."""
        from unittest.mock import MagicMock, patch

        from benchmarks.runner import BenchmarkRunner
        from benchmarks.tasks.niah import NIAHBenchmark

        runner = BenchmarkRunner(
            backend="openai",
            model="gpt-4o-test",
            progress="none",
        )

        # Mock RLM to capture the initialization arguments
        captured_kwargs = {}

        class MockRLM:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                # Raise to exit early without actually running RLM
                raise RuntimeError("Mock exit - RLM init captured")

        # RLM is imported inside _inference_rlm, so patch at rlm module level
        with patch("rlm.RLM", MockRLM):
            benchmark = NIAHBenchmark(context_length=1000)
            try:
                runner.run(benchmark, method="rlm", num_samples=1, seed=42)
            except RuntimeError as e:
                if "Mock exit" not in str(e):
                    raise

        # Verify the backend_kwargs passed to RLM
        assert captured_kwargs.get("backend") == "openai"
        assert captured_kwargs.get("backend_kwargs") == {"model_name": "gpt-4o-test"}


class TestErrorClassification:
    """Tests for error classification and fatal error handling."""

    def test_classify_quota_error(self):
        """Test that quota errors are classified as fatal."""
        from benchmarks.runner import classify_error

        error = "Error code: 429 - {'error': {'code': 'insufficient_quota'}}"
        is_fatal, suggestion = classify_error(error)
        assert is_fatal
        assert "quota" in suggestion.lower() or "billing" in suggestion.lower()

    def test_classify_auth_error(self):
        """Test that auth errors are classified as fatal."""
        from benchmarks.runner import classify_error

        error = "Invalid API key provided"
        is_fatal, suggestion = classify_error(error)
        assert is_fatal
        assert "key" in suggestion.lower()

    def test_classify_model_error(self):
        """Test that model not found errors are classified as fatal."""
        from benchmarks.runner import classify_error

        error = "The model 'gpt-99' does not exist"
        is_fatal, suggestion = classify_error(error)
        assert is_fatal
        assert "model" in suggestion.lower()

    def test_classify_transient_error(self):
        """Test that transient errors are not classified as fatal."""
        from benchmarks.runner import classify_error

        error = "Connection timeout after 30 seconds"
        is_fatal, suggestion = classify_error(error)
        assert not is_fatal
        assert suggestion == ""

    def test_classify_generic_error(self):
        """Test that unknown errors are not classified as fatal."""
        from benchmarks.runner import classify_error

        error = "Something unexpected happened"
        is_fatal, suggestion = classify_error(error)
        assert not is_fatal

    def test_classify_timeout_error(self):
        """Test that timeout errors are classified as fatal with helpful suggestion."""
        from benchmarks.runner import classify_error

        error = "Sample timed out after 300s"
        is_fatal, suggestion = classify_error(error)
        assert is_fatal
        assert "timeout" in suggestion.lower() or "context" in suggestion.lower()


class TestBenchmarkIntegration:
    """Integration tests for benchmark framework."""

    def test_niah_reproducibility(self):
        """Test that same seed produces same samples."""
        benchmark = NIAHBenchmark(context_length=5_000)

        samples1 = list(benchmark.load_samples(num_samples=3, seed=123))
        samples2 = list(benchmark.load_samples(num_samples=3, seed=123))

        for s1, s2 in zip(samples1, samples2, strict=True):
            assert s1.expected_answer == s2.expected_answer
            assert s1.question == s2.question

    def test_different_seeds_different_samples(self):
        """Test that different seeds produce different samples."""
        benchmark = NIAHBenchmark(context_length=5_000)

        samples1 = list(benchmark.load_samples(num_samples=3, seed=1))
        samples2 = list(benchmark.load_samples(num_samples=3, seed=2))

        # At least one should be different
        answers1 = [s.expected_answer for s in samples1]
        answers2 = [s.expected_answer for s in samples2]
        assert answers1 != answers2
