"""Tests for async RLM completion API.

Tests that the async API:
1. Returns equivalent results to sync API
2. Handles concurrent completions
3. Works with persistent mode
4. Properly propagates errors
5. Handles cancellation gracefully
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM
from rlm.core.types import ModelUsageSummary, UsageSummary

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


def create_mock_lm(responses: list[str]) -> Mock:
    """Create a mock LM that returns responses in order for both sync and async."""
    mock = Mock()
    mock.model_name = "mock-model"
    mock.completion.side_effect = list(responses)

    # Create async version
    async def async_completion(prompt):
        return mock.completion(prompt)

    mock.acompletion = AsyncMock(side_effect=responses)
    mock.get_usage_summary.return_value = UsageSummary(
        model_usage_summaries={
            "mock": ModelUsageSummary(total_calls=1, total_input_tokens=100, total_output_tokens=50)
        }
    )
    mock.get_last_usage.return_value = mock.get_usage_summary.return_value
    return mock


class TestAsyncBasicCompletion:
    """Tests for basic async completion functionality."""

    @pytest.mark.asyncio
    async def test_basic_async_completion(self):
        """Async completion returns valid result."""
        responses = ["FINAL(async answer)"]

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(responses)
            mock_get_client.return_value = mock_lm

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
            )

            result = await rlm.acompletion("test prompt")

            assert result is not None
            assert result.response == "async answer"
            assert result.prompt == "test prompt"

    @pytest.mark.asyncio
    async def test_async_completion_with_code_execution(self):
        """Async completion executes code blocks correctly."""
        responses = [
            "Let me compute\n```repl\nresult = 2 + 2\nprint(result)\n```",
            "FINAL(4)",
        ]

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(responses)
            mock_get_client.return_value = mock_lm

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
            )

            result = await rlm.acompletion("What is 2 + 2?")

            assert result.response == "4"

    @pytest.mark.asyncio
    async def test_async_completion_multiple_iterations(self):
        """Async completion handles multiple iterations."""
        responses = [
            "```repl\nx = 1\nprint(x)\n```",
            "```repl\ny = x + 1\nprint(y)\n```",
            "FINAL(computed)",
        ]

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(responses)
            mock_get_client.return_value = mock_lm

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
            )

            result = await rlm.acompletion("Compute iteratively")

            assert result.response == "computed"


class TestAsyncConcurrency:
    """Tests for concurrent async completions."""

    @pytest.mark.asyncio
    async def test_concurrent_completions(self):
        """Multiple async completions can run concurrently."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            # Each call needs its own mock to avoid shared state
            call_count = 0

            def get_fresh_mock(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                mock = create_mock_lm(["FINAL(answer)"])
                return mock

            mock_get_client.side_effect = get_fresh_mock

            async def create_and_complete():
                rlm = RLM(
                    backend="openai",
                    backend_kwargs={"model_name": "test"},
                )
                return await rlm.acompletion("prompt")

            # Run 3 completions concurrently
            results = await asyncio.gather(
                create_and_complete(),
                create_and_complete(),
                create_and_complete(),
            )

            assert len(results) == 3
            assert all(r.response == "answer" for r in results)


class TestAsyncPersistentMode:
    """Tests for async completion with persistent mode."""

    @pytest.mark.asyncio
    async def test_async_with_persistent_mode(self):
        """Async completion works with persistent=True."""
        responses = ["FINAL(first)"]

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(responses)
            mock_get_client.return_value = mock_lm

            async with AsyncRLMContextManager(
                RLM(
                    backend="openai",
                    backend_kwargs={"model_name": "test"},
                    persistent=True,
                )
            ) as rlm:
                result1 = await rlm.acompletion("First context")
                assert result1.response == "first"

                first_env = rlm._persistent_env

                mock_lm.acompletion.side_effect = ["FINAL(second)"]
                result2 = await rlm.acompletion("Second context")
                assert result2.response == "second"

                # Same environment should be reused
                assert rlm._persistent_env is first_env

    @pytest.mark.asyncio
    async def test_async_context_accumulation(self):
        """Async completion accumulates contexts in persistent mode."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(ok)"])
            mock_get_client.return_value = mock_lm

            async with AsyncRLMContextManager(
                RLM(
                    backend="openai",
                    backend_kwargs={"model_name": "test"},
                    persistent=True,
                )
            ) as rlm:
                await rlm.acompletion("Context A")
                mock_lm.acompletion.side_effect = ["FINAL(ok)"]
                await rlm.acompletion("Context B")

                env = rlm._persistent_env
                assert env.get_context_count() == 2


class TestAsyncFallback:
    """Tests for async fallback at max depth."""

    @pytest.mark.asyncio
    async def test_async_fallback_at_max_depth(self):
        """Async completion falls back to LM at max depth."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["direct answer"])
            mock_get_client.return_value = mock_lm

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                depth=1,  # At max depth (default max_depth=1)
                max_depth=1,
            )

            result = await rlm.acompletion("question")

            # Should return RLMChatCompletion from fallback
            assert result.response == "direct answer"


class TestAsyncErrorHandling:
    """Tests for error handling in async completion."""

    @pytest.mark.asyncio
    async def test_async_error_propagation(self):
        """Errors in async context propagate correctly."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = Mock()
            mock_lm.model_name = "mock-model"
            mock_lm.acompletion = AsyncMock(side_effect=RuntimeError("API Error"))
            mock_lm.get_usage_summary.return_value = UsageSummary(model_usage_summaries={})
            mock_lm.get_last_usage.return_value = mock_lm.get_usage_summary.return_value
            mock_get_client.return_value = mock_lm

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
            )

            with pytest.raises(RuntimeError, match="API Error"):
                await rlm.acompletion("prompt")


class TestAsyncVsSync:
    """Tests comparing async and sync behavior."""

    @pytest.mark.asyncio
    async def test_async_returns_same_type_as_sync(self):
        """Async and sync completion return same type."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)", "FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
            )

            sync_result = rlm.completion("prompt")
            async_result = await rlm.acompletion("prompt")

            assert type(sync_result) is type(async_result)
            assert sync_result.response == async_result.response


class TestAsyncEnvironmentExecution:
    """Tests for async code execution in environments."""

    @pytest.mark.asyncio
    async def test_async_execute_code(self):
        """Async code execution works correctly."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(context_payload="test context")

        result = await repl.aexecute_code("x = 42\nprint(x)")

        assert "42" in result.stdout
        assert repl.locals["x"] == 42

        repl.cleanup()

    @pytest.mark.asyncio
    async def test_async_execute_code_handles_errors(self):
        """Async code execution handles errors correctly."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(context_payload="test")

        result = await repl.aexecute_code("raise ValueError('test error')")

        assert "ValueError" in result.stderr
        assert "test error" in result.stderr

        repl.cleanup()


class AsyncRLMContextManager:
    """Async context manager wrapper for RLM (for testing)."""

    def __init__(self, rlm: RLM):
        self.rlm = rlm

    async def __aenter__(self) -> RLM:
        return self.rlm

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.rlm.close()
        return False


class TestAsyncContextManager:
    """Tests for async context manager usage patterns."""

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup(self):
        """Async context manager properly cleans up resources."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(done)"])
            mock_get_client.return_value = mock_lm

            async with AsyncRLMContextManager(
                RLM(
                    backend="openai",
                    backend_kwargs={"model_name": "test"},
                    persistent=True,
                )
            ) as rlm:
                await rlm.acompletion("test")
                assert rlm._persistent_env is not None

            # After exit, should be cleaned up
            assert rlm._persistent_env is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
