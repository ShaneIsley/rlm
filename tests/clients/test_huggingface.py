"""Tests for the Hugging Face client."""

import os
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from dotenv import load_dotenv

from rlm.clients.huggingface import HuggingFaceClient
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()


class TestHuggingFaceClientUnit:
    """Unit tests that don't require API calls."""

    def test_init_with_api_key(self):
        """Test client initialization with explicit API key."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key", model_name="test-model")
            assert client.model_name == "test-model"

    def test_init_default_model(self):
        """Test client uses default model name (None if not provided)."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            assert client.model_name is None

    def test_init_uses_env_key(self):
        """Test client uses environment variable if no key provided."""
        with patch.dict(os.environ, {"HF_TOKEN": "env-key"}, clear=True):
             with patch("rlm.clients.huggingface.DEFAULT_HF_TOKEN", "env-key"):
                with patch("rlm.clients.huggingface.InferenceClient") as MockClient, \
                     patch("rlm.clients.huggingface.AsyncInferenceClient") as MockAsyncClient:

                    HuggingFaceClient()
                    MockClient.assert_called_with(token="env-key")
                    MockAsyncClient.assert_called_with(token="env-key")

    def test_usage_tracking_initialization(self):
        """Test that usage tracking is properly initialized."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            assert client.model_call_counts == {}
            assert client.model_input_tokens == {}
            assert client.model_output_tokens == {}
            assert client.last_prompt_tokens == 0
            assert client.last_completion_tokens == 0

    def test_get_usage_summary_empty(self):
        """Test usage summary when no calls have been made."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            summary = client.get_usage_summary()
            assert isinstance(summary, UsageSummary)
            assert summary.model_usage_summaries == {}

    def test_get_last_usage(self):
        """Test last usage returns correct format."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            client.last_prompt_tokens = 100
            client.last_completion_tokens = 50
            usage = client.get_last_usage()
            assert isinstance(usage, ModelUsageSummary)
            assert usage.total_calls == 1
            assert usage.total_input_tokens == 100
            assert usage.total_output_tokens == 50

    def test_prepare_messages_string(self):
        """Test _prepare_messages with string input."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            messages = client._prepare_messages("Hello world")
            assert messages == [{"role": "user", "content": "Hello world"}]

    def test_prepare_messages_list(self):
        """Test _prepare_messages with list input."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            input_messages = [{"role": "user", "content": "Hello"}]
            messages = client._prepare_messages(input_messages)
            assert messages == input_messages

    def test_prepare_messages_invalid_type(self):
        """Test _prepare_messages raises on invalid input."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            with pytest.raises(ValueError, match="Invalid prompt type"):
                client._prepare_messages(12345)

    def test_completion_requires_model(self):
        """Test completion raises when no model specified."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key", model_name=None)
            with pytest.raises(ValueError, match="Model name is required"):
                client.completion("Hello")

    def test_completion_with_mocked_response(self):
        """Test completion with mocked API response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from HF!"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        with patch("rlm.clients.huggingface.InferenceClient") as mock_client_class, \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):

            mock_client = MagicMock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = HuggingFaceClient(api_key="test-key", model_name="test-model")
            result = client.completion("Hello")

            assert result == "Hello from HF!"
            assert client.model_call_counts["test-model"] == 1
            assert client.model_input_tokens["test-model"] == 10
            assert client.model_output_tokens["test-model"] == 5

    def test_completion_with_missing_usage(self):
        """Test completion when usage data is missing (common in some HF endpoints)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        del mock_response.usage # Simulate missing usage

        with patch("rlm.clients.huggingface.InferenceClient") as mock_client_class, \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):

            mock_client = MagicMock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = HuggingFaceClient(api_key="test-key", model_name="test-model")
            result = client.completion("Hello")

            assert result == "Hello!"
            assert client.model_call_counts["test-model"] == 1
            assert client.last_prompt_tokens == 0
            assert client.last_completion_tokens == 0


class TestHuggingFaceClientIntegration:
    """Integration tests that require a real API key."""

    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="HF_TOKEN not set",
    )
    def test_simple_completion(self):
        """Test a simple completion with real API."""
        # Using a popular model supported by Inference API
        model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        client = HuggingFaceClient(model_name=model)
        result = client.completion("What is 2+2? Reply with just the number.")
        assert len(result) > 0

        # Verify usage was tracked (if available, otherwise 0)
        usage = client.get_usage_summary()
        assert model in usage.model_usage_summaries
        assert usage.model_usage_summaries[model].total_calls == 1

    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="HF_TOKEN not set",
    )
    @pytest.mark.asyncio
    async def test_async_completion(self):
        """Test async completion."""
        model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        client = HuggingFaceClient(model_name=model)
        result = await client.acompletion("What is 3+3? Reply with just the number.")
        assert len(result) > 0


if __name__ == "__main__":
    # Run integration tests directly
    pass
