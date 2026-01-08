"""Tests for the Hugging Face client."""

import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from dotenv import load_dotenv

from rlm.clients.huggingface import HuggingFaceClient
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()


class TestHuggingFaceClientUnit(unittest.TestCase):
    """Unit tests that don't require API calls."""

    def test_init_with_api_key(self):
        """Test client initialization with explicit API key."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key", model_name="test-model")
            self.assertEqual(client.model_name, "test-model")

    def test_init_with_base_url(self):
        """Test client initialization with base_url."""
        url = "https://example.com"
        model = "test-model"
        with patch("rlm.clients.huggingface.InferenceClient") as MockClient, \
             patch("rlm.clients.huggingface.AsyncInferenceClient") as MockAsyncClient:
            client = HuggingFaceClient(api_key="test-key", model_name=model, base_url=url)
            self.assertEqual(client.model_name, model)
            self.assertEqual(client.base_url, url)

            # Client init with URL
            MockClient.assert_called_with(token="test-key", model=url)

            # Completion calls use model name
            mock_inst = MockClient.return_value
            mock_inst.chat_completion.return_value.choices = [MagicMock(message=MagicMock(content="Hi"))]

            client.completion("Hello")
            mock_inst.chat_completion.assert_called_with(
                messages=[{"role": "user", "content": "Hello"}],
                model=model
            )

    def test_init_with_url_model_name(self):
        """Test legacy behavior: model_name is URL, no base_url."""
        url = "https://example.com"
        with patch("rlm.clients.huggingface.InferenceClient") as MockClient, \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key", model_name=url)

            # Client init with URL
            MockClient.assert_called_with(token="test-key", model=url)

            mock_inst = MockClient.return_value
            mock_inst.chat_completion.return_value.choices = [MagicMock(message=MagicMock(content="Hi"))]

            client.completion("Hello")
            # Completion call uses None for model
            mock_inst.chat_completion.assert_called_with(
                messages=[{"role": "user", "content": "Hello"}],
                model=None
            )

    def test_init_default_model(self):
        """Test client uses default model name (None if not provided)."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            self.assertIsNone(client.model_name)

    def test_usage_tracking_initialization(self):
        """Test that usage tracking is properly initialized."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            self.assertEqual(client.model_call_counts, {})
            self.assertEqual(client.model_input_tokens, {})
            self.assertEqual(client.model_output_tokens, {})
            self.assertEqual(client.last_prompt_tokens, 0)
            self.assertEqual(client.last_completion_tokens, 0)

    def test_get_usage_summary_empty(self):
        """Test usage summary when no calls have been made."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            summary = client.get_usage_summary()
            self.assertIsInstance(summary, UsageSummary)
            self.assertEqual(summary.model_usage_summaries, {})

    def test_get_last_usage(self):
        """Test last usage returns correct format."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            client.last_prompt_tokens = 100
            client.last_completion_tokens = 50
            usage = client.get_last_usage()
            self.assertIsInstance(usage, ModelUsageSummary)
            self.assertEqual(usage.total_calls, 1)
            self.assertEqual(usage.total_input_tokens, 100)
            self.assertEqual(usage.total_output_tokens, 50)

    def test_prepare_messages_string(self):
        """Test _prepare_messages with string input."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            messages = client._prepare_messages("Hello world")
            self.assertEqual(messages, [{"role": "user", "content": "Hello world"}])

    def test_completion_requires_model(self):
        """Test completion raises when no model specified."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key", model_name=None)
            with self.assertRaisesRegex(ValueError, "Model name is required"):
                client.completion("Hello")


class TestHuggingFaceClientIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests that require a real API key."""

    def setUp(self):
        if not os.environ.get("HF_TOKEN"):
            self.skipTest("HF_TOKEN not set")

    def test_simple_completion(self):
        """Test a simple completion with real API."""
        # Config provided by user
        base_url = "https://gmbqgfi725l71vl9.us-east4.gcp.endpoints.huggingface.cloud"
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        client = HuggingFaceClient(model_name=model_name, base_url=base_url)
        result = client.completion("What is 2+2? Reply with just the number.")
        self.assertTrue(len(result) > 0)

        # Verify usage was tracked
        usage = client.get_usage_summary()
        self.assertIn(model_name, usage.model_usage_summaries)
        self.assertEqual(usage.model_usage_summaries[model_name].total_calls, 1)

    async def test_async_completion(self):
        """Test async completion."""
        base_url = "https://gmbqgfi725l71vl9.us-east4.gcp.endpoints.huggingface.cloud"
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        client = HuggingFaceClient(model_name=model_name, base_url=base_url)
        result = await client.acompletion("What is 3+3? Reply with just the number.")
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()
