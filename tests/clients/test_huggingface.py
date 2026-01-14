"""Tests for the Hugging Face client."""

import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from dotenv import load_dotenv
from huggingface_hub.errors import HfHubHTTPError

from rlm.clients.huggingface import HuggingFaceClient
from rlm.core.exceptions import InvalidPromptError, ModelRequiredError
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

    def test_prepare_messages_list(self):
        """Test _prepare_messages with list of dicts input."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            input_messages = [{"role": "user", "content": "Hello"}]
            messages = client._prepare_messages(input_messages)
            self.assertEqual(messages, input_messages)

    def test_prepare_messages_invalid_type(self):
        """Test _prepare_messages raises InvalidPromptError for invalid input."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key")
            with self.assertRaises(InvalidPromptError):
                client._prepare_messages(12345)

    def test_completion_requires_model(self):
        """Test completion raises ModelRequiredError when no model specified."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key", model_name=None)
            with self.assertRaises(ModelRequiredError):
                client.completion("Hello")

    def test_completion_requires_model_with_base_url(self):
        """Test completion raises ModelRequiredError when using base_url without model."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"):
            client = HuggingFaceClient(api_key="test-key", base_url="https://example.com")
            with self.assertRaises(ModelRequiredError):
                client.completion("Hello")


class TestHuggingFaceListModels(unittest.TestCase):
    """Tests for list_models functionality."""

    def test_list_models_returns_sorted_list(self):
        """Test list_models returns sorted list of model IDs."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"), \
             patch("rlm.clients.huggingface.list_models") as mock_list:
            # Mock the list_models response
            mock_model1 = MagicMock()
            mock_model1.id = "meta-llama/Llama-2-7b"
            mock_model2 = MagicMock()
            mock_model2.id = "gpt2"
            mock_list.return_value = [mock_model1, mock_model2]

            client = HuggingFaceClient(api_key="test-key")
            models = client.list_models()

            self.assertIsNotNone(models)
            self.assertIsInstance(models, list)
            # Should be sorted
            self.assertEqual(models, sorted(models))
            self.assertIn("gpt2", models)
            self.assertIn("meta-llama/Llama-2-7b", models)

    def test_list_models_returns_none_on_error(self):
        """Test list_models returns None when API fails."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"), \
             patch("rlm.clients.huggingface.list_models") as mock_list:
            mock_list.side_effect = Exception("API Error")

            client = HuggingFaceClient(api_key="test-key")
            models = client.list_models()

            self.assertIsNone(models)

    def test_alist_models_delegates_to_list_models(self):
        """Test alist_models uses the sync list_models implementation."""
        with patch("rlm.clients.huggingface.InferenceClient"), \
             patch("rlm.clients.huggingface.AsyncInferenceClient"), \
             patch("rlm.clients.huggingface.list_models") as mock_list:
            mock_model = MagicMock()
            mock_model.id = "test-model"
            mock_list.return_value = [mock_model]

            client = HuggingFaceClient(api_key="test-key")

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(client.alist_models())

            self.assertEqual(result, ["test-model"])


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
        try:
            result = client.completion("What is 2+2? Reply with just the number.")
            self.assertTrue(len(result) > 0)

            # Verify usage was tracked
            usage = client.get_usage_summary()
            self.assertIn(model_name, usage.model_usage_summaries)
            self.assertEqual(usage.model_usage_summaries[model_name].total_calls, 1)
        except HfHubHTTPError as e:
            if e.response.status_code == 503:
                self.skipTest("Endpoint is scaled to zero (503 Service Unavailable)")
            raise

    async def test_async_completion(self):
        """Test async completion."""
        base_url = "https://gmbqgfi725l71vl9.us-east4.gcp.endpoints.huggingface.cloud"
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        client = HuggingFaceClient(model_name=model_name, base_url=base_url)
        try:
            result = await client.acompletion("What is 3+3? Reply with just the number.")
            self.assertTrue(len(result) > 0)
        except HfHubHTTPError as e:
            if e.response.status_code == 503:
                self.skipTest("Endpoint is scaled to zero (503 Service Unavailable)")
            raise


if __name__ == "__main__":
    unittest.main()
