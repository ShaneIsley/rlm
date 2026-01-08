import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from rlm.clients.huggingface import HuggingFaceClient
from rlm.core.types import ModelUsageSummary, UsageSummary

class TestHuggingFaceClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_async_client = AsyncMock()

        # Patch InferenceClient and AsyncInferenceClient
        self.patcher1 = patch('rlm.clients.huggingface.InferenceClient', return_value=self.mock_client)
        self.patcher2 = patch('rlm.clients.huggingface.AsyncInferenceClient', return_value=self.mock_async_client)

        self.mock_inference_client = self.patcher1.start()
        self.mock_async_inference_client = self.patcher2.start()

        self.api_key = "hf_test_key"
        self.model_name = "test-model"
        self.client = HuggingFaceClient(api_key=self.api_key, model_name=self.model_name)

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()

    def test_init(self):
        self.mock_inference_client.assert_called_with(token=self.api_key)
        self.mock_async_inference_client.assert_called_with(token=self.api_key)
        self.assertEqual(self.client.model_name, self.model_name)

    def test_completion(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        self.mock_client.chat_completion.return_value = mock_response

        prompt = "Test prompt"
        response = self.client.completion(prompt)

        self.assertEqual(response, "Test response")
        self.mock_client.chat_completion.assert_called_once_with(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name
        )

        # Verify usage tracking
        last_usage = self.client.get_last_usage()
        self.assertEqual(last_usage.total_input_tokens, 10)
        self.assertEqual(last_usage.total_output_tokens, 20)

    async def test_acompletion(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Async test response"))]
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40

        self.mock_async_client.chat_completion.return_value = mock_response

        prompt = "Async test prompt"
        response = await self.client.acompletion(prompt)

        self.assertEqual(response, "Async test response")
        self.mock_async_client.chat_completion.assert_called_once_with(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name
        )

        # Verify usage tracking
        last_usage = self.client.get_last_usage()
        self.assertEqual(last_usage.total_input_tokens, 15)
        self.assertEqual(last_usage.total_output_tokens, 25)

    def test_usage_summary(self):
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock(message=MagicMock(content="Resp 1"))]
        mock_response1.usage.prompt_tokens = 10
        mock_response1.usage.completion_tokens = 10
        mock_response1.usage.total_tokens = 20

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock(message=MagicMock(content="Resp 2"))]
        mock_response2.usage.prompt_tokens = 20
        mock_response2.usage.completion_tokens = 20
        mock_response2.usage.total_tokens = 40

        self.mock_client.chat_completion.side_effect = [mock_response1, mock_response2]

        self.client.completion("Prompt 1")
        self.client.completion("Prompt 2")

        usage_summary = self.client.get_usage_summary()
        model_summary = usage_summary.model_usage_summaries[self.model_name]

        self.assertEqual(model_summary.total_calls, 2)
        self.assertEqual(model_summary.total_input_tokens, 30)
        self.assertEqual(model_summary.total_output_tokens, 30)

    def test_prepare_messages(self):
        # Test string prompt
        msgs = self.client._prepare_messages("Hello")
        self.assertEqual(msgs, [{"role": "user", "content": "Hello"}])

        # Test list prompt
        msgs_in = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hi"}]
        msgs_out = self.client._prepare_messages(msgs_in)
        self.assertEqual(msgs_out, msgs_in)

        # Test invalid
        with self.assertRaises(ValueError):
            self.client._prepare_messages(123)

    def test_completion_no_model_error(self):
        client = HuggingFaceClient(api_key=self.api_key) # No model name
        with self.assertRaises(ValueError):
            client.completion("Prompt")

    def test_completion_no_usage(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        del mock_response.usage # Simulate no usage data

        self.mock_client.chat_completion.return_value = mock_response

        self.client.completion("Prompt")

        last_usage = self.client.get_last_usage()
        self.assertEqual(last_usage.total_input_tokens, 0)
        self.assertEqual(last_usage.total_output_tokens, 0)

        summary = self.client.get_usage_summary()
        self.assertEqual(summary.model_usage_summaries[self.model_name].total_calls, 1)

if __name__ == '__main__':
    unittest.main()
