import os
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient, InferenceClient

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()

DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN")


class HuggingFaceClient(BaseLM):
    """
    LM Client for running models with the Hugging Face Inference API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if api_key is None:
            api_key = DEFAULT_HF_TOKEN

        # api_key is optional for public models, but required for private/gated models
        self.client = InferenceClient(token=api_key)
        self.async_client = AsyncInferenceClient(token=api_key)
        self.model_name = model_name

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        messages = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Hugging Face client.")

        # Note: InferenceClient.chat_completion is compatible with OpenAI messages format
        response = self.client.chat_completion(messages=messages, model=model)

        self._track_cost(response, model)
        return response.choices[0].message.content

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        messages = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Hugging Face client.")

        response = await self.async_client.chat_completion(messages=messages, model=model)

        self._track_cost(response, model)
        return response.choices[0].message.content

    def _prepare_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_cost(self, response: Any, model: str):
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage:
            self.model_input_tokens[model] += usage.prompt_tokens
            self.model_output_tokens[model] += usage.completion_tokens
            self.model_total_tokens[model] += usage.total_tokens

            self.last_prompt_tokens = usage.prompt_tokens
            self.last_completion_tokens = usage.completion_tokens
        else:
            # If usage is not available, we can't track it.
            # Some HF endpoints might not return usage stats.
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
