from collections import defaultdict
from typing import Any

import openai

from rlm.clients.base_lm import BaseLM
from rlm.core.exceptions import APIError, InvalidPromptError, ModelRequiredError
from rlm.core.types import ModelUsageSummary, UsageSummary

# Prime Intellect uses OpenAI-compatible API
DEFAULT_PRIME_INTELLECT_BASE_URL = "https://api.pinference.ai/api/v1/"


class OpenAIClient(BaseLM):
    """
    LM Client for running models with the OpenAI API. Works with vLLM and
    other OpenAI-compatible providers as well.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        # API key is resolved by registry from env vars, but can be passed directly
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise InvalidPromptError(type(prompt))

        model = model or self.model_name
        if not model:
            raise ModelRequiredError("OpenAI client")

        extra_body = {}
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}

        response = self.client.chat.completions.create(
            model=model, messages=messages, extra_body=extra_body
        )
        self._track_cost(response, model)
        return response.choices[0].message.content

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise InvalidPromptError(type(prompt))

        model = model or self.model_name
        if not model:
            raise ModelRequiredError("OpenAI client")

        extra_body = {}
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra_body["usage"] = {"include": True}

        response = await self.async_client.chat.completions.create(
            model=model, messages=messages, extra_body=extra_body
        )
        self._track_cost(response, model)
        return response.choices[0].message.content

    def _track_cost(self, response: openai.ChatCompletion, model: str):
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage is None:
            raise APIError("No usage data received. Tracking tokens not possible.")

        self.model_input_tokens[model] += usage.prompt_tokens
        self.model_output_tokens[model] += usage.completion_tokens
        self.model_total_tokens[model] += usage.total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_completion_tokens = usage.completion_tokens

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

    def list_models(self) -> list[str] | None:
        """List available models from the OpenAI API.

        Returns:
            Sorted list of model IDs available from the API.
        """
        try:
            models = self.client.models.list()
            return sorted([model.id for model in models.data])
        except Exception:
            # May fail if API doesn't support model listing (e.g., some vLLM setups)
            return None

    async def alist_models(self) -> list[str] | None:
        """Async version of list_models.

        Returns:
            Sorted list of model IDs available from the API.
        """
        try:
            models = await self.async_client.models.list()
            return sorted([model.id for model in models.data])
        except Exception:
            return None
