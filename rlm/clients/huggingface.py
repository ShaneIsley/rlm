import os
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient, InferenceClient, list_models

from rlm.clients.base_lm import BaseLM
from rlm.core.exceptions import InvalidPromptError, ModelRequiredError
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
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if api_key is None:
            api_key = DEFAULT_HF_TOKEN

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        # Initialize clients
        # Logic:
        # 1. If base_url is provided, connect to it. Use model_name for payload.
        # 2. If model_name is a URL, connect to it. Use None for payload (legacy/TGI behavior).
        # 3. Else, connect to Hub (default). Use model_name for payload.

        client_model = None
        if base_url:
            client_model = base_url
        elif model_name and model_name.startswith(("http://", "https://")):
            client_model = model_name

        # api_key is optional for public models, but required for private/gated models
        self.client = InferenceClient(token=api_key, model=client_model)
        self.async_client = AsyncInferenceClient(token=api_key, model=client_model)

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

        # Determine effective model for payload
        call_model = model
        if self.base_url:
            # Explicit base_url means we must send the model name in payload
            if not call_model:
                raise ModelRequiredError("HuggingFace client (base_url mode)")
        elif model and model.startswith(("http://", "https://")):
            # URL as model name -> Don't send it in payload (avoids 404/Bad Request on some backends)
            call_model = None
        elif not model:
            raise ModelRequiredError("HuggingFace client")

        # Note: InferenceClient.chat_completion is compatible with OpenAI messages format
        response = self.client.chat_completion(messages=messages, model=call_model)

        # Track cost using the logical model name (for reporting)
        track_model = model or "unknown"
        self._track_cost(response, track_model)

        return response.choices[0].message.content

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        messages = self._prepare_messages(prompt)

        model = model or self.model_name

        call_model = model
        if self.base_url:
            if not call_model:
                raise ModelRequiredError("HuggingFace client (base_url mode)")
        elif model and model.startswith(("http://", "https://")):
            call_model = None
        elif not model:
            raise ModelRequiredError("HuggingFace client")

        response = await self.async_client.chat_completion(messages=messages, model=call_model)

        track_model = model or "unknown"
        self._track_cost(response, track_model)

        return response.choices[0].message.content

    def _prepare_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return prompt
        else:
            raise InvalidPromptError(type(prompt))

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

    def list_models(self) -> list[str] | None:
        """List available text-generation models from Hugging Face Hub.

        Returns:
            Sorted list of model IDs that support text generation,
            or None if listing fails.
        """
        try:
            # List models that support text-generation (chat/completion)
            models = list_models(
                task="text-generation",
                sort="downloads",
                direction=-1,
                limit=100,
            )
            model_ids = sorted([m.id for m in models])
            return model_ids
        except Exception:
            return None

    async def alist_models(self) -> list[str] | None:
        """Async version of list_models.

        Returns:
            Sorted list of model IDs that support text generation,
            or None if listing fails.

        Note:
            Currently uses sync implementation as huggingface_hub's
            list_models doesn't have an async variant.
        """
        # huggingface_hub doesn't provide async list_models, use sync version
        return self.list_models()
