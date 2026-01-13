from abc import ABC, abstractmethod
from typing import Any

from rlm.core.types import UsageSummary


class BaseLM(ABC):
    """
    Base class for all language model routers / clients. When the RLM makes sub-calls, it currently
    does so in a model-agnostic way, so this class provides a base interface for all language models.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def completion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """Get cost summary for all model calls."""
        raise NotImplementedError

    @abstractmethod
    def get_last_usage(self) -> UsageSummary:
        """Get the last cost summary of the model."""
        raise NotImplementedError

    def list_models(self) -> list[str] | None:
        """List available models from the provider.

        Returns:
            List of model IDs if the provider supports listing models,
            None if not supported.

        Note:
            This method is optional. Subclasses that support model listing
            should override this method. The default implementation returns None.
        """
        return None

    async def alist_models(self) -> list[str] | None:
        """Async version of list_models.

        Returns:
            List of model IDs if the provider supports listing models,
            None if not supported.
        """
        return None
