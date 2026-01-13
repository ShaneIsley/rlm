"""Registry-based client loading for extensible backend support."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientConfig:
    """Configuration for a client backend.

    Attributes:
        module: The module path containing the client class (e.g., "rlm.clients.openai")
        class_name: The class name to instantiate (e.g., "OpenAIClient")
        defaults: Default kwargs to apply before user-provided kwargs
        validator: Optional function to validate kwargs before instantiation
    """

    module: str
    class_name: str
    defaults: dict[str, Any] = field(default_factory=dict)
    validator: Callable[[dict[str, Any]], None] | None = None


def _validate_vllm_kwargs(kwargs: dict[str, Any]) -> None:
    """Validate that vLLM backend has required base_url."""
    if "base_url" not in kwargs:
        raise ValueError("base_url is required to be set to local vLLM server address for vLLM")


# Registry of all supported client backends
CLIENT_REGISTRY: dict[str, ClientConfig] = {
    "openai": ClientConfig(
        module="rlm.clients.openai",
        class_name="OpenAIClient",
    ),
    "vllm": ClientConfig(
        module="rlm.clients.openai",
        class_name="OpenAIClient",
        validator=_validate_vllm_kwargs,
    ),
    "portkey": ClientConfig(
        module="rlm.clients.portkey",
        class_name="PortkeyClient",
    ),
    "openrouter": ClientConfig(
        module="rlm.clients.openai",
        class_name="OpenAIClient",
        defaults={"base_url": "https://openrouter.ai/api/v1"},
    ),
    "vercel": ClientConfig(
        module="rlm.clients.openai",
        class_name="OpenAIClient",
        defaults={"base_url": "https://ai-gateway.vercel.sh/v1"},
    ),
    "litellm": ClientConfig(
        module="rlm.clients.litellm",
        class_name="LiteLLMClient",
    ),
    "anthropic": ClientConfig(
        module="rlm.clients.anthropic",
        class_name="AnthropicClient",
    ),
    "gemini": ClientConfig(
        module="rlm.clients.gemini",
        class_name="GeminiClient",
    ),
    "azure_openai": ClientConfig(
        module="rlm.clients.azure_openai",
        class_name="AzureOpenAIClient",
    ),
}


def get_supported_backends() -> list[str]:
    """Return sorted list of supported backend names."""
    return sorted(CLIENT_REGISTRY.keys())


def load_client_class(backend: str) -> type:
    """Load and return the client class for a backend.

    Args:
        backend: The backend name (e.g., "openai", "anthropic")

    Returns:
        The client class (not instantiated)

    Raises:
        ValueError: If backend is not in registry
    """
    if backend not in CLIENT_REGISTRY:
        supported = get_supported_backends()
        raise ValueError(f"Unknown backend: {backend}. Supported backends: {supported}")

    config = CLIENT_REGISTRY[backend]
    module = importlib.import_module(config.module)
    return getattr(module, config.class_name)


def create_client(backend: str, backend_kwargs: dict[str, Any]):
    """Create a client instance for the given backend.

    Args:
        backend: The backend name (e.g., "openai", "anthropic")
        backend_kwargs: Keyword arguments to pass to the client constructor

    Returns:
        An instance of BaseLM subclass

    Raises:
        ValueError: If backend is not in registry or validation fails
    """
    if backend not in CLIENT_REGISTRY:
        supported = get_supported_backends()
        raise ValueError(f"Unknown backend: {backend}. Supported backends: {supported}")

    config = CLIENT_REGISTRY[backend]

    # Apply defaults first, then user kwargs override
    kwargs = {**config.defaults, **backend_kwargs}

    # Run validator if present
    if config.validator:
        config.validator(kwargs)

    # Load class and instantiate
    client_cls = load_client_class(backend)
    return client_cls(**kwargs)
