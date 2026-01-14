"""Registry-based client loading for extensible backend support."""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rlm.core.exceptions import UnknownBackendError, ValidationError


@dataclass
class ClientConfig:
    """Configuration for a client backend.

    Attributes:
        module: The module path containing the client class (e.g., "rlm.clients.openai")
        class_name: The class name to instantiate (e.g., "OpenAIClient")
        defaults: Default kwargs to apply before user-provided kwargs
        env_vars: Mapping of kwarg names to environment variable names for automatic resolution
        validator: Optional function to validate kwargs before instantiation
    """

    module: str
    class_name: str
    defaults: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    validator: Callable[[dict[str, Any]], None] | None = None


def _validate_vllm_kwargs(kwargs: dict[str, Any]) -> None:
    """Validate that vLLM backend has required base_url."""
    if "base_url" not in kwargs:
        raise ValidationError(
            "base_url is required to be set to local vLLM server address for vLLM",
            backend="vllm",
        )


# Registry of all supported client backends
CLIENT_REGISTRY: dict[str, ClientConfig] = {
    "openai": ClientConfig(
        module="rlm.clients.openai",
        class_name="OpenAIClient",
        env_vars={"api_key": "OPENAI_API_KEY"},
    ),
    "vllm": ClientConfig(
        module="rlm.clients.openai",
        class_name="OpenAIClient",
        validator=_validate_vllm_kwargs,
        # vLLM typically doesn't need an API key, but allow override
        env_vars={"api_key": "VLLM_API_KEY"},
    ),
    "portkey": ClientConfig(
        module="rlm.clients.portkey",
        class_name="PortkeyClient",
        env_vars={"api_key": "PORTKEY_API_KEY"},
    ),
    "openrouter": ClientConfig(
        module="rlm.clients.openai",
        class_name="OpenAIClient",
        defaults={"base_url": "https://openrouter.ai/api/v1"},
        env_vars={"api_key": "OPENROUTER_API_KEY"},
    ),
    "vercel": ClientConfig(
        module="rlm.clients.openai",
        class_name="OpenAIClient",
        defaults={"base_url": "https://ai-gateway.vercel.sh/v1"},
        env_vars={"api_key": "AI_GATEWAY_API_KEY"},
    ),
    "litellm": ClientConfig(
        module="rlm.clients.litellm",
        class_name="LiteLLMClient",
        # LiteLLM handles its own env vars internally
    ),
    "anthropic": ClientConfig(
        module="rlm.clients.anthropic",
        class_name="AnthropicClient",
        env_vars={"api_key": "ANTHROPIC_API_KEY"},
    ),
    "gemini": ClientConfig(
        module="rlm.clients.gemini",
        class_name="GeminiClient",
        env_vars={"api_key": "GOOGLE_API_KEY"},
    ),
    "azure_openai": ClientConfig(
        module="rlm.clients.azure_openai",
        class_name="AzureOpenAIClient",
        env_vars={
            "api_key": "AZURE_OPENAI_API_KEY",
            "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        },
    ),
    "huggingface": ClientConfig(
        module="rlm.clients.huggingface",
        class_name="HuggingFaceClient",
        env_vars={"api_key": "HF_TOKEN"},
    ),
}


def get_supported_backends() -> list[str]:
    """Return sorted list of supported backend names."""
    return sorted(CLIENT_REGISTRY.keys())


def get_required_env_vars(backend: str) -> dict[str, str]:
    """Return the environment variable mappings for a backend.

    Args:
        backend: The backend name

    Returns:
        Dict mapping kwarg names to environment variable names

    Raises:
        UnknownBackendError: If backend is not in registry
    """
    if backend not in CLIENT_REGISTRY:
        raise UnknownBackendError(backend, get_supported_backends())

    return CLIENT_REGISTRY[backend].env_vars.copy()


def load_client_class(backend: str) -> type:
    """Load and return the client class for a backend.

    Args:
        backend: The backend name (e.g., "openai", "anthropic")

    Returns:
        The client class (not instantiated)

    Raises:
        UnknownBackendError: If backend is not in registry
    """
    if backend not in CLIENT_REGISTRY:
        raise UnknownBackendError(backend, get_supported_backends())

    config = CLIENT_REGISTRY[backend]
    module = importlib.import_module(config.module)
    return getattr(module, config.class_name)


def create_client(backend: str, backend_kwargs: dict[str, Any]):
    """Create a client instance for the given backend.

    Resolution order for kwargs:
    1. Environment variables (from env_vars mapping) - lowest priority
    2. Registry defaults - medium priority
    3. User-provided backend_kwargs - highest priority

    Args:
        backend: The backend name (e.g., "openai", "anthropic")
        backend_kwargs: Keyword arguments to pass to the client constructor

    Returns:
        An instance of BaseLM subclass

    Raises:
        UnknownBackendError: If backend is not in registry
        ValidationError: If validation fails
    """
    if backend not in CLIENT_REGISTRY:
        raise UnknownBackendError(backend, get_supported_backends())

    config = CLIENT_REGISTRY[backend]

    # 1. Start with env var values (lowest priority)
    kwargs: dict[str, Any] = {}
    for kwarg_name, env_var_name in config.env_vars.items():
        value = os.getenv(env_var_name)
        if value is not None:
            kwargs[kwarg_name] = value

    # 2. Apply registry defaults (override env vars)
    kwargs.update(config.defaults)

    # 3. Apply user kwargs (highest priority, override everything)
    kwargs.update(backend_kwargs)

    # Run validator if present
    if config.validator:
        config.validator(kwargs)

    # Load class and instantiate
    client_cls = load_client_class(backend)
    return client_cls(**kwargs)
