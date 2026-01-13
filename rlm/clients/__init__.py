from typing import Any

from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.clients.registry import (
    CLIENT_REGISTRY,
    ClientConfig,
    create_client,
    get_required_env_vars,
    get_supported_backends,
)
from rlm.core.types import ClientBackend

load_dotenv()

__all__ = [
    "BaseLM",
    "get_client",
    "CLIENT_REGISTRY",
    "ClientConfig",
    "get_supported_backends",
    "get_required_env_vars",
]


def get_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any],
) -> BaseLM:
    """
    Routes a specific backend and the args (as a dict) to the appropriate client if supported.

    Args:
        backend: The backend identifier (e.g., "openai", "anthropic", "gemini")
        backend_kwargs: Keyword arguments to pass to the client constructor

    Returns:
        An instance of BaseLM for the specified backend

    Raises:
        ValueError: If backend is not supported or validation fails
    """
    return create_client(backend, backend_kwargs)
