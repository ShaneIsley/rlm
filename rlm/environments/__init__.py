from typing import Any, Literal

from rlm.environments.base_env import BaseEnv, SupportsPersistence
from rlm.environments.local_repl import LocalREPL
from rlm.environments.registry import (
    ENVIRONMENT_REGISTRY,
    EnvironmentConfig,
    create_environment,
    get_supported_environments,
)

__all__ = [
    "BaseEnv",
    "LocalREPL",
    "SupportsPersistence",
    "get_environment",
    "ENVIRONMENT_REGISTRY",
    "EnvironmentConfig",
    "get_supported_environments",
]


def get_environment(
    environment: Literal["local", "modal", "docker", "prime"],
    environment_kwargs: dict[str, Any],
) -> BaseEnv:
    """
    Routes a specific environment and the args (as a dict) to the appropriate environment if supported.

    Args:
        environment: The environment type (e.g., "local", "docker", "modal", "prime")
        environment_kwargs: Keyword arguments to pass to the environment constructor

    Returns:
        An instance of BaseEnv for the specified environment type

    Raises:
        ValueError: If environment is not supported or validation fails
    """
    return create_environment(environment, environment_kwargs)
