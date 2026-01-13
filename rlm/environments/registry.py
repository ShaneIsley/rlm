"""Registry-based environment loading for extensible environment support."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rlm.core.exceptions import UnknownEnvironmentError


@dataclass
class EnvironmentConfig:
    """Configuration for an environment type.

    Attributes:
        module: The module path containing the environment class
        class_name: The class name to instantiate
        defaults: Default kwargs to apply before user-provided kwargs
        validator: Optional function to validate kwargs before instantiation
    """

    module: str
    class_name: str
    defaults: dict[str, Any] = field(default_factory=dict)
    validator: Callable[[dict[str, Any]], None] | None = None


# Registry of all supported environment types
ENVIRONMENT_REGISTRY: dict[str, EnvironmentConfig] = {
    "local": EnvironmentConfig(
        module="rlm.environments.local_repl",
        class_name="LocalREPL",
    ),
    "subprocess": EnvironmentConfig(
        module="rlm.environments.subprocess_repl",
        class_name="SubprocessREPL",
    ),
    "modal": EnvironmentConfig(
        module="rlm.environments.modal_repl",
        class_name="ModalREPL",
    ),
    "docker": EnvironmentConfig(
        module="rlm.environments.docker_repl",
        class_name="DockerREPL",
    ),
    "prime": EnvironmentConfig(
        module="rlm.environments.prime_repl",
        class_name="PrimeREPL",
    ),
}


def get_supported_environments() -> list[str]:
    """Return sorted list of supported environment names."""
    return sorted(ENVIRONMENT_REGISTRY.keys())


def load_environment_class(environment: str) -> type:
    """Load and return the environment class.

    Args:
        environment: The environment name (e.g., "local", "docker")

    Returns:
        The environment class (not instantiated)

    Raises:
        UnknownEnvironmentError: If environment is not in registry
    """
    if environment not in ENVIRONMENT_REGISTRY:
        raise UnknownEnvironmentError(environment, get_supported_environments())

    config = ENVIRONMENT_REGISTRY[environment]
    module = importlib.import_module(config.module)
    return getattr(module, config.class_name)


def create_environment(environment: str, environment_kwargs: dict[str, Any]):
    """Create an environment instance for the given type.

    Args:
        environment: The environment name (e.g., "local", "docker")
        environment_kwargs: Keyword arguments to pass to the environment constructor

    Returns:
        An instance of BaseEnv subclass

    Raises:
        UnknownEnvironmentError: If environment is not in registry
    """
    if environment not in ENVIRONMENT_REGISTRY:
        raise UnknownEnvironmentError(environment, get_supported_environments())

    config = ENVIRONMENT_REGISTRY[environment]

    # Apply defaults first, then user kwargs override
    kwargs = {**config.defaults, **environment_kwargs}

    # Run validator if present
    if config.validator:
        config.validator(kwargs)

    # Load class and instantiate
    env_cls = load_environment_class(environment)
    return env_cls(**kwargs)
