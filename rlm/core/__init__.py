"""Core RLM module."""

from rlm.core.exceptions import (
    APIError,
    ClientError,
    CodeExecutionError,
    ConfigurationError,
    EnvironmentError,
    HandlerNotConfiguredError,
    InvalidPromptError,
    LMQueryError,
    MaxIterationsError,
    ModelRequiredError,
    PersistenceError,
    RegistryError,
    RLMError,
    UnknownBackendError,
    UnknownEnvironmentError,
    ValidationError,
    VariableNotFoundError,
)

__all__ = [
    # Base
    "RLMError",
    # Client errors
    "ClientError",
    "ConfigurationError",
    "ModelRequiredError",
    "APIError",
    "InvalidPromptError",
    # Environment errors
    "EnvironmentError",
    "CodeExecutionError",
    "VariableNotFoundError",
    "HandlerNotConfiguredError",
    "LMQueryError",
    # Registry errors
    "RegistryError",
    "UnknownBackendError",
    "UnknownEnvironmentError",
    "ValidationError",
    # Core errors
    "MaxIterationsError",
    "PersistenceError",
]
