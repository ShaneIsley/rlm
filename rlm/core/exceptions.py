"""Custom exception hierarchy for RLM.

This module provides structured exceptions for better error handling across
the RLM codebase. The hierarchy allows catching errors at different levels
of specificity.

Exception Hierarchy:
    RLMError (base)
    ├── ClientError
    │   ├── ConfigurationError
    │   ├── ModelRequiredError
    │   └── APIError
    ├── EnvironmentError
    │   ├── CodeExecutionError
    │   ├── VariableNotFoundError
    │   └── SandboxError
    │       ├── SandboxUnavailableError
    │       └── SandboxCapabilityError
    └── RegistryError
        ├── UnknownBackendError
        └── UnknownEnvironmentError
"""

from typing import Any


class RLMError(Exception):
    """Base exception for all RLM errors.

    All RLM-specific exceptions inherit from this class, allowing
    users to catch all RLM errors with a single except clause.

    Example:
        try:
            rlm.completion(prompt)
        except RLMError as e:
            print(f"RLM error occurred: {e}")
    """

    pass


# =============================================================================
# Client Errors
# =============================================================================


class ClientError(RLMError):
    """Base exception for LLM client errors.

    Raised when there's an issue with client configuration or API calls.
    """

    pass


class ConfigurationError(ClientError):
    """Invalid client configuration.

    Raised when required configuration is missing or invalid.
    """

    def __init__(self, message: str, missing_field: str | None = None):
        super().__init__(message)
        self.missing_field = missing_field


class ModelRequiredError(ClientError):
    """Model name is required but not provided.

    Raised when a completion is attempted without specifying a model.
    """

    def __init__(self, client_name: str = "client"):
        message = f"Model name is required for {client_name}."
        super().__init__(message)
        self.client_name = client_name


class APIError(ClientError):
    """Error from upstream LLM API.

    Raised when the LLM provider returns an error response.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class InvalidPromptError(ClientError):
    """Invalid prompt format.

    Raised when the prompt is not a valid type or format.
    """

    def __init__(self, prompt_type: type):
        message = f"Invalid prompt type: {prompt_type.__name__}. Expected str or list[dict]."
        super().__init__(message)
        self.prompt_type = prompt_type


# =============================================================================
# Environment Errors
# =============================================================================


class EnvironmentError(RLMError):
    """Base exception for environment errors.

    Raised when there's an issue with code execution environments.

    Note: This shadows the built-in EnvironmentError, but that's acceptable
    since the built-in is rarely used directly and RLM's is more specific.
    """

    pass


class CodeExecutionError(EnvironmentError):
    """Error executing code in environment.

    Raised when code execution fails in a REPL environment.
    """

    def __init__(self, message: str, code: str | None = None, stderr: str | None = None):
        super().__init__(message)
        self.code = code
        self.stderr = stderr


class VariableNotFoundError(EnvironmentError):
    """Variable not found in environment namespace.

    Raised when FINAL_VAR references a non-existent variable.
    """

    def __init__(self, variable_name: str):
        message = f"Variable '{variable_name}' not found"
        super().__init__(message)
        self.variable_name = variable_name


class HandlerNotConfiguredError(EnvironmentError):
    """LM handler not configured for environment.

    Raised when attempting LLM queries without a configured handler.
    """

    def __init__(self):
        super().__init__("No LM handler configured")


class LMQueryError(EnvironmentError):
    """Error during LLM query from environment.

    Raised when an LLM query from within the environment fails.
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(f"LM query failed: {message}")
        self.original_error = original_error


# =============================================================================
# Sandbox Errors
# =============================================================================


class SandboxError(EnvironmentError):
    """Base exception for sandbox-related errors.

    Raised when there's an issue with sandbox configuration or enforcement.
    """

    pass


class SandboxUnavailableError(SandboxError):
    """No sandbox strategy available on this system.

    Raised when SubprocessREPL cannot find a usable sandbox implementation.
    """

    def __init__(self, messages: list[str]):
        combined = "\n".join(f"  - {m}" for m in messages)
        message = f"No sandbox available on this system. Options:\n{combined}"
        super().__init__(message)
        self.messages = messages


class SandboxCapabilityError(SandboxError):
    """Requested permissions exceed sandbox capabilities.

    Raised when permissions require capabilities the sandbox cannot enforce.
    """

    def __init__(
        self,
        strategy_name: str,
        capability: str,
        suggestion: str | None = None,
    ):
        message = f"{strategy_name} cannot enforce {capability}."
        if suggestion:
            message += f" {suggestion}"
        super().__init__(message)
        self.strategy_name = strategy_name
        self.capability = capability
        self.suggestion = suggestion


# =============================================================================
# Registry Errors
# =============================================================================


class RegistryError(RLMError):
    """Base exception for registry errors.

    Raised when there's an issue with client or environment registry.
    """

    pass


class UnknownBackendError(RegistryError):
    """Unknown backend requested.

    Raised when attempting to use a backend not in the registry.
    """

    def __init__(self, backend: str, supported: list[str]):
        message = f"Unknown backend: {backend}. Supported backends: {supported}"
        super().__init__(message)
        self.backend = backend
        self.supported = supported


class UnknownEnvironmentError(RegistryError):
    """Unknown environment requested.

    Raised when attempting to use an environment not in the registry.
    """

    def __init__(self, environment: str, supported: list[str]):
        message = f"Unknown environment: {environment}. Supported: {supported}"
        super().__init__(message)
        self.environment = environment
        self.supported = supported


class ValidationError(RegistryError):
    """Backend or environment validation failed.

    Raised when configuration validation fails (e.g., missing required fields).
    """

    def __init__(self, message: str, backend: str | None = None):
        super().__init__(message)
        self.backend = backend


# =============================================================================
# Core RLM Errors
# =============================================================================


class MaxIterationsError(RLMError):
    """Maximum iterations reached without finding answer.

    Raised when the RLM loop exhausts max_iterations without a FINAL answer.
    """

    def __init__(self, max_iterations: int):
        message = f"Maximum iterations ({max_iterations}) reached without finding a final answer"
        super().__init__(message)
        self.max_iterations = max_iterations


class PersistenceError(RLMError):
    """Error with persistent environment.

    Raised when there's an issue with persistent mode configuration or state.
    """

    def __init__(self, message: str, environment_type: str | None = None):
        super().__init__(message)
        self.environment_type = environment_type
