"""Tests for custom exception hierarchy."""

import pytest

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


class TestExceptionHierarchy:
    """Tests for exception inheritance structure."""

    def test_all_exceptions_inherit_from_rlm_error(self):
        """All custom exceptions should inherit from RLMError."""
        exceptions = [
            ClientError,
            ConfigurationError,
            ModelRequiredError,
            APIError,
            InvalidPromptError,
            EnvironmentError,
            CodeExecutionError,
            VariableNotFoundError,
            HandlerNotConfiguredError,
            LMQueryError,
            RegistryError,
            UnknownBackendError,
            UnknownEnvironmentError,
            ValidationError,
            MaxIterationsError,
            PersistenceError,
        ]
        for exc_cls in exceptions:
            assert issubclass(exc_cls, RLMError), f"{exc_cls.__name__} should inherit from RLMError"

    def test_client_errors_inherit_from_client_error(self):
        """Client-related exceptions should inherit from ClientError."""
        client_exceptions = [
            ConfigurationError,
            ModelRequiredError,
            APIError,
            InvalidPromptError,
        ]
        for exc_cls in client_exceptions:
            assert issubclass(exc_cls, ClientError), (
                f"{exc_cls.__name__} should inherit from ClientError"
            )

    def test_environment_errors_inherit_from_environment_error(self):
        """Environment-related exceptions should inherit from EnvironmentError."""
        env_exceptions = [
            CodeExecutionError,
            VariableNotFoundError,
            HandlerNotConfiguredError,
            LMQueryError,
        ]
        for exc_cls in env_exceptions:
            assert issubclass(exc_cls, EnvironmentError), (
                f"{exc_cls.__name__} should inherit from EnvironmentError"
            )

    def test_registry_errors_inherit_from_registry_error(self):
        """Registry-related exceptions should inherit from RegistryError."""
        registry_exceptions = [
            UnknownBackendError,
            UnknownEnvironmentError,
            ValidationError,
        ]
        for exc_cls in registry_exceptions:
            assert issubclass(exc_cls, RegistryError), (
                f"{exc_cls.__name__} should inherit from RegistryError"
            )


class TestClientExceptions:
    """Tests for client exception behavior."""

    def test_model_required_error_message(self):
        """ModelRequiredError should format message correctly."""
        exc = ModelRequiredError("OpenAI client")
        assert "Model name is required for OpenAI client" in str(exc)
        assert exc.client_name == "OpenAI client"

    def test_model_required_error_default_client_name(self):
        """ModelRequiredError should use default client name."""
        exc = ModelRequiredError()
        assert "client" in str(exc)

    def test_invalid_prompt_error_message(self):
        """InvalidPromptError should format message with type name."""
        exc = InvalidPromptError(int)
        assert "int" in str(exc)
        assert exc.prompt_type is int

    def test_configuration_error_with_missing_field(self):
        """ConfigurationError should store missing field."""
        exc = ConfigurationError("API key is required", missing_field="api_key")
        assert "API key is required" in str(exc)
        assert exc.missing_field == "api_key"

    def test_api_error_with_status_code(self):
        """APIError should store status code and response body."""
        exc = APIError(
            "Rate limit exceeded", status_code=429, response_body={"error": "too many requests"}
        )
        assert "Rate limit exceeded" in str(exc)
        assert exc.status_code == 429
        assert exc.response_body == {"error": "too many requests"}


class TestEnvironmentExceptions:
    """Tests for environment exception behavior."""

    def test_variable_not_found_error_message(self):
        """VariableNotFoundError should format message correctly."""
        exc = VariableNotFoundError("result")
        assert "result" in str(exc)
        assert exc.variable_name == "result"

    def test_code_execution_error_with_details(self):
        """CodeExecutionError should store code and stderr."""
        exc = CodeExecutionError(
            "Syntax error", code="print(", stderr="SyntaxError: unexpected EOF"
        )
        assert "Syntax error" in str(exc)
        assert exc.code == "print("
        assert exc.stderr == "SyntaxError: unexpected EOF"

    def test_handler_not_configured_error_message(self):
        """HandlerNotConfiguredError should have a clear message."""
        exc = HandlerNotConfiguredError()
        assert "No LM handler configured" in str(exc)

    def test_lm_query_error_with_original(self):
        """LMQueryError should store original exception."""
        original = ValueError("connection failed")
        exc = LMQueryError("Query failed", original_error=original)
        assert "Query failed" in str(exc)
        assert exc.original_error is original


class TestRegistryExceptions:
    """Tests for registry exception behavior."""

    def test_unknown_backend_error_message(self):
        """UnknownBackendError should list supported backends."""
        exc = UnknownBackendError("unknown", ["openai", "anthropic", "gemini"])
        assert "unknown" in str(exc)
        assert "openai" in str(exc)
        assert exc.backend == "unknown"
        assert "anthropic" in exc.supported

    def test_unknown_environment_error_message(self):
        """UnknownEnvironmentError should list supported environments."""
        exc = UnknownEnvironmentError("unknown", ["local", "docker"])
        assert "unknown" in str(exc)
        assert "local" in str(exc)
        assert exc.environment == "unknown"
        assert "docker" in exc.supported

    def test_validation_error_with_backend(self):
        """ValidationError should store backend name."""
        exc = ValidationError("base_url is required", backend="vllm")
        assert "base_url is required" in str(exc)
        assert exc.backend == "vllm"


class TestCoreExceptions:
    """Tests for core RLM exception behavior."""

    def test_max_iterations_error_message(self):
        """MaxIterationsError should include iteration count."""
        exc = MaxIterationsError(10)
        assert "10" in str(exc)
        assert exc.max_iterations == 10

    def test_persistence_error_with_environment_type(self):
        """PersistenceError should store environment type."""
        exc = PersistenceError("Persistence not supported", environment_type="docker")
        assert "Persistence not supported" in str(exc)
        assert exc.environment_type == "docker"


class TestExceptionCatching:
    """Tests for catching exceptions at different levels."""

    def test_catch_all_rlm_errors(self):
        """Should be able to catch all RLM errors with RLMError."""
        exceptions_to_raise = [
            ModelRequiredError("test"),
            InvalidPromptError(str),
            UnknownBackendError("test", []),
            MaxIterationsError(10),
        ]

        for exc in exceptions_to_raise:
            with pytest.raises(RLMError):
                raise exc

    def test_catch_client_errors_subset(self):
        """Should be able to catch client-specific errors."""
        with pytest.raises(ClientError):
            raise ModelRequiredError("test")

        with pytest.raises(ClientError):
            raise InvalidPromptError(str)

    def test_catch_registry_errors_subset(self):
        """Should be able to catch registry-specific errors."""
        with pytest.raises(RegistryError):
            raise UnknownBackendError("test", [])

        with pytest.raises(RegistryError):
            raise ValidationError("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
