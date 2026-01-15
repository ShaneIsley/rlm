"""Tests for registry-based client and environment loading."""

import os
from unittest.mock import MagicMock, patch

import pytest

from rlm.clients import get_client, get_required_env_vars, get_supported_backends
from rlm.clients.registry import (
    CLIENT_REGISTRY,
    ClientConfig,
    create_client,
    load_client_class,
)
from rlm.core.exceptions import UnknownBackendError, UnknownEnvironmentError, ValidationError
from rlm.environments import get_environment, get_supported_environments
from rlm.environments.registry import (
    ENVIRONMENT_REGISTRY,
    EnvironmentConfig,
    create_environment,
    load_environment_class,
)


class TestClientRegistry:
    """Tests for client registry functionality."""

    def test_get_supported_backends_returns_sorted_list(self):
        """get_supported_backends returns sorted list of all backends."""
        backends = get_supported_backends()
        assert isinstance(backends, list)
        assert backends == sorted(backends)
        assert "openai" in backends
        assert "anthropic" in backends

    def test_registry_contains_all_expected_backends(self):
        """CLIENT_REGISTRY has all expected backends."""
        expected = [
            "openai",
            "vllm",
            "portkey",
            "openrouter",
            "vercel",
            "litellm",
            "anthropic",
            "gemini",
            "azure_openai",
            "huggingface",
        ]
        for backend in expected:
            assert backend in CLIENT_REGISTRY

    def test_unknown_backend_raises_with_supported_list(self):
        """Unknown backend raises UnknownBackendError listing all supported backends."""
        with pytest.raises(UnknownBackendError) as exc:
            get_client("unknown_backend", {})

        error_msg = str(exc.value)
        assert "unknown_backend" in error_msg
        assert "openai" in error_msg
        assert "anthropic" in error_msg

    def test_load_client_class_returns_class(self):
        """load_client_class returns the class, not an instance."""
        from rlm.clients.openai import OpenAIClient

        cls = load_client_class("openai")
        assert cls is OpenAIClient

    def test_openrouter_applies_default_base_url(self):
        """openrouter backend applies default base_url."""
        with patch("rlm.clients.openai.OpenAIClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            create_client("openrouter", {"model_name": "test", "api_key": "key"})

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_vercel_applies_default_base_url(self):
        """vercel backend applies default base_url."""
        with patch("rlm.clients.openai.OpenAIClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            create_client("vercel", {"model_name": "test", "api_key": "key"})

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://ai-gateway.vercel.sh/v1"

    def test_user_kwargs_override_defaults(self):
        """User-provided kwargs override registry defaults."""
        with patch("rlm.clients.openai.OpenAIClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            create_client(
                "openrouter",
                {"model_name": "test", "api_key": "key", "base_url": "https://custom.url"},
            )

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://custom.url"

    def test_vllm_requires_base_url(self):
        """vllm backend raises ValidationError without base_url."""
        with pytest.raises(ValidationError, match="base_url is required"):
            create_client("vllm", {"model_name": "test"})

    def test_vllm_with_base_url_succeeds(self):
        """vllm backend succeeds with base_url provided."""
        with patch("rlm.clients.openai.OpenAIClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            create_client(
                "vllm",
                {"model_name": "test", "base_url": "http://localhost:8000"},
            )

            mock_cls.assert_called_once()

    def test_client_config_dataclass(self):
        """ClientConfig dataclass works correctly."""
        config = ClientConfig(
            module="test.module",
            class_name="TestClass",
            defaults={"key": "value"},
        )
        assert config.module == "test.module"
        assert config.class_name == "TestClass"
        assert config.defaults == {"key": "value"}
        assert config.validator is None


class TestEnvironmentRegistry:
    """Tests for environment registry functionality."""

    def test_get_supported_environments_returns_sorted_list(self):
        """get_supported_environments returns sorted list."""
        environments = get_supported_environments()
        assert isinstance(environments, list)
        assert environments == sorted(environments)
        assert "local" in environments
        assert "docker" in environments

    def test_registry_contains_all_expected_environments(self):
        """ENVIRONMENT_REGISTRY has all expected environments."""
        expected = ["local", "modal", "docker", "prime"]
        for env in expected:
            assert env in ENVIRONMENT_REGISTRY

    def test_unknown_environment_raises_with_supported_list(self):
        """Unknown environment raises UnknownEnvironmentError listing all supported."""
        with pytest.raises(UnknownEnvironmentError) as exc:
            get_environment("unknown_env", {})

        error_msg = str(exc.value)
        assert "unknown_env" in error_msg
        assert "local" in error_msg
        assert "docker" in error_msg

    def test_load_environment_class_returns_class(self):
        """load_environment_class returns the class, not an instance."""
        from rlm.environments.local_repl import LocalREPL

        cls = load_environment_class("local")
        assert cls is LocalREPL

    def test_local_environment_creates_instance(self):
        """local environment creates LocalREPL instance."""
        from rlm.environments.local_repl import LocalREPL

        env = create_environment("local", {"context_payload": "test"})
        assert isinstance(env, LocalREPL)
        env.cleanup()

    def test_environment_config_dataclass(self):
        """EnvironmentConfig dataclass works correctly."""
        config = EnvironmentConfig(
            module="test.module",
            class_name="TestClass",
            defaults={"key": "value"},
        )
        assert config.module == "test.module"
        assert config.class_name == "TestClass"
        assert config.defaults == {"key": "value"}
        assert config.validator is None


class TestRegistryIntegration:
    """Integration tests for registry with RLM."""

    def test_get_client_returns_correct_type(self):
        """get_client returns a BaseLM instance."""
        from rlm.clients.base_lm import BaseLM

        # Mock the OpenAI client to avoid API key requirement
        with patch("rlm.clients.openai.OpenAIClient") as mock_cls:
            mock_instance = MagicMock(spec=BaseLM)
            mock_cls.return_value = mock_instance

            result = get_client("openai", {"model_name": "test"})
            assert result is mock_instance

    def test_get_environment_returns_correct_type(self):
        """get_environment returns a BaseEnv instance."""
        from rlm.environments.base_env import BaseEnv
        from rlm.environments.local_repl import LocalREPL

        env = get_environment("local", {"context_payload": "test"})
        assert isinstance(env, BaseEnv)
        assert isinstance(env, LocalREPL)
        env.cleanup()


class TestEnvVarResolution:
    """Tests for environment variable resolution in registry."""

    def test_get_required_env_vars_returns_mapping(self):
        """get_required_env_vars returns env var mapping for backend."""
        env_vars = get_required_env_vars("openai")
        assert env_vars == {"api_key": "OPENAI_API_KEY"}

    def test_get_required_env_vars_for_openrouter(self):
        """openrouter has its own env var."""
        env_vars = get_required_env_vars("openrouter")
        assert env_vars == {"api_key": "OPENROUTER_API_KEY"}

    def test_get_required_env_vars_for_azure(self):
        """azure_openai has multiple env vars."""
        env_vars = get_required_env_vars("azure_openai")
        assert "api_key" in env_vars
        assert "azure_endpoint" in env_vars
        assert env_vars["api_key"] == "AZURE_OPENAI_API_KEY"
        assert env_vars["azure_endpoint"] == "AZURE_OPENAI_ENDPOINT"

    def test_env_var_resolved_when_set(self):
        """Environment variables are resolved when set."""
        with patch("rlm.clients.openai.OpenAIClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            # Set env var
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-from-env"}):
                create_client("openai", {"model_name": "gpt-4"})

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["api_key"] == "test-key-from-env"

    def test_user_kwargs_override_env_vars(self):
        """User-provided kwargs override environment variables."""
        with patch("rlm.clients.openai.OpenAIClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
                create_client("openai", {"model_name": "gpt-4", "api_key": "user-key"})

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["api_key"] == "user-key"

    def test_defaults_override_env_vars(self):
        """Registry defaults override environment variables."""
        with patch("rlm.clients.openai.OpenAIClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            # openrouter has default base_url, even if env var was set for base_url
            create_client("openrouter", {"model_name": "test", "api_key": "key"})

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_client_config_has_env_vars_field(self):
        """ClientConfig dataclass has env_vars field."""
        config = ClientConfig(
            module="test.module",
            class_name="TestClass",
            env_vars={"api_key": "TEST_API_KEY"},
        )
        assert config.env_vars == {"api_key": "TEST_API_KEY"}

    def test_all_backends_have_env_vars_defined(self):
        """All backends that need API keys have env_vars defined."""
        # These backends should have api_key env var
        api_key_backends = [
            "openai",
            "openrouter",
            "vercel",
            "anthropic",
            "gemini",
            "portkey",
            "huggingface",
        ]
        for backend in api_key_backends:
            config = CLIENT_REGISTRY[backend]
            assert "api_key" in config.env_vars, f"{backend} missing api_key env var"


class TestListModels:
    """Tests for list_models functionality."""

    def test_base_lm_list_models_returns_none(self):
        """BaseLM.list_models returns None by default."""
        from rlm.clients.base_lm import BaseLM

        # Create a minimal concrete implementation for testing
        class TestLM(BaseLM):
            def completion(self, prompt):
                return "test"

            async def acompletion(self, prompt):
                return "test"

            def get_usage_summary(self):
                return None

            def get_last_usage(self):
                return None

        lm = TestLM(model_name="test")
        assert lm.list_models() is None

    def test_openai_client_has_list_models(self):
        """OpenAIClient implements list_models."""
        from rlm.clients.openai import OpenAIClient

        assert hasattr(OpenAIClient, "list_models")
        assert hasattr(OpenAIClient, "alist_models")

    def test_openai_list_models_returns_list_on_success(self):
        """OpenAIClient.list_models returns list when API succeeds."""
        from rlm.clients.openai import OpenAIClient

        with patch("openai.OpenAI") as mock_openai:
            # Mock the models.list() response
            mock_model1 = MagicMock()
            mock_model1.id = "gpt-4"
            mock_model2 = MagicMock()
            mock_model2.id = "gpt-3.5-turbo"
            mock_models_response = MagicMock()
            mock_models_response.data = [mock_model1, mock_model2]

            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_models_response
            mock_openai.return_value = mock_client

            client = OpenAIClient(api_key="test", model_name="gpt-4")
            models = client.list_models()

            assert models is not None
            assert isinstance(models, list)
            assert "gpt-4" in models
            assert "gpt-3.5-turbo" in models
            assert models == sorted(models)  # Should be sorted

    def test_openai_list_models_returns_none_on_error(self):
        """OpenAIClient.list_models returns None when API fails."""
        from rlm.clients.openai import OpenAIClient

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.models.list.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            client = OpenAIClient(api_key="test", model_name="gpt-4")
            models = client.list_models()

            assert models is None

    def test_huggingface_client_has_list_models(self):
        """HuggingFaceClient implements list_models."""
        from rlm.clients.huggingface import HuggingFaceClient

        assert hasattr(HuggingFaceClient, "list_models")
        assert hasattr(HuggingFaceClient, "alist_models")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
