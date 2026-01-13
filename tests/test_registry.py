"""Tests for registry-based client and environment loading."""

from unittest.mock import MagicMock, patch

import pytest

from rlm.clients import get_client, get_supported_backends
from rlm.clients.registry import (
    CLIENT_REGISTRY,
    ClientConfig,
    create_client,
    load_client_class,
)
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
        ]
        for backend in expected:
            assert backend in CLIENT_REGISTRY

    def test_unknown_backend_raises_with_supported_list(self):
        """Unknown backend raises ValueError listing all supported backends."""
        with pytest.raises(ValueError) as exc:
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
        """vllm backend raises error without base_url."""
        with pytest.raises(ValueError, match="base_url is required"):
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
        """Unknown environment raises ValueError listing all supported."""
        with pytest.raises(ValueError) as exc:
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
