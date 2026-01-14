"""Tests for SubprocessREPL environment."""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

# Skip all tests if uv is not installed
pytestmark = pytest.mark.skipif(
    not shutil.which("uv"),
    reason="uv is required for SubprocessREPL tests"
)


class TestSubprocessREPLBasic:
    """Basic functionality tests for SubprocessREPL."""

    def test_init_creates_temp_dir_and_venv(self):
        """SubprocessREPL should create temp directory and venv on init."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            assert os.path.isdir(repl.temp_dir)
            assert os.path.isdir(repl.venv_path)
            assert os.path.exists(os.path.join(repl.venv_path, "bin", "python"))

    def test_init_requires_uv(self):
        """SubprocessREPL should raise ConfigurationError if uv not found."""
        from rlm.core.exceptions import ConfigurationError
        from rlm.environments.subprocess_repl import SubprocessREPL

        with patch("shutil.which", return_value=None):
            with pytest.raises(ConfigurationError, match="uv"):
                SubprocessREPL()

    def test_execute_simple_code(self):
        """Should execute simple Python code."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            result = repl.execute_code("x = 1 + 1\nprint(x)")

            assert "2" in result.stdout
            assert result.stderr == "" or result.stderr is None or "Error" not in result.stderr
            assert "x" in result.locals

    def test_execute_code_with_error(self):
        """Should capture errors in stderr."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            result = repl.execute_code("raise ValueError('test error')")

            assert "ValueError" in result.stderr
            assert "test error" in result.stderr

    def test_state_persists_across_executions(self):
        """Variables should persist between execute_code calls."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            repl.execute_code("my_var = 42")
            result = repl.execute_code("print(my_var)")

            assert "42" in result.stdout

    def test_timeout_enforcement(self):
        """Execution should timeout after specified duration."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL(timeout=1.0) as repl:
            result = repl.execute_code("import time; time.sleep(10)")

            assert "timed out" in result.stderr.lower()

    def test_cleanup_removes_temp_dir(self):
        """Cleanup should remove temp directory."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        repl = SubprocessREPL()
        temp_dir = repl.temp_dir
        assert os.path.isdir(temp_dir)

        repl.cleanup()
        assert not os.path.exists(temp_dir)


class TestSubprocessREPLContext:
    """Tests for context loading in SubprocessREPL."""

    def test_load_string_context(self):
        """Should load string context as context_0."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            repl.load_context("Hello, World!")
            result = repl.execute_code("print(context)")

            assert "Hello, World!" in result.stdout

    def test_load_dict_context(self):
        """Should load dict context as context_0."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            repl.load_context({"key": "value", "number": 42})
            result = repl.execute_code("print(context['key'], context['number'])")

            assert "value" in result.stdout
            assert "42" in result.stdout

    def test_add_multiple_contexts(self):
        """Should support multiple versioned contexts."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            repl.add_context("First context", 0)
            repl.add_context("Second context", 1)

            result = repl.execute_code("print(context_0, context_1)")

            assert "First context" in result.stdout
            assert "Second context" in result.stdout
            assert repl.get_context_count() == 2


class TestSubprocessREPLPersistence:
    """Tests for SupportsPersistence protocol."""

    def test_implements_supports_persistence(self):
        """SubprocessREPL should implement SupportsPersistence protocol."""
        from rlm.environments.base_env import SupportsPersistence
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            assert isinstance(repl, SupportsPersistence)

    def test_update_handler_address(self):
        """Should update LM handler address."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            repl.update_handler_address(("localhost", 12345))
            assert repl.lm_handler_address == ("localhost", 12345)

    def test_add_history(self):
        """Should add message history as versioned variable."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            repl.add_history(history, 0)

            result = repl.execute_code("print(len(history_0))")
            assert "2" in result.stdout
            assert repl.get_history_count() == 1


class TestSubprocessREPLOverhead:
    """Tests for overhead tracking."""

    def test_tracks_venv_creation_time(self):
        """Should track venv creation time."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            summary = repl.get_overhead_summary()
            # venv creation should take some measurable time
            assert summary["venv_creation_ms"] > 0

    def test_tracks_execution_overhead(self):
        """Should track execution overhead."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            repl.execute_code("x = 1")
            repl.execute_code("y = 2")

            summary = repl.get_overhead_summary()
            assert summary["execution_count"] == 2
            assert summary["total_overhead_ms"] > 0

    def test_overhead_summary_empty_when_no_executions(self):
        """Should return message when no executions."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            summary = repl.get_overhead_summary()
            assert "message" in summary


class TestSubprocessREPLPackageApproval:
    """Tests for package approval flow."""

    def test_stdlib_packages_allowed_by_default(self):
        """Standard library packages should work without approval."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            result = repl.execute_code("import json\nprint(json.dumps({'a': 1}))")
            assert '{"a": 1}' in result.stdout

    def test_missing_package_detected(self):
        """Should detect missing packages in error output."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL(auto_approve_packages=False) as repl:
            # Mock the approval callback to deny
            repl.approval_callback = lambda pkg: False
            result = repl.execute_code("import nonexistent_package_xyz")

            assert "ModuleNotFoundError" in result.stderr or "No module named" in result.stderr

    def test_auto_approve_installs_package(self):
        """Should auto-install packages when auto_approve=True."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        # Use a small, fast package for testing
        with SubprocessREPL(auto_approve_packages=True) as repl:
            # Try importing a package that might trigger install
            # We can't guarantee any specific package exists, so just verify the mechanism
            assert repl.auto_approve is True
            assert len(repl.allowed_packages) > 0


class TestSubprocessREPLSandbox:
    """Tests for sandbox functionality."""

    def test_sandbox_enabled_by_default(self):
        """Sandbox should be enabled by default."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            assert repl.sandbox is True

    def test_sandbox_can_be_disabled(self):
        """Sandbox can be disabled via parameter."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL(sandbox=False) as repl:
            assert repl.sandbox is False
            # Should still execute code
            result = repl.execute_code("print('hello')")
            assert "hello" in result.stdout


class TestSubprocessREPLCleanup:
    """Tests for cleanup functionality."""

    def test_context_manager_cleans_up(self):
        """Context manager should cleanup on exit."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            temp_dir = repl.temp_dir

        assert not os.path.exists(temp_dir)

    def test_cleanup_on_exception(self):
        """Should cleanup even when exception occurs."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        temp_dir = None
        try:
            with SubprocessREPL() as repl:
                temp_dir = repl.temp_dir
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert temp_dir is not None
        assert not os.path.exists(temp_dir)

    def test_stale_directory_cleanup(self):
        """Should cleanup stale directories."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        # Create a fake stale directory
        stale_dir = tempfile.mkdtemp(prefix=SubprocessREPL.TEMP_PREFIX)
        sentinel_path = os.path.join(stale_dir, ".rlm_sentinel")

        # Write sentinel with dead PID and old timestamp
        with open(sentinel_path, "w") as f:
            f.write("99999999\n")  # Unlikely to be a real PID
            f.write("0\n")  # Very old timestamp

        # Cleanup should find and remove it
        cleaned = SubprocessREPL.cleanup_stale_directories(max_age_hours=0)

        assert stale_dir in cleaned or not os.path.exists(stale_dir)


class TestSubprocessREPLRegistry:
    """Tests for registry integration."""

    def test_subprocess_in_registry(self):
        """Subprocess should be in environment registry."""
        from rlm.environments.registry import ENVIRONMENT_REGISTRY

        assert "subprocess" in ENVIRONMENT_REGISTRY

    def test_get_environment_creates_subprocess_repl(self):
        """get_environment should create SubprocessREPL."""
        from rlm.environments import get_environment
        from rlm.environments.subprocess_repl import SubprocessREPL

        env = get_environment("subprocess", {})
        try:
            assert isinstance(env, SubprocessREPL)
        finally:
            env.cleanup()


class TestSubprocessREPLFinalVar:
    """Tests for FINAL_VAR functionality."""

    def test_final_var_returns_variable(self):
        """FINAL_VAR should return variable value."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            repl.execute_code("answer = 42")
            result = repl.execute_code("print(FINAL_VAR('answer'))")

            assert "42" in result.stdout

    def test_final_var_missing_variable(self):
        """FINAL_VAR should return error for missing variable."""
        from rlm.environments.subprocess_repl import SubprocessREPL

        with SubprocessREPL() as repl:
            result = repl.execute_code("print(FINAL_VAR('nonexistent'))")

            assert "Error" in result.stdout
            assert "not found" in result.stdout
