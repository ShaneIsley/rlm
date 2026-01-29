"""Tests for the sandbox module."""

import platform

import pytest

from rlm.environments.sandbox import (
    SandboxCapabilities,
    SandboxPermissions,
    SandboxProfiles,
    get_sandbox_info,
    get_sandbox_strategy,
    is_sandbox_available,
)


class TestSandboxPermissions:
    """Tests for SandboxPermissions dataclass."""

    def test_default_is_deny_all(self):
        """Default permissions should deny all network and filesystem."""
        perms = SandboxPermissions()
        assert perms.allow_network is False
        assert perms.allow_network_hosts == ()
        assert perms.allow_localhost is False
        assert perms.allow_read_paths == ()
        assert perms.allow_write_paths == ()

    def test_network_mode_deny_all(self):
        """Network mode should be 'deny-all' by default."""
        perms = SandboxPermissions()
        assert perms.network_mode == "deny-all"

    def test_network_mode_allow_all(self):
        """Network mode should be 'allow-all' when enabled."""
        perms = SandboxPermissions(allow_network=True)
        assert perms.network_mode == "allow-all"

    def test_network_mode_allowlist(self):
        """Network mode should show allowlist when hosts specified."""
        perms = SandboxPermissions(allow_network_hosts=("api.example.com:443",))
        assert "allowlist" in perms.network_mode
        assert "api.example.com:443" in perms.network_mode

    def test_has_network_host_allowlist(self):
        """has_network_host_allowlist should detect fine-grained filtering."""
        perms = SandboxPermissions()
        assert perms.has_network_host_allowlist is False

        perms = SandboxPermissions(allow_network=True)
        assert perms.has_network_host_allowlist is False

        perms = SandboxPermissions(allow_network_hosts=("api.example.com:443",))
        assert perms.has_network_host_allowlist is True

    def test_cannot_have_both_allow_network_and_hosts(self):
        """Should raise if both allow_network and allow_network_hosts set."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            SandboxPermissions(
                allow_network=True,
                allow_network_hosts=("api.example.com:443",),
            )

    def test_builder_with_network(self):
        """with_network should return new permissions with network enabled."""
        perms = SandboxPermissions()
        new_perms = perms.with_network(True)

        assert perms.allow_network is False  # Original unchanged
        assert new_perms.allow_network is True

    def test_builder_with_network_hosts(self):
        """with_network_hosts should add hosts to allowlist."""
        perms = SandboxPermissions()
        new_perms = perms.with_network_hosts("api.example.com:443")

        assert perms.allow_network_hosts == ()  # Original unchanged
        assert new_perms.allow_network_hosts == ("api.example.com:443",)

    def test_builder_with_read_paths(self):
        """with_read_paths should add read paths."""
        perms = SandboxPermissions()
        new_perms = perms.with_read_paths("/data")

        assert perms.allow_read_paths == ()  # Original unchanged
        assert new_perms.allow_read_paths == ("/data",)

    def test_builder_chaining(self):
        """Builder methods should be chainable."""
        perms = (
            SandboxPermissions()
            .with_network_hosts("api.example.com:443")
            .with_read_paths("/data")
            .with_localhost(True)
        )

        assert perms.allow_network_hosts == ("api.example.com:443",)
        assert perms.allow_read_paths == ("/data",)
        assert perms.allow_localhost is True

    def test_immutable(self):
        """Permissions should be immutable (frozen dataclass)."""
        perms = SandboxPermissions()
        with pytest.raises(AttributeError):
            perms.allow_network = True


class TestSandboxProfiles:
    """Tests for predefined SandboxProfiles."""

    def test_strict_is_deny_all(self):
        """STRICT profile should deny all."""
        assert SandboxProfiles.STRICT.allow_network is False
        assert SandboxProfiles.STRICT.allow_network_hosts == ()

    def test_network_https_allows_443(self):
        """NETWORK_HTTPS should allow port 443."""
        assert SandboxProfiles.NETWORK_HTTPS.allow_network_hosts == ("*:443",)

    def test_network_all_allows_all(self):
        """NETWORK_ALL should allow all network."""
        assert SandboxProfiles.NETWORK_ALL.allow_network is True

    def test_custom_profile(self):
        """custom() should create profile with specified options."""
        perms = SandboxProfiles.custom(
            allow_network=True,
            allow_read_paths=("/data",),
        )
        assert perms.allow_network is True
        assert perms.allow_read_paths == ("/data",)


class TestSandboxCapabilities:
    """Tests for SandboxCapabilities."""

    def test_default_capabilities(self):
        """Default capabilities should have reasonable values."""
        caps = SandboxCapabilities()
        assert caps.filesystem_read_allowlist is True
        assert caps.network_deny_all is True
        assert caps.network_host_allowlist is False  # Not all platforms support


class TestSandboxRegistry:
    """Tests for sandbox registry functions."""

    def test_get_sandbox_info(self):
        """get_sandbox_info should return dict of strategy availability."""
        info = get_sandbox_info()
        assert isinstance(info, dict)
        assert "macOS sandbox-exec" in info
        assert "Linux bubblewrap" in info

    def test_is_sandbox_available(self):
        """is_sandbox_available should return bool."""
        result = is_sandbox_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not is_sandbox_available(), reason="No sandbox available")
    def test_get_sandbox_strategy(self):
        """get_sandbox_strategy should return strategy when available."""
        strategy = get_sandbox_strategy()
        assert strategy is not None
        assert hasattr(strategy, "wrap_command")
        assert hasattr(strategy, "validate_permissions")

    def test_get_sandbox_strategy_unavailable(self):
        """get_sandbox_strategy should raise when no sandbox available."""
        from unittest.mock import patch

        from rlm.core.exceptions import SandboxUnavailableError
        from rlm.environments.sandbox.linux import LinuxBwrap
        from rlm.environments.sandbox.macos import MacOSSandboxExec

        with (
            patch.object(MacOSSandboxExec, "is_available", return_value=False),
            patch.object(LinuxBwrap, "is_available", return_value=False),
        ):
            with pytest.raises(SandboxUnavailableError):
                get_sandbox_strategy()


class TestLinuxBwrap:
    """Tests for Linux bubblewrap strategy."""

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
    def test_capabilities(self):
        """Linux bwrap should not support network host allowlist."""
        from rlm.environments.sandbox.linux import LinuxBwrap

        caps = LinuxBwrap.capabilities()
        assert caps.network_host_allowlist is False
        assert caps.network_deny_all is True

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
    def test_validate_rejects_host_allowlist(self):
        """Linux bwrap should reject network host allowlist permissions."""
        from rlm.core.exceptions import SandboxCapabilityError
        from rlm.environments.sandbox.linux import LinuxBwrap

        strategy = LinuxBwrap()
        perms = SandboxPermissions(allow_network_hosts=("api.example.com:443",))

        with pytest.raises(SandboxCapabilityError):
            strategy.validate_permissions(perms)

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
    def test_validate_accepts_allow_network(self):
        """Linux bwrap should accept allow_network=True."""
        from rlm.environments.sandbox.linux import LinuxBwrap

        strategy = LinuxBwrap()
        perms = SandboxPermissions(allow_network=True)

        # Should not raise
        strategy.validate_permissions(perms)


class TestMacOSSandboxExec:
    """Tests for macOS sandbox-exec strategy."""

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_capabilities(self):
        """macOS sandbox-exec should support network host allowlist."""
        from rlm.environments.sandbox.macos import MacOSSandboxExec

        caps = MacOSSandboxExec.capabilities()
        assert caps.network_host_allowlist is True
        assert caps.network_deny_all is True

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_validate_accepts_host_allowlist(self):
        """macOS sandbox-exec should accept network host allowlist."""
        from rlm.environments.sandbox.macos import MacOSSandboxExec

        strategy = MacOSSandboxExec()
        perms = SandboxPermissions(allow_network_hosts=("api.example.com:443",))

        # Should not raise
        strategy.validate_permissions(perms)
