"""Sandbox permission configuration with deny-all defaults."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SandboxCapabilities:
    """Capabilities that a sandbox strategy can enforce.

    Each sandbox implementation declares what it can enforce.
    Used to validate that requested permissions are achievable.
    """

    # Filesystem capabilities
    filesystem_read_allowlist: bool = True
    filesystem_write_allowlist: bool = True

    # Network capabilities
    network_deny_all: bool = True
    network_allow_all: bool = True
    network_host_allowlist: bool = False  # Fine-grained host filtering

    # Process capabilities
    process_isolation: bool = True
    subprocess_deny: bool = False  # Can prevent subprocess spawning


@dataclass(frozen=True)
class SandboxPermissions:
    """Immutable sandbox permission configuration.

    Default is deny-all: no network, no filesystem access outside temp_dir/venv.
    Use builder methods to add specific permissions.

    Example:
        # Maximum isolation (default)
        perms = SandboxPermissions()

        # Allow specific network hosts (macOS only)
        perms = SandboxPermissions(allow_network_hosts=("api.openai.com:443",))

        # Allow all network (both platforms)
        perms = SandboxPermissions(allow_network=True)

        # Allow reading specific paths
        perms = SandboxPermissions(allow_read_paths=("/data/datasets",))

        # Builder pattern
        perms = (
            SandboxPermissions()
            .with_network_hosts("api.openai.com:443", "api.anthropic.com:443")
            .with_read_paths("/data")
        )
    """

    # Network permissions (default: deny all)
    # allow_network=True means allow all network access
    # allow_network=False with allow_network_hosts=() means deny all
    # allow_network=False with allow_network_hosts=(...) means allow only those hosts
    allow_network: bool = False
    allow_network_hosts: tuple[str, ...] = ()  # e.g., ("api.openai.com:443",)
    allow_localhost: bool = False

    # Filesystem permissions (default: only temp_dir writable, only venv readable)
    # These are ADDITIONAL paths beyond the always-allowed temp_dir and venv
    allow_read_paths: tuple[str, ...] = ()
    allow_write_paths: tuple[str, ...] = ()

    # Process permissions (default: allow - needed for Python)
    allow_subprocess: bool = True  # Required for Python to work

    def __post_init__(self):
        """Validate permission consistency."""
        # If allow_network is True, allow_network_hosts should be empty
        # (allow_network=True means allow ALL, hosts list is ignored)
        if self.allow_network and self.allow_network_hosts:
            raise ValueError(
                "Cannot specify both allow_network=True and allow_network_hosts. "
                "Use allow_network=True for all network access, or "
                "allow_network_hosts=(...) for specific hosts only."
            )

    @property
    def has_network_host_allowlist(self) -> bool:
        """Check if fine-grained network host filtering is requested."""
        return not self.allow_network and len(self.allow_network_hosts) > 0

    @property
    def network_mode(self) -> str:
        """Return human-readable network mode."""
        if self.allow_network:
            return "allow-all"
        elif self.allow_network_hosts:
            return f"allowlist: {', '.join(self.allow_network_hosts)}"
        else:
            return "deny-all"

    # Builder methods for fluent API

    def with_network(self, allow: bool = True) -> "SandboxPermissions":
        """Return new permissions with network access enabled/disabled."""
        return replace(self, allow_network=allow, allow_network_hosts=())

    def with_network_hosts(self, *hosts: str) -> "SandboxPermissions":
        """Return new permissions with specific network hosts allowed.

        Args:
            *hosts: Host specifications like "api.example.com:443"

        Note:
            This requires network_host_allowlist capability (macOS only).
            On Linux, use with_network(True) or with_network(False).
        """
        return replace(
            self,
            allow_network=False,
            allow_network_hosts=self.allow_network_hosts + hosts,
        )

    def with_localhost(self, allow: bool = True) -> "SandboxPermissions":
        """Return new permissions with localhost access enabled/disabled."""
        return replace(self, allow_localhost=allow)

    def with_read_paths(self, *paths: str) -> "SandboxPermissions":
        """Return new permissions with additional read paths."""
        return replace(self, allow_read_paths=self.allow_read_paths + paths)

    def with_write_paths(self, *paths: str) -> "SandboxPermissions":
        """Return new permissions with additional write paths."""
        return replace(self, allow_write_paths=self.allow_write_paths + paths)

    def with_subprocess(self, allow: bool) -> "SandboxPermissions":
        """Return new permissions with subprocess spawning enabled/disabled."""
        return replace(self, allow_subprocess=allow)
