"""Predefined sandbox permission profiles for common use cases."""

from rlm.environments.sandbox.permissions import SandboxPermissions


class SandboxProfiles:
    """Predefined sandbox configurations for common use cases.

    All profiles use deny-all as the base and explicitly allow only
    what is needed for the use case.

    Example:
        from rlm.environments import SubprocessREPL
        from rlm.environments.sandbox import SandboxProfiles

        # Maximum isolation
        with SubprocessREPL(permissions=SandboxProfiles.STRICT) as repl:
            ...

        # Allow HTTPS API calls
        with SubprocessREPL(permissions=SandboxProfiles.NETWORK_HTTPS) as repl:
            ...
    """

    # Maximum isolation - no network, no filesystem beyond temp_dir/venv
    STRICT = SandboxPermissions()

    # Allow all outbound HTTPS traffic (port 443)
    # Note: Only enforced on macOS. On Linux, falls back to allow-all network.
    NETWORK_HTTPS = SandboxPermissions(
        allow_network_hosts=("*:443",),
    )

    # Allow all network access
    NETWORK_ALL = SandboxPermissions(
        allow_network=True,
    )

    # Allow localhost connections (e.g., for local databases)
    LOCALHOST_ONLY = SandboxPermissions(
        allow_localhost=True,
    )

    # Read-only access to common data directories
    DATA_READONLY = SandboxPermissions(
        allow_read_paths=(
            "/data",
            "/datasets",
        ),
    )

    # Research profile: network + data access
    # Useful for ML/data science workloads that need to fetch data
    # and read local datasets
    RESEARCH = SandboxPermissions(
        allow_network=True,
        allow_read_paths=(
            "/data",
            "/datasets",
        ),
    )

    @classmethod
    def custom(
        cls,
        *,
        allow_network: bool = False,
        allow_network_hosts: tuple[str, ...] = (),
        allow_localhost: bool = False,
        allow_read_paths: tuple[str, ...] = (),
        allow_write_paths: tuple[str, ...] = (),
    ) -> SandboxPermissions:
        """Create a custom permission profile.

        This is a convenience method that provides named parameters
        for clarity when creating custom profiles.

        Args:
            allow_network: Allow all network access
            allow_network_hosts: Specific hosts to allow (macOS only)
            allow_localhost: Allow localhost connections
            allow_read_paths: Additional paths to allow reading
            allow_write_paths: Additional paths to allow writing

        Returns:
            SandboxPermissions with the specified configuration
        """
        return SandboxPermissions(
            allow_network=allow_network,
            allow_network_hosts=allow_network_hosts,
            allow_localhost=allow_localhost,
            allow_read_paths=allow_read_paths,
            allow_write_paths=allow_write_paths,
        )
