"""Abstract base class for sandbox strategy implementations."""

from abc import ABC, abstractmethod

from rlm.environments.sandbox.permissions import SandboxCapabilities, SandboxPermissions


class SandboxStrategy(ABC):
    """Abstract base class for platform-specific sandbox implementations.

    Each sandbox strategy must:
    1. Report availability via is_available()
    2. Declare capabilities via capabilities()
    3. Validate permissions via validate_permissions()
    4. Wrap commands via wrap_command()

    The validate_permissions() method should raise SandboxCapabilityError
    if the requested permissions cannot be enforced by this strategy.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the human-readable name of this sandbox strategy."""
        pass

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this sandbox strategy can be used on the current system.

        Returns:
            True if all required tools/features are available.
        """
        pass

    @classmethod
    @abstractmethod
    def get_missing_dependency_message(cls) -> str:
        """Return instructions for installing missing dependencies.

        Called when is_available() returns False to provide helpful
        error messages to users.
        """
        pass

    @classmethod
    @abstractmethod
    def capabilities(cls) -> SandboxCapabilities:
        """Return the capabilities this strategy can enforce.

        Used by validate_permissions() to check if requested
        permissions are achievable.
        """
        pass

    @abstractmethod
    def validate_permissions(self, permissions: SandboxPermissions) -> None:
        """Validate that permissions can be enforced by this strategy.

        Args:
            permissions: The requested sandbox permissions.

        Raises:
            SandboxCapabilityError: If any permission cannot be enforced.
        """
        pass

    @abstractmethod
    def wrap_command(
        self,
        cmd: list[str],
        temp_dir: str,
        venv_path: str,
        permissions: SandboxPermissions,
    ) -> list[str]:
        """Wrap a command with sandbox enforcement.

        Args:
            cmd: The command to wrap (e.g., ["/path/to/python", "-c", "..."])
            temp_dir: Path to temp directory (always readable/writable)
            venv_path: Path to venv directory (always readable)
            permissions: The sandbox permissions to enforce

        Returns:
            The wrapped command with sandbox enforcement prepended.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
