"""Linux bubblewrap (bwrap) sandbox strategy."""

import os
import platform
import shutil

from rlm.core.exceptions import SandboxCapabilityError
from rlm.environments.sandbox.base import SandboxStrategy
from rlm.environments.sandbox.permissions import SandboxCapabilities, SandboxPermissions


class LinuxBwrap(SandboxStrategy):
    """Linux bubblewrap (bwrap) sandbox implementation.

    Uses bubblewrap for filesystem and process isolation via Linux namespaces.

    Limitations:
        - Network is all-or-nothing (--unshare-net or full access)
        - Cannot filter by specific network hosts
        - Requires bubblewrap to be installed

    Requirements:
        - Linux
        - bubblewrap (bwrap) installed

    Install bubblewrap:
        - Debian/Ubuntu: sudo apt install bubblewrap
        - Fedora/RHEL: sudo dnf install bubblewrap
        - Arch: sudo pacman -S bubblewrap
    """

    @classmethod
    def name(cls) -> str:
        return "Linux bubblewrap"

    @classmethod
    def is_available(cls) -> bool:
        return platform.system() == "Linux" and shutil.which("bwrap") is not None

    @classmethod
    def get_missing_dependency_message(cls) -> str:
        if platform.system() != "Linux":
            return "Linux bubblewrap requires Linux"
        return (
            "Linux requires bubblewrap (bwrap) for sandboxing.\n"
            "Install with:\n"
            "  Debian/Ubuntu: sudo apt install bubblewrap\n"
            "  Fedora/RHEL:   sudo dnf install bubblewrap\n"
            "  Arch:          sudo pacman -S bubblewrap"
        )

    @classmethod
    def capabilities(cls) -> SandboxCapabilities:
        return SandboxCapabilities(
            filesystem_read_allowlist=True,
            filesystem_write_allowlist=True,
            network_deny_all=True,
            network_allow_all=True,
            network_host_allowlist=False,  # bwrap cannot filter by host
            process_isolation=True,
            subprocess_deny=False,  # bwrap doesn't prevent subprocess spawning
        )

    def validate_permissions(self, permissions: SandboxPermissions) -> None:
        """Validate permissions - bwrap cannot do fine-grained network filtering."""
        if permissions.has_network_host_allowlist:
            raise SandboxCapabilityError(
                strategy_name=self.name(),
                capability="network host allowlist",
                suggestion=(
                    "Use allow_network=True for full network access, "
                    "or allow_network=False for no network access."
                ),
            )

    def wrap_command(
        self,
        cmd: list[str],
        temp_dir: str,
        venv_path: str,
        permissions: SandboxPermissions,
    ) -> list[str]:
        """Wrap command with bubblewrap enforcement."""
        bwrap_cmd = ["bwrap"]

        # Minimal filesystem - read-only system directories
        bwrap_cmd.extend(["--ro-bind", "/usr", "/usr"])
        bwrap_cmd.extend(["--ro-bind", "/lib", "/lib"])
        bwrap_cmd.extend(["--ro-bind", "/bin", "/bin"])
        bwrap_cmd.extend(["--ro-bind", "/sbin", "/sbin"])

        # Add /lib64 if it exists (common on 64-bit systems)
        if os.path.exists("/lib64"):
            bwrap_cmd.extend(["--ro-bind", "/lib64", "/lib64"])

        # Add /etc read-only for SSL certificates and DNS resolution
        # This is more permissive than macOS but necessary for many operations
        bwrap_cmd.extend(["--ro-bind", "/etc", "/etc"])

        # Required paths - venv read-only, temp_dir read-write
        real_venv_path = os.path.realpath(venv_path)
        real_temp_dir = os.path.realpath(temp_dir)

        bwrap_cmd.extend(["--ro-bind", real_venv_path, real_venv_path])
        if venv_path != real_venv_path:
            bwrap_cmd.extend(["--ro-bind", venv_path, venv_path])

        bwrap_cmd.extend(["--bind", real_temp_dir, real_temp_dir])
        if temp_dir != real_temp_dir:
            bwrap_cmd.extend(["--bind", temp_dir, temp_dir])

        # Additional read paths from permissions
        for path in permissions.allow_read_paths:
            real_path = os.path.realpath(os.path.expanduser(path))
            if os.path.exists(real_path):
                bwrap_cmd.extend(["--ro-bind", real_path, real_path])
                if path != real_path:
                    bwrap_cmd.extend(["--ro-bind", path, path])

        # Additional write paths from permissions
        for path in permissions.allow_write_paths:
            real_path = os.path.realpath(os.path.expanduser(path))
            if os.path.exists(real_path):
                bwrap_cmd.extend(["--bind", real_path, real_path])
                if path != real_path:
                    bwrap_cmd.extend(["--bind", path, path])

        # Network isolation (all-or-nothing)
        if not permissions.allow_network:
            bwrap_cmd.append("--unshare-net")

        # Process isolation
        bwrap_cmd.append("--unshare-pid")

        # Die with parent process (cleanup on parent exit)
        bwrap_cmd.append("--die-with-parent")

        # Provide a minimal /proc for Python to function
        bwrap_cmd.extend(["--proc", "/proc"])

        # Provide a minimal /dev
        bwrap_cmd.extend(["--dev", "/dev"])

        # End of bwrap options, start of command
        bwrap_cmd.append("--")

        return bwrap_cmd + cmd
