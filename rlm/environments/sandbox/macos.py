"""macOS sandbox-exec (Seatbelt) sandbox strategy."""

import os
import platform
from pathlib import Path

from rlm.core.exceptions import SandboxCapabilityError
from rlm.environments.sandbox.base import SandboxStrategy
from rlm.environments.sandbox.permissions import SandboxCapabilities, SandboxPermissions


class MacOSSandboxExec(SandboxStrategy):
    """macOS sandbox-exec (Seatbelt) implementation.

    Uses Apple's Sandbox Profile Language (SBPL) to enforce permissions.
    Supports fine-grained network host filtering, filesystem paths, and more.

    Requirements:
        - macOS (Darwin)
        - /usr/bin/sandbox-exec (built-in on macOS)
    """

    SANDBOX_EXEC_PATH = "/usr/bin/sandbox-exec"

    @classmethod
    def name(cls) -> str:
        return "macOS sandbox-exec"

    @classmethod
    def is_available(cls) -> bool:
        return (
            platform.system() == "Darwin"
            and os.path.exists(cls.SANDBOX_EXEC_PATH)
        )

    @classmethod
    def get_missing_dependency_message(cls) -> str:
        if platform.system() != "Darwin":
            return "macOS sandbox-exec requires macOS (Darwin)"
        return (
            "macOS sandbox-exec not found at /usr/bin/sandbox-exec. "
            "This is built-in on macOS and should not be missing."
        )

    @classmethod
    def capabilities(cls) -> SandboxCapabilities:
        return SandboxCapabilities(
            filesystem_read_allowlist=True,
            filesystem_write_allowlist=True,
            network_deny_all=True,
            network_allow_all=True,
            network_host_allowlist=True,  # macOS supports fine-grained network
            process_isolation=True,
            subprocess_deny=True,
        )

    def validate_permissions(self, permissions: SandboxPermissions) -> None:
        """Validate permissions - macOS supports all permission types."""
        # macOS sandbox-exec supports all permission types
        # No validation failures possible for supported permissions
        pass

    def wrap_command(
        self,
        cmd: list[str],
        temp_dir: str,
        venv_path: str,
        permissions: SandboxPermissions,
    ) -> list[str]:
        """Wrap command with sandbox-exec enforcement."""
        profile = self._build_profile(temp_dir, venv_path, permissions)
        profile_path = os.path.join(temp_dir, "sandbox.sb")
        Path(profile_path).write_text(profile)
        return [self.SANDBOX_EXEC_PATH, "-f", profile_path] + cmd

    def _build_profile(
        self,
        temp_dir: str,
        venv_path: str,
        permissions: SandboxPermissions,
    ) -> str:
        """Build SBPL (Sandbox Profile Language) profile from permissions.

        Uses a hybrid approach:
        - Start with (allow default) for Python/dyld compatibility
        - Explicitly deny sensitive operations
        - Allow only specific paths and network destinations
        """
        # Resolve symlinks (macOS /var -> /private/var)
        real_temp_dir = os.path.realpath(temp_dir)
        real_venv_path = os.path.realpath(venv_path)
        home_dir = os.path.expanduser("~")
        real_home_dir = os.path.realpath(home_dir)

        # Find Python interpreter location (may be in ~/.local/share/uv/)
        python_path = os.path.join(venv_path, "bin", "python")
        if os.path.exists(python_path):
            real_python = os.path.realpath(python_path)
            python_install_dir = os.path.dirname(os.path.dirname(real_python))
        else:
            python_install_dir = ""

        rules = [
            "(version 1)",
            "",
            ";; Hybrid approach: allow default then deny specific sensitive paths",
            ";; This ensures Python can run while blocking access to sensitive data",
            "(allow default)",
            "",
        ]

        # Network rules
        rules.append(";; ============ NETWORK ============")
        if permissions.allow_network:
            rules.append(";; Network: allow all (explicit permission)")
        elif permissions.allow_network_hosts:
            rules.append(";; Network: allowlist mode")
            rules.append("(deny network-outbound (remote ip))")
            for host in permissions.allow_network_hosts:
                if host.startswith("*:"):
                    # Wildcard host, specific port
                    port = host[2:]
                    rules.append(f'(allow network-outbound (remote tcp "*:{port}"))')
                elif ":" in host:
                    # Specific host:port
                    rules.append(f'(allow network-outbound (remote tcp "{host}"))')
                else:
                    # Host only, allow any port
                    rules.append(f'(allow network-outbound (remote tcp "{host}:*"))')
            if permissions.allow_localhost:
                rules.append('(allow network-outbound (remote tcp "localhost:*"))')
                rules.append('(allow network-outbound (remote tcp "127.0.0.1:*"))')
        else:
            rules.append(";; Network: deny all")
            rules.append("(deny network-outbound (remote ip))")
            if permissions.allow_localhost:
                rules.append('(allow network-outbound (remote tcp "localhost:*"))')
                rules.append('(allow network-outbound (remote tcp "127.0.0.1:*"))')

        rules.append("")

        # File write rules
        rules.append(";; ============ FILE WRITES ============")
        rules.append(";; Block ALL writes first")
        rules.append("(deny file-write*)")
        rules.append("")
        rules.append(";; Allow writes ONLY to temp directory")
        rules.append(f'(allow file-write* (subpath "{real_temp_dir}"))')
        if temp_dir != real_temp_dir:
            rules.append(f'(allow file-write* (subpath "{temp_dir}"))')

        # Additional write paths
        for path in permissions.allow_write_paths:
            real_path = os.path.realpath(os.path.expanduser(path))
            rules.append(f'(allow file-write* (subpath "{real_path}"))')
            if path != real_path:
                rules.append(f'(allow file-write* (subpath "{path}"))')

        rules.append("")

        # File read rules
        rules.append(";; ============ FILE READS ============")
        rules.append(";; Block reads to sensitive directories")
        rules.append(f'(deny file-read* (subpath "{real_home_dir}"))')
        if home_dir != real_home_dir:
            rules.append(f'(deny file-read* (subpath "{home_dir}"))')
        rules.append('(deny file-read* (subpath "/Users"))')
        rules.append('(deny file-read* (subpath "/home"))')
        rules.append('(deny file-read* (subpath "/etc"))')
        rules.append('(deny file-read* (subpath "/private/etc"))')
        rules.append("")

        # Allow specific /etc files needed for SSL/DNS
        rules.append(";; Allow specific /etc files needed for SSL/DNS")
        rules.append('(allow file-read* (literal "/etc/ssl/cert.pem"))')
        rules.append('(allow file-read* (literal "/etc/ssl/certs"))')
        rules.append('(allow file-read* (literal "/etc/resolv.conf"))')
        rules.append('(allow file-read* (literal "/private/etc/ssl/cert.pem"))')
        rules.append('(allow file-read* (literal "/private/etc/resolv.conf"))')
        rules.append("")

        # Allow our specific paths (overrides the denies above)
        rules.append(";; Allow our specific paths (overrides the denies above)")
        rules.append(f'(allow file-read* (subpath "{real_temp_dir}"))')
        if temp_dir != real_temp_dir:
            rules.append(f'(allow file-read* (subpath "{temp_dir}"))')
        rules.append(f'(allow file-read* (subpath "{real_venv_path}"))')
        if venv_path != real_venv_path:
            rules.append(f'(allow file-read* (subpath "{venv_path}"))')
        if python_install_dir:
            rules.append(f'(allow file-read* (subpath "{python_install_dir}"))')

        # Additional read paths
        for path in permissions.allow_read_paths:
            real_path = os.path.realpath(os.path.expanduser(path))
            rules.append(f'(allow file-read* (subpath "{real_path}"))')
            if path != real_path:
                rules.append(f'(allow file-read* (subpath "{path}"))')

        return "\n".join(rules)
