"""Sandbox strategy module for secure code execution.

This module provides platform-agnostic sandbox configuration with
deny-all defaults and explicit permission grants.

Example:
    from rlm.environments import SubprocessREPL
    from rlm.environments.sandbox import SandboxPermissions, SandboxProfiles

    # Maximum isolation (default)
    with SubprocessREPL() as repl:
        result = repl.execute_code("print('isolated')")

    # Allow HTTPS network access
    with SubprocessREPL(permissions=SandboxProfiles.NETWORK_HTTPS) as repl:
        result = repl.execute_code("import requests; requests.get('https://...')")

    # Custom permissions with builder pattern
    perms = (
        SandboxPermissions()
        .with_network_hosts("api.openai.com:443")
        .with_read_paths("/data/datasets")
    )
    with SubprocessREPL(permissions=perms) as repl:
        ...

Supported Platforms:
    - macOS: sandbox-exec (Seatbelt) - full network host filtering support
    - Linux: bubblewrap (bwrap) - network is all-or-nothing

Note:
    On Linux, network host allowlists are not supported. Use
    allow_network=True or allow_network=False instead.
"""

from rlm.environments.sandbox.base import SandboxStrategy
from rlm.environments.sandbox.linux import LinuxBwrap
from rlm.environments.sandbox.macos import MacOSSandboxExec
from rlm.environments.sandbox.permissions import SandboxCapabilities, SandboxPermissions
from rlm.environments.sandbox.profiles import SandboxProfiles
from rlm.environments.sandbox.registry import (
    get_available_strategies,
    get_sandbox_info,
    get_sandbox_strategy,
    is_sandbox_available,
)

__all__ = [
    # Permissions and profiles
    "SandboxPermissions",
    "SandboxCapabilities",
    "SandboxProfiles",
    # Strategy base and implementations
    "SandboxStrategy",
    "MacOSSandboxExec",
    "LinuxBwrap",
    # Registry functions
    "get_sandbox_strategy",
    "get_available_strategies",
    "is_sandbox_available",
    "get_sandbox_info",
]
