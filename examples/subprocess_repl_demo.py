#!/usr/bin/env python3
"""
SubprocessREPL Demo - Direct Sandbox Usage

This example demonstrates using SubprocessREPL directly (without an LLM)
for secure, isolated Python code execution. Shows:
- UV-managed virtual environments (~20ms creation)
- Mandatory sandboxing with configurable permissions
- macOS sandbox-exec isolation (network, filesystem)
- Linux bubblewrap isolation
- Automatic package installation with approval

Requirements:
    - uv: curl -LsSf https://astral.sh/uv/install.sh | sh
    - macOS: sandbox-exec (built-in)
    - Linux: bubblewrap (sudo apt install bubblewrap)

Usage:
    python examples/subprocess_repl_demo.py
"""

import platform

from rlm.environments import SubprocessREPL
from rlm.environments.sandbox import SandboxPermissions, SandboxProfiles


def basic_execution():
    """Basic code execution in sandboxed environment."""
    print("=" * 60)
    print("Basic Execution (with NETWORK_ALL permissions)")
    print("=" * 60)

    # Use NETWORK_ALL to avoid bwrap loopback issues in containers
    with SubprocessREPL(permissions=SandboxProfiles.NETWORK_ALL, timeout=10.0) as repl:
        print(f"Sandbox strategy: {repl.sandbox_strategy}")
        print(f"Permissions: {repl.permissions.network_mode}")

        # Simple calculation
        result = repl.execute_code("x = 2 + 2; print(f'Result: {x}')")
        print(f"stdout: {result.stdout}")
        print(f"locals: {result.locals}")


def demonstrate_isolation():
    """Show what the sandbox blocks."""
    print("\n" + "=" * 60)
    print("Sandbox Isolation Demo (STRICT permissions)")
    print("=" * 60)
    print(f"Platform: {platform.system()}")

    # STRICT = deny all network and filesystem
    # Note: In containerized environments, this may fail due to nested namespace issues
    try:
        with SubprocessREPL(permissions=SandboxProfiles.STRICT, timeout=10.0) as repl:
            print(f"Permissions: {repl.permissions.network_mode}")

            # Network is blocked
            result = repl.execute_code("""
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("8.8.8.8", 53))
    print("Network: ALLOWED (unexpected!)")
except Exception as e:
    print(f"Network: BLOCKED ({type(e).__name__})")
""")
            print(f"Network test: {result.stdout.strip()}")

            # Writing inside temp_dir works
            result = repl.execute_code("""
with open("local_file.txt", "w") as f:
    f.write("hello sandbox")
with open("local_file.txt", "r") as f:
    print(f"Write temp_dir: OK - {f.read()}")
""")
            print(f"Local write test: {result.stdout.strip()}")
    except Exception as e:
        print(f"STRICT mode failed (common in containers): {e}")
        print("Trying with NETWORK_ALL instead...")

        with SubprocessREPL(permissions=SandboxProfiles.NETWORK_ALL, timeout=10.0) as repl:
            result = repl.execute_code("""
with open("local_file.txt", "w") as f:
    f.write("hello sandbox")
with open("local_file.txt", "r") as f:
    print(f"Write temp_dir: OK - {f.read()}")
""")
            print(f"Local write test: {result.stdout.strip()}")


def state_persistence():
    """Variables persist across executions."""
    print("\n" + "=" * 60)
    print("State Persistence")
    print("=" * 60)

    with SubprocessREPL(permissions=SandboxProfiles.NETWORK_ALL, persistent=True) as repl:
        repl.execute_code("data = [1, 2, 3]")
        repl.execute_code("data.append(4)")
        result = repl.execute_code("print(f'Final data: {data}')")
        print(f"Persistent state: {result.stdout.strip()}")


def package_installation():
    """Auto-install packages with approval."""
    print("\n" + "=" * 60)
    print("Package Installation (auto-approve)")
    print("=" * 60)

    # Need NETWORK_ALL to download packages
    with SubprocessREPL(
        permissions=SandboxProfiles.NETWORK_ALL,
        auto_approve_packages=True,
        timeout=30.0,
    ) as repl:
        result = repl.execute_code("""
import requests
print(f"requests version: {requests.__version__}")
""")
        if "requests version" in result.stdout:
            print(f"Package installed: {result.stdout.strip()}")
        else:
            print(f"Install output: {result.stderr}")


def context_loading():
    """Load context data for the subprocess to use."""
    print("\n" + "=" * 60)
    print("Context Loading")
    print("=" * 60)

    with SubprocessREPL(permissions=SandboxProfiles.NETWORK_ALL) as repl:
        # Load dict context
        repl.load_context({"users": ["alice", "bob"], "count": 2})

        result = repl.execute_code("""
print(f"Users: {context['users']}")
print(f"Count: {context['count']}")
""")
        print(f"Context access:\n{result.stdout}")


def overhead_tracking():
    """Track execution overhead."""
    print("\n" + "=" * 60)
    print("Overhead Tracking")
    print("=" * 60)

    with SubprocessREPL(permissions=SandboxProfiles.NETWORK_ALL) as repl:
        # Run a few executions
        for i in range(3):
            repl.execute_code(f"x = {i} * 2")

        # Get overhead summary
        summary = repl.get_overhead_summary()
        print(f"Venv creation: {summary['venv_creation_ms']:.1f}ms")
        print(f"Executions: {summary['execution_count']}")
        print(f"Avg overhead/exec: {summary['avg_overhead_per_execution_ms']:.1f}ms")
        print(f"Overhead ratio: {summary['overhead_percentage']:.1f}%")


def compare_permission_modes():
    """Compare different permission configurations."""
    print("\n" + "=" * 60)
    print("Permission Mode Comparison")
    print("=" * 60)

    code = """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("8.8.8.8", 53))
    s.close()
    result = "Network accessible"
except Exception as e:
    result = f"Network blocked: {type(e).__name__}"
print(result)
"""

    # With network allowed
    print("\nTesting NETWORK_ALL permissions:")
    with SubprocessREPL(permissions=SandboxProfiles.NETWORK_ALL, timeout=10.0) as repl:
        result = repl.execute_code(code)
        print(f"  Result: {result.stdout.strip()}")

    # Custom permissions with specific read paths
    print("\nTesting custom permissions (network + read /tmp):")
    custom_perms = SandboxPermissions(allow_network=True, allow_read_paths=("/tmp",))
    with SubprocessREPL(permissions=custom_perms, timeout=10.0) as repl:
        print(f"  Network mode: {repl.permissions.network_mode}")
        print(f"  Read paths: {repl.permissions.allow_read_paths}")


def show_sandbox_info():
    """Display sandbox strategy information."""
    print("\n" + "=" * 60)
    print("Sandbox Strategy Information")
    print("=" * 60)

    from rlm.environments.sandbox import get_sandbox_info, get_sandbox_strategy

    info = get_sandbox_info()
    print("Available strategies:")
    for name, available in info.items():
        status = "✓ available" if available else "✗ not available"
        print(f"  {name}: {status}")

    strategy = get_sandbox_strategy()
    print(f"\nActive strategy: {strategy.name()}")
    print(f"Capabilities:")
    caps = strategy.capabilities()
    print(f"  - Network deny-all: {caps.network_deny_all}")
    print(f"  - Network host allowlist: {caps.network_host_allowlist}")
    print(f"  - Filesystem isolation: {caps.filesystem_read_allowlist}")


if __name__ == "__main__":
    print("SubprocessREPL Demo - Mandatory Sandbox with Configurable Permissions")
    print("Requires: uv (https://astral.sh/uv)")
    print("Requires: sandbox-exec (macOS) or bubblewrap (Linux)\n")

    show_sandbox_info()
    basic_execution()
    demonstrate_isolation()
    state_persistence()
    context_loading()
    overhead_tracking()
    compare_permission_modes()

    # Uncomment to test package installation (requires network)
    # package_installation()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
