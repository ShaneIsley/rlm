#!/usr/bin/env python3
"""
SubprocessREPL Demo - Direct Sandbox Usage

This example demonstrates using SubprocessREPL directly (without an LLM)
for secure, isolated Python code execution. Shows:
- UV-managed virtual environments (~20ms creation)
- macOS sandbox-exec isolation (network, filesystem)
- Linux bubblewrap isolation (if available)
- Automatic package installation with approval

Requirements:
    - uv: curl -LsSf https://astral.sh/uv/install.sh | sh

Platform Notes:
    - macOS: Full sandbox (network + filesystem) via sandbox-exec
    - Linux: Requires bubblewrap (bwrap) for filesystem isolation
    - Linux without bwrap: Only process isolation, no filesystem sandbox

Usage:
    python examples/subprocess_repl_demo.py
"""

import platform

from rlm.environments import SubprocessREPL


def basic_execution():
    """Basic code execution in sandboxed environment."""
    print("=" * 60)
    print("Basic Execution")
    print("=" * 60)

    with SubprocessREPL(sandbox=True, timeout=10.0) as repl:
        # Simple calculation
        result = repl.execute_code("x = 2 + 2; print(f'Result: {x}')")
        print(f"stdout: {result.stdout}")
        print(f"locals: {result.locals}")


def demonstrate_isolation():
    """Show what the sandbox blocks."""
    print("\n" + "=" * 60)
    print("Sandbox Isolation Demo")
    print("=" * 60)
    print(f"Platform: {platform.system()}")
    if platform.system() == "Linux":
        import shutil
        if not shutil.which("bwrap"):
            print("Note: bubblewrap not installed - filesystem sandbox disabled")
    print()

    with SubprocessREPL(sandbox=True, timeout=10.0) as repl:
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

        # Reading sensitive files is blocked
        result = repl.execute_code("""
try:
    with open("/etc/hosts", "r") as f:
        print("Read /etc/hosts: ALLOWED (unexpected!)")
except Exception as e:
    print(f"Read /etc/hosts: BLOCKED ({type(e).__name__})")
""")
        print(f"File read test: {result.stdout.strip()}")

        # Writing outside temp_dir is blocked
        result = repl.execute_code("""
try:
    with open("/tmp/escape_test.txt", "w") as f:
        f.write("test")
    print("Write /tmp: ALLOWED (unexpected!)")
except Exception as e:
    print(f"Write /tmp: BLOCKED ({type(e).__name__})")
""")
        print(f"File write test: {result.stdout.strip()}")

        # Writing inside temp_dir works
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

    with SubprocessREPL(sandbox=True, persistent=True) as repl:
        repl.execute_code("data = [1, 2, 3]")
        repl.execute_code("data.append(4)")
        result = repl.execute_code("print(f'Final data: {data}')")
        print(f"Persistent state: {result.stdout.strip()}")


def package_installation():
    """Auto-install packages with approval."""
    print("\n" + "=" * 60)
    print("Package Installation (auto-approve)")
    print("=" * 60)

    # auto_approve_packages=True skips the interactive prompt
    with SubprocessREPL(
        sandbox=True,
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

    with SubprocessREPL(sandbox=True) as repl:
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

    with SubprocessREPL(sandbox=True) as repl:
        # Run a few executions
        for i in range(3):
            repl.execute_code(f"x = {i} * 2")

        # Get overhead summary
        summary = repl.get_overhead_summary()
        print(f"Venv creation: {summary['venv_creation_ms']:.1f}ms")
        print(f"Executions: {summary['execution_count']}")
        print(f"Avg overhead/exec: {summary['avg_overhead_per_execution_ms']:.1f}ms")
        print(f"Overhead ratio: {summary['overhead_percentage']:.1f}%")


def compare_sandbox_modes():
    """Compare sandboxed vs non-sandboxed execution."""
    print("\n" + "=" * 60)
    print("Sandbox Mode Comparison")
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

    # With sandbox
    with SubprocessREPL(sandbox=True, timeout=10.0) as repl:
        result = repl.execute_code(code)
        print(f"sandbox=True:  {result.stdout.strip()}")

    # Without sandbox
    with SubprocessREPL(sandbox=False, timeout=10.0) as repl:
        result = repl.execute_code(code)
        print(f"sandbox=False: {result.stdout.strip()}")


if __name__ == "__main__":
    print("SubprocessREPL Demo - Direct Sandbox Usage")
    print("Requires: uv (https://astral.sh/uv)\n")

    basic_execution()
    demonstrate_isolation()
    state_persistence()
    context_loading()
    overhead_tracking()
    compare_sandbox_modes()

    # Uncomment to test package installation (requires network when sandbox=False)
    # package_installation()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
