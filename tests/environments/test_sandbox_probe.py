"""Probe sandbox requirements to identify what paths Python needs."""

import os
import platform

import pytest

from rlm.environments import SubprocessREPL


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
class TestSandboxProbe:
    """Probe to discover sandbox requirements."""

    def test_probe_python_paths(self):
        """Discover what paths Python needs to run."""
        with SubprocessREPL(sandbox=False, timeout=10.0) as repl:
            result = repl.execute_code("""
import sys
import os
import sysconfig

print("=== Python Installation Paths ===")
print(f"sys.executable: {os.path.realpath(sys.executable)}")
print(f"sys.prefix: {os.path.realpath(sys.prefix)}")
print(f"sys.base_prefix: {os.path.realpath(sys.base_prefix)}")
print(f"sys.exec_prefix: {os.path.realpath(sys.exec_prefix)}")

print("\\n=== Sysconfig Paths ===")
for name in sysconfig.get_path_names():
    path = sysconfig.get_path(name)
    if path:
        print(f"{name}: {os.path.realpath(path)}")

print("\\n=== Loaded Modules Locations ===")
unique_dirs = set()
for name, mod in sys.modules.items():
    if hasattr(mod, '__file__') and mod.__file__:
        try:
            real_path = os.path.realpath(mod.__file__)
            parent = os.path.dirname(real_path)
            # Get root-level directories
            parts = parent.split(os.sep)
            if len(parts) > 2:
                root = os.sep.join(parts[:4])  # e.g., /Users/foo/.local
                unique_dirs.add(root)
        except:
            pass

for d in sorted(unique_dirs):
    print(f"Module root: {d}")

print("\\n=== Dynamic Libraries ===")
try:
    import ctypes.util
    libs = ['python3.11', 'ssl', 'crypto', 'z']
    for lib in libs:
        path = ctypes.util.find_library(lib)
        if path:
            print(f"{lib}: {path}")
except Exception as e:
    print(f"Error finding libs: {e}")

print("\\n=== Environment ===")
print(f"HOME: {os.environ.get('HOME', 'not set')}")
print(f"TMPDIR: {os.environ.get('TMPDIR', 'not set')}")
print(f"PATH: {os.environ.get('PATH', 'not set')}")
""")
            print("\n" + "=" * 60)
            print("PROBE RESULTS (sandbox=False)")
            print("=" * 60)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

    def test_probe_with_sandbox(self):
        """Run same probe with sandbox enabled to see what fails."""
        with SubprocessREPL(sandbox=True, timeout=10.0) as repl:
            result = repl.execute_code("""
import sys
import os

print("=== Sandbox Probe ===")
print(f"Python running: {sys.version}")
print(f"sys.executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

# Test specific paths
test_paths = [
    "/etc/hosts",
    "/tmp",
    "/var/tmp",
    "/private/tmp",
    os.path.expanduser("~"),
    os.path.expanduser("~/.ssh"),
    os.path.expanduser("~/.aws"),
    "/Users",
    "/System",
    "/Library",
]

print("\\n=== Path Access Tests ===")
for path in test_paths:
    try:
        if os.path.isdir(path):
            os.listdir(path)
            print(f"READ OK  (dir):  {path}")
        elif os.path.isfile(path):
            with open(path, 'r') as f:
                f.read(1)
            print(f"READ OK  (file): {path}")
        else:
            print(f"NOT FOUND:       {path}")
    except PermissionError:
        print(f"BLOCKED:         {path}")
    except Exception as e:
        print(f"ERROR ({type(e).__name__}): {path}")

# Test write access
print("\\n=== Write Access Tests ===")
write_tests = [
    "/tmp/rlm_probe_test.txt",
    "/var/tmp/rlm_probe_test.txt",
    os.path.expanduser("~/rlm_probe_test.txt"),
    "local_file.txt",  # Should work (in temp_dir)
]

for path in write_tests:
    try:
        with open(path, 'w') as f:
            f.write("test")
        os.remove(path)
        print(f"WRITE OK: {path}")
    except PermissionError:
        print(f"BLOCKED:  {path}")
    except Exception as e:
        print(f"ERROR ({type(e).__name__}): {path}")

# Test network
print("\\n=== Network Tests ===")
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("8.8.8.8", 53))
    s.close()
    print("NETWORK: TCP outbound ALLOWED")
except Exception as e:
    print(f"NETWORK: TCP outbound BLOCKED ({type(e).__name__})")

# Test Unix socket (for llm_query)
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Just test socket creation
    print("NETWORK: Unix socket creation OK")
except Exception as e:
    print(f"NETWORK: Unix socket BLOCKED ({type(e).__name__})")
""")
            print("\n" + "=" * 60)
            print("PROBE RESULTS (sandbox=True)")
            print("=" * 60)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

    def test_probe_llm_socket(self):
        """Test that llm_query socket works in sandbox."""
        with SubprocessREPL(sandbox=True, timeout=10.0) as repl:
            # First verify the socket path
            socket_path = repl.socket_path
            print(f"\nSocket path: {socket_path}")
            print(f"Socket exists: {os.path.exists(socket_path)}")

            result = repl.execute_code(f"""
import socket
import os

socket_path = "{socket_path}"
print(f"Testing socket: {{socket_path}}")
print(f"Socket exists: {{os.path.exists(socket_path)}}")

try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(socket_path)
    print("SOCKET: Connection successful")
    s.close()
except Exception as e:
    print(f"SOCKET: Connection failed - {{type(e).__name__}}: {{e}}")
""")
            print("\n" + "=" * 60)
            print("LLM SOCKET PROBE")
            print("=" * 60)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
