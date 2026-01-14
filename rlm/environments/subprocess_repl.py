"""
Subprocess-based REPL environment with optional platform-specific sandboxing.

This environment runs Python code in a separate subprocess with UV-managed
virtual environments, providing process-level isolation and optional
filesystem/network sandboxing.

Requirements:
    - uv (https://github.com/astral-sh/uv)

Optional (for sandboxing):
    - macOS: sandbox-exec (built-in)
    - Linux: bubblewrap (bwrap)
"""

import atexit
import base64
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Callable

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.exceptions import ConfigurationError
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import NonIsolatedEnv, SupportsPersistence

# =============================================================================
# Global Cleanup Registry
# =============================================================================

_ACTIVE_TEMP_DIRS: set[str] = set()
_CLEANUP_REGISTERED = False
_CLEANUP_LOCK = threading.Lock()


def _emergency_cleanup():
    """Called by atexit/signal handlers to clean up orphaned temp dirs."""
    with _CLEANUP_LOCK:
        for temp_dir in list(_ACTIVE_TEMP_DIRS):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        _ACTIVE_TEMP_DIRS.clear()


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT gracefully."""
    _emergency_cleanup()
    # Re-raise with default handler
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _register_global_cleanup():
    """Register cleanup handlers once per process."""
    global _CLEANUP_REGISTERED
    with _CLEANUP_LOCK:
        if _CLEANUP_REGISTERED:
            return

        atexit.register(_emergency_cleanup)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _signal_handler)
            except (OSError, ValueError):
                pass  # May fail in threads or restricted environments

        _CLEANUP_REGISTERED = True


# =============================================================================
# Execution Script Template
# =============================================================================

EXEC_SCRIPT_TEMPLATE = textwrap.dedent('''
import sys
import io
import json
import socket
import base64
import traceback
import os

# IPC Configuration
SOCKET_PATH = "{socket_path}"
DEPTH = {depth}

def llm_query(prompt, model=None):
    """Query LLM via Unix socket to parent process."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        request = json.dumps({{"type": "single", "prompt": prompt, "model": model, "depth": DEPTH}})
        sock.sendall(request.encode() + b"\\n")

        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b"\\n" in response:
                break
        sock.close()

        data = json.loads(response.decode().strip())
        if data.get("error"):
            return f"Error: {{data['error']}}"
        return data.get("response", "Error: No response")
    except Exception as e:
        return f"Error: LLM query failed - {{e}}"

def llm_query_batched(prompts, model=None):
    """Query LLM with multiple prompts via Unix socket."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        request = json.dumps({{"type": "batched", "prompts": prompts, "model": model, "depth": DEPTH}})
        sock.sendall(request.encode() + b"\\n")

        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b"\\n" in response:
                break
        sock.close()

        data = json.loads(response.decode().strip())
        if data.get("error"):
            return [f"Error: {{data['error']}}"] * len(prompts)
        return data.get("responses", ["Error: No response"] * len(prompts))
    except Exception as e:
        return [f"Error: LLM query failed - {{e}}"] * len(prompts)

# State persistence
STATE_FILE = "{state_file}"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {{}}

def save_state(s):
    serializable = {{}}
    for k, v in s.items():
        if not k.startswith("_"):
            try:
                json.dumps(v)
                serializable[k] = v
            except Exception:
                serializable[k] = repr(v)
    with open(STATE_FILE, "w") as f:
        json.dump(serializable, f)

_locals = load_state()

def FINAL_VAR(name):
    name = name.strip().strip("\\"\\'")
    return str(_locals.get(name, f"Error: Variable '{{name}}' not found"))

_globals = {{
    "__builtins__": __builtins__,
    "__name__": "__main__",
    "llm_query": llm_query,
    "llm_query_batched": llm_query_batched,
    "FINAL_VAR": FINAL_VAR,
}}

# Execute user code
code = base64.b64decode("{code_b64}").decode()
stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr
exec_time_start = __import__("time").perf_counter()

try:
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    combined = {{**_globals, **_locals}}
    exec(code, combined, combined)
    for k, v in combined.items():
        if k not in _globals and not k.startswith("_"):
            _locals[k] = v
except Exception:
    traceback.print_exc(file=stderr_buf)
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr

exec_time = __import__("time").perf_counter() - exec_time_start
save_state(_locals)

# Output result as JSON
result = {{
    "stdout": stdout_buf.getvalue(),
    "stderr": stderr_buf.getvalue(),
    "locals": {{k: repr(v) for k, v in _locals.items() if not k.startswith("_")}},
    "execution_time": exec_time,
}}
print(json.dumps(result, ensure_ascii=False))
''')


# =============================================================================
# SubprocessREPL Implementation
# =============================================================================


class SubprocessREPL(NonIsolatedEnv):
    """
    Subprocess-based REPL with UV venv and optional platform sandboxing.

    Features:
        - Process isolation (separate Python interpreter)
        - UV-managed virtual environments (~50ms creation)
        - Platform-specific sandboxing (macOS sandbox-exec, Linux bwrap)
        - Interactive package approval
        - Overhead tracking with summary
        - Robust cleanup (atexit, signals, stale detection)
        - SupportsPersistence protocol for multi-turn conversations

    Example:
        with SubprocessREPL() as repl:
            result = repl.execute_code("x = 1 + 1")
            print(result.stdout)
    """

    TEMP_PREFIX = "rlm_subprocess_"

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        # SubprocessREPL-specific options
        timeout: float = 30.0,
        memory_limit_mb: int = 512,
        sandbox: bool = True,
        allowed_packages: list[str] | None = None,
        auto_approve_packages: bool = False,
        package_approval_callback: Callable[[str], bool] | None = None,
        **kwargs,
    ):
        """
        Initialize SubprocessREPL.

        Args:
            lm_handler_address: (host, port) for LLM queries from subprocess.
            context_payload: Initial context data to load.
            setup_code: Code to run during initialization.
            persistent: If True, preserve state across execute_code calls.
            depth: Recursion depth for nested LLM calls.
            timeout: Maximum execution time in seconds.
            memory_limit_mb: Memory limit for subprocess (best-effort).
            sandbox: Enable platform-specific sandboxing if available.
            allowed_packages: Pre-approved packages (no prompt needed).
            auto_approve_packages: If True, install packages without prompting.
            package_approval_callback: Custom function for package approval.
        """
        super().__init__(persistent=persistent, depth=depth, **kwargs)

        # Verify uv is installed
        if not shutil.which("uv"):
            raise ConfigurationError(
                "SubprocessREPL requires 'uv' to be installed. "
                "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh",
                missing_field="uv",
            )

        _register_global_cleanup()

        self.lm_handler_address = lm_handler_address
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.sandbox = sandbox
        self.auto_approve = auto_approve_packages
        self.approval_callback = package_approval_callback or self._default_approval

        # Pre-approved packages (stdlib + user-specified)
        self.allowed_packages: set[str] = {
            "json", "re", "math", "collections", "itertools",
            "functools", "datetime", "random", "string", "typing",
            "os", "sys", "io", "time", "pathlib", "copy",
        }
        if allowed_packages:
            self.allowed_packages.update(allowed_packages)

        # Track installed packages during session
        self._installed_packages: set[str] = set()

        # Overhead tracking
        self._overhead_stats: dict[str, Any] = {
            "venv_creation_ms": 0.0,
            "executions": [],
            "package_installs": [],
        }

        # Persistence tracking
        self._context_count: int = 0
        self._history_count: int = 0

        # Pending LLM calls from current execution
        self._pending_llm_calls: list[RLMChatCompletion] = []

        # Setup directories and venv
        self.setup()

        if context_payload is not None:
            self.load_context(context_payload)
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Create temp directory, venv, and socket server."""
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix=self.TEMP_PREFIX)
        _ACTIVE_TEMP_DIRS.add(self.temp_dir)

        # Write sentinel file for stale detection
        self._write_sentinel()

        # Create UV venv
        start = time.perf_counter()
        self.venv_path = os.path.join(self.temp_dir, "venv")
        subprocess.run(
            ["uv", "venv", self.venv_path, "--python", "3.11", "-q"],
            check=True,
            capture_output=True,
        )
        self._overhead_stats["venv_creation_ms"] = (time.perf_counter() - start) * 1000

        # State file for persistent locals
        self.state_file = os.path.join(self.temp_dir, "state.json")

        # Socket for IPC
        self.socket_path = os.path.join(self.temp_dir, "llm.sock")
        self._socket_server: socket.socket | None = None
        self._socket_thread: threading.Thread | None = None
        self._start_socket_server()

    def _write_sentinel(self):
        """Write metadata file for stale directory detection."""
        sentinel_path = Path(self.temp_dir) / ".rlm_sentinel"
        sentinel_path.write_text(f"{os.getpid()}\n{time.time()}\n")

    def _start_socket_server(self):
        """Start Unix socket server for llm_query IPC."""
        self._socket_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket_server.bind(self.socket_path)
        self._socket_server.listen(5)
        self._socket_server.settimeout(1.0)  # Allow periodic checking

        self._socket_running = True
        self._socket_thread = threading.Thread(target=self._socket_handler, daemon=True)
        self._socket_thread.start()

    def _socket_handler(self):
        """Handle incoming llm_query requests from subprocess."""
        while self._socket_running:
            try:
                conn, _ = self._socket_server.accept()
                conn.settimeout(300.0)  # 5 min timeout for LLM calls

                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in data:
                        break

                if data:
                    request = json.loads(data.decode().strip())
                    response = self._handle_llm_request(request)
                    conn.sendall(json.dumps(response).encode() + b"\n")

                conn.close()
            except socket.timeout:
                continue
            except Exception:
                continue

    def _handle_llm_request(self, request: dict) -> dict:
        """Process an llm_query request from the subprocess."""
        if not self.lm_handler_address:
            return {"error": "No LM handler configured"}

        try:
            req_type = request.get("type", "single")
            model = request.get("model")
            depth = request.get("depth", self.depth)

            if req_type == "single":
                prompt = request.get("prompt", "")
                lm_request = LMRequest(prompt=prompt, model=model, depth=depth)
                response = send_lm_request(self.lm_handler_address, lm_request)

                if not response.success:
                    return {"error": response.error}

                self._pending_llm_calls.append(response.chat_completion)
                return {"response": response.chat_completion.response}

            elif req_type == "batched":
                prompts = request.get("prompts", [])
                responses = send_lm_request_batched(
                    self.lm_handler_address, prompts, model=model, depth=depth
                )

                results = []
                for resp in responses:
                    if not resp.success:
                        results.append(f"Error: {resp.error}")
                    else:
                        self._pending_llm_calls.append(resp.chat_completion)
                        results.append(resp.chat_completion.response)

                return {"responses": results}

            else:
                return {"error": f"Unknown request type: {req_type}"}

        except Exception as e:
            return {"error": str(e)}

    def _default_approval(self, package: str) -> bool:
        """Default package approval: prompt user via stdin."""
        try:
            response = input(f"Code wants to import '{package}'. Install? [y/N]: ")
            return response.lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _request_package_approval(self, package: str) -> bool:
        """Check if package should be installed."""
        if package in self.allowed_packages:
            return True
        if package in self._installed_packages:
            return True
        if self.auto_approve:
            return True
        return self.approval_callback(package)

    def _install_package(self, package: str):
        """Install package into venv using uv."""
        start = time.perf_counter()
        python_path = os.path.join(self.venv_path, "bin", "python")
        subprocess.run(
            ["uv", "pip", "install", "-q", "--python", python_path, package],
            check=True,
            capture_output=True,
        )
        self._installed_packages.add(package)
        self._overhead_stats["package_installs"].append({
            "package": package,
            "time_ms": (time.perf_counter() - start) * 1000,
        })

    def _extract_missing_module(self, stderr: str) -> str | None:
        """Extract module name from ImportError/ModuleNotFoundError."""
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
        return match.group(1).split(".")[0] if match else None

    def _build_exec_script(self, code: str) -> str:
        """Build the execution script with injected helpers."""
        code_b64 = base64.b64encode(code.encode()).decode()
        return EXEC_SCRIPT_TEMPLATE.format(
            socket_path=self.socket_path,
            depth=self.depth,
            state_file=self.state_file,
            code_b64=code_b64,
        )

    def _get_sandbox_command(self, python_cmd: list[str]) -> list[str]:
        """Wrap python command with platform-specific sandbox."""
        if not self.sandbox:
            return python_cmd

        system = platform.system()

        if system == "Darwin":
            return self._macos_sandbox_wrap(python_cmd)
        elif system == "Linux":
            return self._linux_sandbox_wrap(python_cmd)
        else:
            return python_cmd

    def _macos_sandbox_wrap(self, cmd: list[str]) -> list[str]:
        """Wrap with macOS sandbox-exec."""
        if not os.path.exists("/usr/bin/sandbox-exec"):
            return cmd

        # Resolve symlinks (e.g., /var/folders -> /private/var/folders)
        real_temp_dir = os.path.realpath(self.temp_dir)
        real_venv_path = os.path.realpath(self.venv_path)
        home_dir = os.path.expanduser("~")
        real_home_dir = os.path.realpath(home_dir)

        # Find the actual Python interpreter location (may be in ~/.local/share/uv/)
        python_path = os.path.join(self.venv_path, "bin", "python")
        real_python = os.path.realpath(python_path)
        # Get the parent directory containing the Python installation
        # e.g., /Users/foo/.local/share/uv/python/cpython-3.11.11-macos-aarch64-none
        python_install_dir = os.path.dirname(os.path.dirname(real_python))

        # Use a permissive base and deny specific operations
        # This approach works better with Python's runtime requirements
        profile = f"""
(version 1)

;; Start permissive, then restrict
(allow default)

;; DENY: External network access (TCP/UDP to remote hosts)
(deny network-outbound (remote ip))

;; DENY: Writing outside allowed directories
(deny file-write*
    (subpath "/Users")
    (subpath "/home")
    (subpath "{real_home_dir}")
)

;; ALLOW: Write to our temp directory (override the deny above)
(allow file-write* (subpath "{real_temp_dir}"))

;; DENY: Reading user's home directory (privacy) - but allow specific paths
(deny file-read*
    (subpath "{real_home_dir}")
)

;; ALLOW: Read from our temp and venv
(allow file-read* (subpath "{real_temp_dir}"))
(allow file-read* (subpath "{real_venv_path}"))

;; ALLOW: Read from UV's Python installation (needed for libpython)
(allow file-read* (subpath "{python_install_dir}"))
"""
        profile_path = os.path.join(self.temp_dir, "sandbox.sb")
        with open(profile_path, "w") as f:
            f.write(profile)

        return ["/usr/bin/sandbox-exec", "-f", profile_path] + cmd

    def _linux_sandbox_wrap(self, cmd: list[str]) -> list[str]:
        """Wrap with Linux bubblewrap (if available)."""
        if not shutil.which("bwrap"):
            return cmd

        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/sbin", "/sbin",
        ]

        # Add /lib64 if it exists
        if os.path.exists("/lib64"):
            bwrap_cmd.extend(["--ro-bind", "/lib64", "/lib64"])

        bwrap_cmd.extend([
            "--ro-bind", self.venv_path, self.venv_path,
            "--bind", self.temp_dir, self.temp_dir,
            "--unshare-net",
            "--unshare-pid",
            "--die-with-parent",
            "--",
        ])

        return bwrap_cmd + cmd

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in subprocess and return result."""
        total_start = time.perf_counter()

        # Clear pending LLM calls
        self._pending_llm_calls = []

        result = self._try_execute(code)

        # Check for missing module and offer to install
        if "ModuleNotFoundError" in result.stderr or "No module named" in result.stderr:
            missing = self._extract_missing_module(result.stderr)
            if missing and missing not in self._installed_packages:
                if self._request_package_approval(missing):
                    self._install_package(missing)
                    result = self._try_execute(code)

        total_time = time.perf_counter() - total_start

        # Track overhead
        self._overhead_stats["executions"].append({
            "total_ms": total_time * 1000,
            "code_ms": (result.execution_time or 0) * 1000,
            "overhead_ms": (total_time - (result.execution_time or 0)) * 1000,
        })

        return result

    def _try_execute(self, code: str) -> REPLResult:
        """Attempt to execute code in subprocess."""
        script = self._build_exec_script(code)
        python_path = os.path.join(self.venv_path, "bin", "python")

        cmd = self._get_sandbox_command([python_path, "-c", script])

        # Build environment - inherit essential vars for macOS compatibility
        env = os.environ.copy()
        env.update({
            "PATH": os.path.join(self.venv_path, "bin") + ":/usr/bin:/bin",
            "HOME": self.temp_dir,
            "TMPDIR": self.temp_dir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        })

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.temp_dir,
                env=env,
            )

            # Parse JSON output
            try:
                lines = result.stdout.strip().split("\n")
                if lines and lines[-1]:
                    data = json.loads(lines[-1])
                    return REPLResult(
                        stdout=data.get("stdout", ""),
                        stderr=data.get("stderr", "") + result.stderr,
                        locals=data.get("locals", {}),
                        execution_time=data.get("execution_time"),
                        rlm_calls=self._pending_llm_calls.copy(),
                    )
            except json.JSONDecodeError:
                pass

            # Build debug info for failed execution
            debug_info = f"exit_code={result.returncode}"
            if result.stderr:
                debug_info = result.stderr
            elif not result.stdout:
                debug_info = f"No output (exit_code={result.returncode})"

            return REPLResult(
                stdout=result.stdout,
                stderr=debug_info,
                locals={},
                execution_time=None,
                rlm_calls=self._pending_llm_calls.copy(),
            )

        except subprocess.TimeoutExpired:
            return REPLResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout}s",
                locals={},
                execution_time=self.timeout,
                rlm_calls=self._pending_llm_calls.copy(),
            )

    # =========================================================================
    # SupportsPersistence Protocol
    # =========================================================================

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for a new completion call."""
        self.lm_handler_address = address

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """Add a context with versioned variable name."""
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(
                f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()"
            )
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)"
            )

        if context_index == 0:
            self.execute_code(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """Store a conversation's message history as a versioned variable."""
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Write history to file and load in subprocess
        history_path = os.path.join(self.temp_dir, f"history_{history_index}.json")
        with open(history_path, "w") as f:
            json.dump(message_history, f)

        self.execute_code(
            f"import json\nwith open(r'{history_path}', 'r') as f:\n    {var_name} = json.load(f)"
        )

        if history_index == 0:
            self.execute_code(f"history = {var_name}")

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    # =========================================================================
    # Overhead Tracking
    # =========================================================================

    def get_overhead_summary(self) -> dict:
        """Get summary of execution overhead for this session."""
        executions = self._overhead_stats["executions"]
        installs = self._overhead_stats["package_installs"]

        if not executions:
            return {
                "venv_creation_ms": round(self._overhead_stats["venv_creation_ms"], 2),
                "message": "No executions recorded",
            }

        total_overhead = sum(e["overhead_ms"] for e in executions)
        total_code_time = sum(e["code_ms"] for e in executions)
        total_install_time = sum(i["time_ms"] for i in installs)

        return {
            "venv_creation_ms": round(self._overhead_stats["venv_creation_ms"], 2),
            "execution_count": len(executions),
            "total_code_time_ms": round(total_code_time, 2),
            "total_overhead_ms": round(total_overhead, 2),
            "total_install_time_ms": round(total_install_time, 2),
            "avg_overhead_per_execution_ms": round(total_overhead / len(executions), 2),
            "packages_installed": [i["package"] for i in installs],
            "overhead_percentage": round(
                (total_overhead / (total_overhead + total_code_time)) * 100, 1
            ) if total_code_time > 0 else 0,
        }

    def print_overhead_summary(self):
        """Print human-readable overhead summary."""
        summary = self.get_overhead_summary()
        if "message" in summary:
            return

        print("\n" + "=" * 50)
        print("SubprocessREPL Overhead Summary")
        print("=" * 50)
        print(f"  Venv creation:     {summary['venv_creation_ms']:.1f}ms")
        print(f"  Executions:        {summary['execution_count']}")
        print(f"  Total code time:   {summary['total_code_time_ms']:.1f}ms")
        print(f"  Total overhead:    {summary['total_overhead_ms']:.1f}ms")
        print(f"  Avg overhead/exec: {summary['avg_overhead_per_execution_ms']:.1f}ms")
        if summary["packages_installed"]:
            print(f"  Packages installed: {', '.join(summary['packages_installed'])}")
            print(f"  Install time:      {summary['total_install_time_ms']:.1f}ms")
        print(f"  Overhead ratio:    {summary['overhead_percentage']:.1f}%")
        print("=" * 50 + "\n")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self):
        """Clean up resources."""
        # Stop socket server
        self._socket_running = False
        if self._socket_server:
            try:
                self._socket_server.close()
            except Exception:
                pass
            self._socket_server = None

        # Print overhead summary
        if hasattr(self, "_overhead_stats"):
            self.print_overhead_summary()

        # Unregister from emergency cleanup
        _ACTIVE_TEMP_DIRS.discard(getattr(self, "temp_dir", ""))

        # Remove temp directory
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    @classmethod
    def cleanup_stale_directories(cls, max_age_hours: float = 24) -> list[str]:
        """
        Clean up orphaned temp directories from crashed processes.

        Call periodically or at application startup.

        Args:
            max_age_hours: Remove directories older than this.

        Returns:
            List of removed directory paths.
        """
        temp_root = tempfile.gettempdir()
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned = []

        for entry in Path(temp_root).iterdir():
            if not entry.name.startswith(cls.TEMP_PREFIX):
                continue
            if not entry.is_dir():
                continue

            sentinel = entry / ".rlm_sentinel"
            should_clean = False

            if sentinel.exists():
                try:
                    lines = sentinel.read_text().strip().split("\n")
                    pid = int(lines[0])
                    created_time = float(lines[1])

                    # Check if process is dead
                    try:
                        os.kill(pid, 0)
                        process_alive = True
                    except OSError:
                        process_alive = False

                    if not process_alive or (now - created_time) > max_age_seconds:
                        should_clean = True

                except (ValueError, IndexError, OSError):
                    if (now - entry.stat().st_mtime) > max_age_seconds:
                        should_clean = True
            else:
                if (now - entry.stat().st_mtime) > max_age_seconds:
                    should_clean = True

            if should_clean:
                try:
                    shutil.rmtree(entry, ignore_errors=True)
                    cleaned.append(str(entry))
                except Exception:
                    pass

        return cleaned
