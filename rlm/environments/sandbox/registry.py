"""Sandbox strategy registry and factory."""

from rlm.core.exceptions import SandboxUnavailableError
from rlm.environments.sandbox.base import SandboxStrategy
from rlm.environments.sandbox.linux import LinuxBwrap
from rlm.environments.sandbox.macos import MacOSSandboxExec

# Registry of all sandbox strategies, in order of preference
SANDBOX_STRATEGIES: list[type[SandboxStrategy]] = [
    MacOSSandboxExec,
    LinuxBwrap,
    # Future strategies can be added here:
    # LinuxLandlock,
    # LinuxSeccomp,
    # FreeBSDCapsicum,
]


def get_available_strategies() -> list[type[SandboxStrategy]]:
    """Return list of sandbox strategies available on this system."""
    return [s for s in SANDBOX_STRATEGIES if s.is_available()]


def get_sandbox_strategy(preferred: str | None = None) -> SandboxStrategy:
    """Get an available sandbox strategy.

    Args:
        preferred: Specific strategy name to use, or None for auto-detection.
                   Names are matched case-insensitively against strategy names.

    Returns:
        An instantiated SandboxStrategy ready for use.

    Raises:
        SandboxUnavailableError: If no sandbox is available, or if the
            preferred strategy is not available.

    Example:
        # Auto-detect best available strategy
        strategy = get_sandbox_strategy()

        # Request specific strategy
        strategy = get_sandbox_strategy("Linux bubblewrap")
    """
    if preferred:
        # Find the requested strategy
        preferred_lower = preferred.lower()
        for strategy_cls in SANDBOX_STRATEGIES:
            if strategy_cls.name().lower() == preferred_lower:
                if strategy_cls.is_available():
                    return strategy_cls()
                raise SandboxUnavailableError([strategy_cls.get_missing_dependency_message()])

        # Unknown strategy name
        known_names = [s.name() for s in SANDBOX_STRATEGIES]
        raise SandboxUnavailableError(
            [f"Unknown sandbox strategy: '{preferred}'. Known strategies: {known_names}"]
        )

    # Auto-detect: use first available strategy
    for strategy_cls in SANDBOX_STRATEGIES:
        if strategy_cls.is_available():
            return strategy_cls()

    # No sandbox available - provide helpful error with all options
    messages = [s.get_missing_dependency_message() for s in SANDBOX_STRATEGIES]
    raise SandboxUnavailableError(messages)


def is_sandbox_available() -> bool:
    """Check if any sandbox strategy is available on this system.

    Returns:
        True if at least one sandbox strategy can be used.
    """
    return any(s.is_available() for s in SANDBOX_STRATEGIES)


def get_sandbox_info() -> dict[str, bool]:
    """Get availability status of all sandbox strategies.

    Returns:
        Dict mapping strategy names to availability status.

    Example:
        >>> get_sandbox_info()
        {'macOS sandbox-exec': True, 'Linux bubblewrap': False}
    """
    return {s.name(): s.is_available() for s in SANDBOX_STRATEGIES}
