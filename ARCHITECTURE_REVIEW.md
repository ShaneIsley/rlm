# RLM Architecture and Style Review

**Repository**: RLM (Recursive Language Models)
**Review Date**: January 2026
**Reviewed by**: Claude

---

## Executive Summary

RLM is a well-architected inference engine that enables LLMs to handle near-infinite context through recursive self-calling via a REPL environment. The codebase demonstrates strong design principles with a plugin architecture, comprehensive type safety, and clear separation of concerns. While the project shows excellent fundamentals, there are opportunities for improvement in async support, error handling patterns, and test coverage.

---

## 1. Project Overview

### Purpose
RLM replaces standard `llm.completion()` calls with a recursive paradigm where models can:
- Programmatically examine and decompose large contexts
- Execute Python code in sandboxed environments
- Make sub-LLM calls for complex reasoning
- Return structured answers via `FINAL()` or `FINAL_VAR()` markers

### Core Architecture Flow
```
User Code → RLM.completion() → Environment (REPL) → LM Handler → LLM Client
                ↑                    ↓
                └─── Iteration Loop ←┘
```

---

## 2. Strengths

### 2.1 Plugin Architecture (Excellent)
The repository excels at extensibility through abstract base classes:

**Clients** (`rlm/clients/`):
- `BaseLM` abstract class defines the interface
- Factory pattern via `get_client()` for instantiation
- Supports 9+ providers: OpenAI, Anthropic, Portkey, Gemini, Azure, LiteLLM, vLLM, OpenRouter, Vercel

**Environments** (`rlm/environments/`):
- `BaseEnv`, `IsolatedEnv`, `NonIsolatedEnv` hierarchy
- `SupportsPersistence` Protocol for multi-turn support
- Four implementations: LocalREPL, DockerREPL, ModalREPL, PrimeREPL

This design allows adding new LLM providers or execution environments with minimal changes to core logic.

### 2.2 Type Safety (Strong)
```python
# Comprehensive type hints throughout
def completion(
    self, prompt: str | dict[str, Any], root_prompt: str | None = None
) -> RLMChatCompletion:
```

- Heavy use of `@dataclass` for data structures
- `Protocol` classes for structural subtyping (`SupportsPersistence`)
- Type literals for enums (`ClientBackend`, `EnvironmentType`)
- Modern Python 3.11+ type syntax

### 2.3 Code Quality Tooling (Comprehensive)
```toml
# pyproject.toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP"]  # Excellent lint rule selection
```

- **ruff** for linting and formatting (modern, fast)
- **ty** for type checking
- **pre-commit** hooks for consistency
- **pytest** with coverage reporting
- CI/CD via GitHub Actions

### 2.4 Security Considerations (Good)
- Safe builtins whitelist in LocalREPL (blocks `eval`, `exec`, `input`)
- Isolated sandbox options (Docker, Modal, Prime)
- Sensitive key filtering in logs (`filter_sensitive_keys()`)
- Thread-safe operations with locks

### 2.5 Documentation (Solid)
- Comprehensive README with usage examples
- Well-documented Protocol classes with implementation guidance
- CONTRIBUTING.md with clear roadmap
- Example files for each environment type
- Docstrings on public methods

### 2.6 Minimal Core Philosophy
The CONTRIBUTING.md explicitly states the goal of keeping `core/` minimal and readable. This is evident in:
- `rlm/core/rlm.py`: ~400 lines, well-organized
- Clear separation: core logic vs. clients vs. environments vs. utilities

---

## 3. Weaknesses

### 3.1 Limited Async Support (Significant)
The main completion loop is synchronous despite async methods existing on `BaseLM`:

```python
# rlm/core/rlm.py:308
response = lm_handler.completion(prompt)  # Synchronous
```

**Impact**: Cannot efficiently handle concurrent RLM calls or leverage async LLM SDKs.

### 3.2 Factory Pattern Could Be More Elegant
The `get_client()` function uses an if-elif chain:

```python
# rlm/clients/__init__.py
if backend == "openai":
    from rlm.clients.openai import OpenAIClient
    return OpenAIClient(**backend_kwargs)
elif backend == "vllm":
    # ... similar pattern repeats
```

This could be replaced with a registry pattern for easier extension:
```python
CLIENT_REGISTRY = {
    "openai": "rlm.clients.openai:OpenAIClient",
    "anthropic": "rlm.clients.anthropic:AnthropicClient",
    # ...
}
```

### 3.3 Error Handling Patterns (Inconsistent)
Some areas use exception raising, others return error strings:

```python
# LocalREPL returns error strings
def FINAL_VAR(var_name):
    if var_name not in locals_dict:
        return f"Error: Variable '{var_name}' not found"  # String error
```

vs.

```python
# RLM raises exceptions
if len(other_backends) != 1:
    raise ValueError("We currently only support one additional backend...")
```

### 3.4 Test Coverage Gaps
While tests exist, coverage could be improved:
- No tests for remote environments (Docker, Modal, Prime)
- Client implementations lack unit tests (only integration tests excluded in CI)
- Missing edge case tests for socket communication
- Type checking in CI uses `exit-zero` (warnings only)

### 3.5 Magic Strings / Numbers
Some values are hardcoded without constants:

```python
# rlm/core/lm_handler.py (inferred from types)
# 4-byte big-endian length prefix - magic number
```

### 3.6 Limited Multi-Model Routing
Currently restricted to one additional backend:
```python
if len(other_backends) != 1:
    raise ValueError(
        "We currently only support one additional backend for the recursive sub-calls!"
    )
```

### 3.7 Persistence Only for LocalREPL
Only `LocalREPL` implements `SupportsPersistence`. Other environments (Docker, Modal, Prime) don't support multi-turn conversations.

---

## 4. Code Style Analysis

### 4.1 Positive Patterns

**Context Managers for Resource Management**:
```python
@contextmanager
def _spawn_completion_context(self, prompt: str | dict[str, Any]):
    # Clean setup and teardown
    try:
        yield lm_handler, environment
    finally:
        lm_handler.stop()
        if not self.persistent and hasattr(environment, "cleanup"):
            environment.cleanup()
```

**Clear Method Naming**:
- `_spawn_completion_context()` - internal, spawns resources
- `completion()` - public entry point
- `_completion_turn()` - single iteration helper

**Lazy Imports** (for optional dependencies):
```python
if backend == "modal":
    from rlm.clients.modal import ModalClient  # Only imported when needed
```

### 4.2 Areas for Improvement

**Long Methods**: `completion()` at ~100 lines could be decomposed further.

**Nested Type Checking**:
```python
if isinstance(environment, SupportsPersistence)
```
This pattern appears multiple times; could use a helper method.

---

## 5. Potential Improvements

### 5.1 High Priority

1. **Add Async Completion API**
   ```python
   async def acompletion(self, prompt: str | dict[str, Any]) -> RLMChatCompletion:
       """Async version of completion for concurrent workloads."""
   ```

2. **Registry-Based Client/Environment Loading**
   - Replace if-elif chains with declarative registry
   - Allow third-party plugins via entry points

3. **Standardize Error Handling**
   - Define custom exception hierarchy (`RLMError`, `ClientError`, `EnvironmentError`)
   - Use Result types or consistent exception patterns

4. **Increase Test Coverage**
   - Add mock-based tests for remote environments
   - Enable strict type checking in CI
   - Add integration test suite that can run with real APIs (optional)

### 5.2 Medium Priority

5. **Multi-Model Routing**
   - Remove single `other_backend` limitation
   - Allow dynamic model selection based on task characteristics

6. **Configuration Management**
   - Add Pydantic or dataclass-based configuration validation
   - Support configuration from YAML/JSON files

7. **Observability Improvements**
   - Add structured logging with levels
   - OpenTelemetry integration for tracing
   - Metrics collection (iteration counts, token usage, latency)

8. **Persistence for Remote Environments**
   - Implement `SupportsPersistence` for DockerREPL
   - Consider session management for cloud environments

### 5.3 Lower Priority

9. **Code Block Extraction**
   - Support for multiple languages beyond Python
   - Better handling of nested code blocks

10. **Rate Limiting / Retry Logic**
    - Built-in exponential backoff for LLM calls
    - Configurable rate limits per client

11. **Caching Layer**
    - LRU cache for repeated prompts
    - Prefix caching support (noted in CONTRIBUTING.md as research)

---

## 6. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                         │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                          RLM Class                               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ completion() │  │ _setup_prompt│  │ _spawn_completion_ctx │  │
│  └──────┬───────┘  └──────────────┘  └───────────────────────┘  │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐        ┌─────────────────────────────────┐
│     LMHandler       │◄──────►│         Environment              │
│  (TCP Socket Server)│        │  ┌───────────┐  ┌─────────────┐ │
│  - completion()     │        │  │LocalREPL  │  │DockerREPL   │ │
│  - register_client()│        │  │ModalREPL  │  │PrimeREPL    │ │
│  - get_usage()      │        │  └───────────┘  └─────────────┘ │
└─────────┬───────────┘        └────────────────────┬────────────┘
          │                                         │
          ▼                                         │
┌─────────────────────────────────────────────┐    │
│              Client Layer                    │◄───┘
│  ┌─────────┐ ┌──────────┐ ┌───────────────┐ │    (llm_query())
│  │ OpenAI  │ │Anthropic │ │    Others     │ │
│  └─────────┘ └──────────┘ └───────────────┘ │
└─────────────────────────────────────────────┘
```

---

## 7. Summary Scorecard

| Aspect                  | Score | Notes                                      |
|-------------------------|-------|--------------------------------------------|
| **Architecture**        | 8/10  | Excellent plugin design, clear separation  |
| **Code Quality**        | 8/10  | Clean, readable, good tooling              |
| **Type Safety**         | 9/10  | Comprehensive hints, protocols, dataclasses|
| **Testing**             | 6/10  | Good unit tests, gaps in integration       |
| **Documentation**       | 7/10  | Solid README/docs, could use more API docs |
| **Extensibility**       | 9/10  | Easy to add clients/environments           |
| **Security**            | 7/10  | Good sandboxing options, safe builtins     |
| **Performance**         | 6/10  | Synchronous only, no caching               |
| **Error Handling**      | 6/10  | Inconsistent patterns                      |
| **Overall**             | 7.5/10| Production-ready research codebase         |

---

## 8. Conclusion

RLM is a well-designed research codebase that successfully balances simplicity with extensibility. The plugin architecture for clients and environments is particularly well-executed, making it easy to adapt to different LLM providers and execution contexts.

The main areas for improvement are:
1. **Async support** for better concurrency
2. **Standardized error handling** across the codebase
3. **Extended test coverage** especially for remote environments

The explicit philosophy of keeping the core minimal (as stated in CONTRIBUTING.md) is commendable and has resulted in a codebase that is genuinely readable in a short sitting. This is a strong foundation for the ambitious research goals outlined in the project roadmap.
