#!/usr/bin/env python3
"""
Quickstart with macOS Sandbox

Same as quickstart.py but runs code in a sandboxed SubprocessREPL environment.
This provides:
- Process isolation via UV virtual environments
- macOS sandbox-exec protection (blocks network, filesystem access)
- Safe execution of LLM-generated code

Requirements:
    - uv: curl -LsSf https://astral.sh/uv/install.sh | sh
    - OpenAI API key (or other backend)

Usage:
    OPENAI_API_KEY=... python examples/quickstart-sandbox-macos.py
"""

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

# Note: API key is automatically loaded from OPENAI_API_KEY environment variable
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "gpt-5-nano",
    },
    environment="subprocess",  # Use SubprocessREPL instead of local
    environment_kwargs={
        "sandbox": True,  # Enable macOS/Linux sandbox
        "timeout": 30.0,
    },
    max_depth=1,
    logger=logger,
    verbose=True,
)

result = rlm.completion("Print me the first 100 powers of two, each on a newline.")

print(result)
