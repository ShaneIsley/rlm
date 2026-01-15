from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

# Note: API key is automatically loaded from OPENAI_API_KEY environment variable
rlm = RLM(
    backend="openai",  # or "portkey", etc.
    backend_kwargs={
        "model_name": "gpt-5-nano",
    },
    environment="local",
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

result = rlm.completion("Print me the first 100 powers of two, each on a newline.")

print(result)
