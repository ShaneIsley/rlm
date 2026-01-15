"""
Example: An example from the Oolong Benchmark from the RLM paper: https://arxiv.org/abs/2512.24601v1
"""

import os
import random
import sys

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

try:
    from datasets import load_dataset
except ImportError:
    print(
        "Please install the 'datasets' library to run this example. Run `uv pip install datasets`"
    )
    sys.exit(1)


def load_oolong_dataset():
    """Load the Oolong benchmark dataset."""
    ds = load_dataset("oolongbench/oolong-real", "toy_dnd", split="test", streaming=False)
    return ds


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Load benchmark data
    ds = load_oolong_dataset()
    dataset_size = len(ds)
    random_index = random.randint(0, dataset_size - 1)
    row = ds[random_index]
    print(f"Loading random row {random_index} from dataset (total: {dataset_size} rows)")
    context = row["context_window_text"]
    question = row["question"]
    expected_answer = row["answer"]

    print(f"Question: {question}")
    print(f"Expected answer: {expected_answer}")
    print("-" * 50)

    # Create logger
    logger = RLMLogger(log_dir="./logs")

    # Create RLM instance
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": "gpt-5-mini",
            "api_key": api_key,
        },
        environment="subprocess",
        max_iterations=30,
        logger=logger,
        verbose=True,
    )

    # Run completion with context and question
    result = rlm.completion(prompt=context, root_prompt=question)

    print("-" * 50)
    print(f"RLM Response: {result.response}")
    print(f"Expected: {expected_answer}")

    # Simple validation (exact match or contained)
    is_correct = (
        expected_answer.lower() in result.response.lower()
        or result.response.lower() in expected_answer.lower()
    )
    print(f"Match: {is_correct}")


if __name__ == "__main__":
    main()
