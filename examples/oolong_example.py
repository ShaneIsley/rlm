"""
Example: An example from the Oolong Benchmark from the RLM paper: https://arxiv.org/abs/2512.24601v1
"""

import argparse
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


def load_random_oolong_row() -> dict:
    """Load a random row from the Oolong benchmark using streaming with shuffle."""
    ds = load_dataset("oolongbench/oolong-real", "toy_dnd", split="test", streaming=True)
    shuffled_ds = ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000)
    row = next(iter(shuffled_ds))
    return row


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Oolong benchmark example with RLM")
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="Backend to use (e.g., openai, anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name to use",
    )
    args = parser.parse_args()

    # Load benchmark data
    print("Loading random row from dataset using shuffle...")
    row = load_random_oolong_row()
    context = row["context_window_text"]
    question = row["question"]
    expected_answer = row["answer"]

    print(f"Question: {question}")
    print(f"Expected answer: {expected_answer}")
    print("-" * 50)

    # Create logger
    logger = RLMLogger(log_dir="./logs")

    # Create RLM instance
    # Note: API key is automatically loaded from environment variable
    print(f"Using backend: {args.backend}, model: {args.model}")
    rlm = RLM(
        backend=args.backend,
        backend_kwargs={
            "model_name": args.model,
        },
        environment="subprocess",
        max_iterations=30,
        logger=logger,
        verbose=True,
    )

    # Run completion with context and question
    result = rlm.completion(prompt=context, root_prompt=question)

    print("-" * 50)
    print(f"Question: {question}")
    print(f"Expected: {expected_answer}")
    print(f"RLM Response: {result.response}")
    print("-" * 50)

    # Simple validation (exact match or contained)
    is_correct = (
        expected_answer.lower() in result.response.lower()
        or result.response.lower() in expected_answer.lower()
    )
    print(f"Match: {is_correct}")


if __name__ == "__main__":
    main()
