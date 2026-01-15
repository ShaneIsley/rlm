"""
Single-Needle-in-a-Haystack (S-NIAH) Benchmark.

Tests the ability to retrieve a single piece of information from a large context.
This is a context-insensitive retrieval task where the answer location doesn't
depend on semantic understanding of the surrounding text.

The benchmark generates synthetic haystacks with a hidden "needle" fact.
"""

import random
import string
from collections.abc import Iterator

from benchmarks.base import Benchmark, BenchmarkSample
from benchmarks.metrics import Metrics


class NIAHBenchmark(Benchmark):
    """Single-Needle-in-a-Haystack benchmark.

    Generates synthetic documents with a single retrievable fact (the "needle")
    hidden at various positions within distractor text (the "haystack").

    Args:
        context_length: Target length of the haystack in characters.
        needle_depth: Position of needle (0.0=start, 0.5=middle, 1.0=end).
                      If None, randomized per sample.
        distractor_type: Type of haystack content ("lorem", "essays", "random").
    """

    LOREM_IPSUM = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

    NEEDLE_TEMPLATES = [
        ("The secret code is {value}.", "What is the secret code?"),
        ("The password for the vault is {value}.", "What is the password for the vault?"),
        ("The answer to the riddle is {value}.", "What is the answer to the riddle?"),
        ("The hidden number is {value}.", "What is the hidden number?"),
        ("The magic word is {value}.", "What is the magic word?"),
    ]

    def __init__(
        self,
        context_length: int = 100_000,
        needle_depth: float | None = None,
        distractor_type: str = "lorem",
    ):
        self.context_length = context_length
        self.needle_depth = needle_depth
        self.distractor_type = distractor_type

    @property
    def name(self) -> str:
        return f"niah-{self.context_length // 1000}k"

    @property
    def description(self) -> str:
        return f"Single-Needle-in-a-Haystack with {self.context_length:,} char context"

    def _generate_haystack(self, length: int, rng: random.Random) -> str:
        """Generate distractor text of approximately the target length."""
        if self.distractor_type == "lorem":
            # Repeat lorem ipsum with slight variations
            paragraphs = []
            while len("\n\n".join(paragraphs)) < length:
                # Shuffle words slightly for variation
                words = self.LOREM_IPSUM.split()
                rng.shuffle(words)
                paragraphs.append(" ".join(words))
            return "\n\n".join(paragraphs)[:length]

        elif self.distractor_type == "random":
            # Random sentences
            words = [
                "the",
                "a",
                "is",
                "are",
                "was",
                "were",
                "has",
                "have",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "can",
                "be",
                "been",
                "being",
                "have",
                "has",
            ]
            nouns = [
                "cat",
                "dog",
                "house",
                "tree",
                "car",
                "book",
                "table",
                "chair",
                "computer",
                "phone",
                "person",
                "city",
                "country",
                "world",
            ]
            adjectives = [
                "big",
                "small",
                "red",
                "blue",
                "green",
                "old",
                "new",
                "fast",
                "slow",
                "hot",
                "cold",
                "bright",
                "dark",
            ]

            sentences = []
            current_length = 0
            while current_length < length:
                sentence = f"The {rng.choice(adjectives)} {rng.choice(nouns)} {rng.choice(words)} {rng.choice(adjectives)}."
                sentences.append(sentence)
                current_length += len(sentence) + 1

            return " ".join(sentences)[:length]

        else:
            raise ValueError(f"Unknown distractor_type: {self.distractor_type}")

    def _generate_needle(self, rng: random.Random) -> tuple[str, str, str]:
        """Generate a needle (fact, question, answer)."""
        template, question = rng.choice(self.NEEDLE_TEMPLATES)

        # Generate a random value
        value_type = rng.choice(["word", "number", "code"])
        if value_type == "word":
            value = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(6, 10)))
        elif value_type == "number":
            value = str(rng.randint(1000, 9999))
        else:
            value = "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))

        needle = template.format(value=value)
        return needle, question, value

    def load_samples(
        self, num_samples: int | None = None, seed: int | None = None
    ) -> Iterator[BenchmarkSample]:
        """Generate synthetic NIAH samples."""
        rng = random.Random(seed)
        num_samples = num_samples or 100

        for i in range(num_samples):
            needle, question, answer = self._generate_needle(rng)

            # Determine needle position
            depth = self.needle_depth if self.needle_depth is not None else rng.random()

            # Generate haystack
            haystack_length = self.context_length - len(needle) - 10
            haystack = self._generate_haystack(haystack_length, rng)

            # Insert needle at depth position
            insert_pos = int(len(haystack) * depth)
            # Find a good break point (paragraph or sentence)
            break_pos = haystack.rfind("\n\n", max(0, insert_pos - 100), insert_pos + 100)
            if break_pos == -1:
                break_pos = haystack.rfind(". ", max(0, insert_pos - 50), insert_pos + 50)
            if break_pos == -1:
                break_pos = insert_pos

            context = haystack[:break_pos] + "\n\n" + needle + "\n\n" + haystack[break_pos:]

            yield BenchmarkSample(
                id=f"niah-{i:04d}",
                context=context,
                question=question,
                expected_answer=answer,
                metadata={
                    "needle_depth": depth,
                    "context_length": len(context),
                    "needle": needle,
                },
            )

    def evaluate(self, prediction: str, expected: str | list[str]) -> dict[str, float]:
        """Evaluate prediction - exact match or containment for NIAH."""
        return Metrics.evaluate_standard(prediction, expected)
