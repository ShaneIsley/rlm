"""
OOLONG Benchmarks - Semantic Aggregation over Long Contexts.

OOLONG tests the ability to aggregate information across a long context.
Two variants:
- OolongBenchmark: Standard QA requiring semantic understanding
- OolongPairsBenchmark: Pairwise combinatorial aggregation (hardest setting)

Uses the oolongbench/oolong-real dataset from HuggingFace.
"""

import random
import re
from collections.abc import Iterator
from typing import Any

from benchmarks.base import Benchmark, BenchmarkSample
from benchmarks.metrics import Metrics


class OolongBenchmark(Benchmark):
    """OOLONG benchmark for semantic aggregation.

    Loads samples from the oolongbench/oolong-real HuggingFace dataset.
    Tests ability to answer questions requiring understanding and aggregation
    of information spread across a long context.

    Args:
        subset: Dataset subset to use (e.g., "toy_dnd", "counting", etc.)
    """

    AVAILABLE_SUBSETS = [
        "toy_dnd",
        "counting",
        "retrieval",
        "reasoning",
        "aggregation",
    ]

    def __init__(self, subset: str = "toy_dnd"):
        self.subset = subset
        self._validate_subset()

    def _validate_subset(self):
        """Check that subset is available."""
        if self.subset not in self.AVAILABLE_SUBSETS:
            raise ValueError(f"Unknown subset: {self.subset}. Available: {self.AVAILABLE_SUBSETS}")

    @property
    def name(self) -> str:
        return f"oolong-{self.subset}"

    @property
    def description(self) -> str:
        return f"OOLONG semantic aggregation benchmark ({self.subset})"

    def _load_dataset(self, seed: int | None = None):
        """Load the oolong dataset with streaming."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("Please install datasets: uv pip install datasets") from e

        ds = load_dataset(
            "oolongbench/oolong-real",
            self.subset,
            split="test",
            streaming=True,
        )

        if seed is not None:
            ds = ds.shuffle(seed=seed, buffer_size=1000)

        return ds

    def load_samples(
        self, num_samples: int | None = None, seed: int | None = None
    ) -> Iterator[BenchmarkSample]:
        """Load samples from the OOLONG dataset."""
        ds = self._load_dataset(seed=seed)

        count = 0
        for row in ds:
            if num_samples is not None and count >= num_samples:
                break

            yield BenchmarkSample(
                id=f"oolong-{self.subset}-{count:04d}",
                context=row["context_window_text"],
                question=row["question"],
                expected_answer=row["answer"],
                metadata={
                    "subset": self.subset,
                    "original_id": row.get("id", count),
                },
            )
            count += 1

    def evaluate(self, prediction: str, expected: str | list[str]) -> dict[str, float]:
        """Evaluate using standard metrics (containment + F1)."""
        return Metrics.evaluate_standard(prediction, expected)


class OolongPairsBenchmark(Benchmark):
    """OOLONG-Pairs benchmark for pairwise combinatorial aggregation.

    The hardest setting from the RLM paper. Requires identifying all pairs
    of items that satisfy a given relationship from a long context.

    This is a synthetic benchmark that generates pairwise relationships.
    """

    def __init__(
        self,
        num_items: int = 50,
        num_pairs: int = 25,
        context_length: int = 50_000,
    ):
        """Initialize OOLONG-Pairs benchmark.

        Args:
            num_items: Number of unique items in the context.
            num_pairs: Number of pairs that satisfy the relationship.
            context_length: Approximate target context length.
        """
        self.num_items = num_items
        self.num_pairs = num_pairs
        self.context_length = context_length

    @property
    def name(self) -> str:
        return f"oolong-pairs-{self.num_items}x{self.num_pairs}"

    @property
    def description(self) -> str:
        return f"OOLONG-Pairs: Find all {self.num_pairs} pairs among {self.num_items} items"

    # Item categories and relationships for generating diverse content
    DOMAINS = [
        {
            "items": [
                "Alice",
                "Bob",
                "Charlie",
                "Diana",
                "Eve",
                "Frank",
                "Grace",
                "Henry",
                "Ivy",
                "Jack",
                "Kate",
                "Leo",
                "Mia",
                "Noah",
                "Olivia",
                "Peter",
                "Quinn",
                "Rose",
                "Sam",
                "Tina",
                "Uma",
                "Victor",
                "Wendy",
                "Xavier",
                "Yara",
                "Zack",
            ],
            "relationship": "collaborated with",
            "question": "List all pairs of people who collaborated together.",
            "context_template": "{a} and {b} worked together on a project last year.",
            "distractor_templates": [
                "{a} attended the conference in {city}.",
                "{a} published a paper on {topic}.",
                "{a} received an award for {achievement}.",
            ],
        },
        {
            "items": [
                "Apple",
                "Banana",
                "Cherry",
                "Date",
                "Elderberry",
                "Fig",
                "Grape",
                "Honeydew",
                "Jackfruit",
                "Kiwi",
                "Lemon",
                "Mango",
                "Nectarine",
                "Orange",
                "Papaya",
                "Quince",
                "Raspberry",
                "Strawberry",
                "Tangerine",
                "Ugli",
                "Vanilla",
                "Watermelon",
            ],
            "relationship": "pairs well with",
            "question": "List all pairs of fruits that pair well together in recipes.",
            "context_template": "{a} and {b} create an excellent flavor combination.",
            "distractor_templates": [
                "{a} is commonly grown in {region}.",
                "{a} contains high levels of {nutrient}.",
                "{a} is harvested during {season}.",
            ],
        },
    ]

    def _generate_sample(self, sample_id: int, rng: random.Random) -> BenchmarkSample:
        """Generate a single OOLONG-Pairs sample."""
        domain = rng.choice(self.DOMAINS)
        items = domain["items"].copy()

        # Ensure we have enough items
        while len(items) < self.num_items:
            items.append(f"Item{len(items)}")

        rng.shuffle(items)
        items = items[: self.num_items]

        # Generate pairs
        all_possible = []
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                all_possible.append((a, b))

        rng.shuffle(all_possible)
        true_pairs = all_possible[: self.num_pairs]

        # Generate context with pair statements and distractors
        statements = []

        # Add true pair statements
        for a, b in true_pairs:
            stmt = domain["context_template"].format(a=a, b=b)
            statements.append(stmt)

        # Add distractor statements
        cities = ["London", "Paris", "Tokyo", "New York", "Sydney", "Berlin"]
        topics = ["machine learning", "sustainability", "economics", "design"]
        achievements = ["innovation", "leadership", "research", "creativity"]
        regions = ["California", "Mediterranean", "South America", "Southeast Asia"]
        nutrients = ["vitamin C", "antioxidants", "fiber", "potassium"]
        seasons = ["summer", "fall", "spring", "winter"]

        while len("\n".join(statements)) < self.context_length:
            item = rng.choice(items)
            template = rng.choice(domain["distractor_templates"])

            stmt = template.format(
                a=item,
                city=rng.choice(cities),
                topic=rng.choice(topics),
                achievement=rng.choice(achievements),
                region=rng.choice(regions),
                nutrient=rng.choice(nutrients),
                season=rng.choice(seasons),
            )
            statements.append(stmt)

        # Shuffle all statements
        rng.shuffle(statements)
        context = "\n".join(statements)

        return BenchmarkSample(
            id=f"oolong-pairs-{sample_id:04d}",
            context=context,
            question=domain["question"],
            expected_answer=true_pairs,  # List of tuples
            metadata={
                "num_items": len(items),
                "num_pairs": len(true_pairs),
                "domain": domain["relationship"],
            },
        )

    def load_samples(
        self, num_samples: int | None = None, seed: int | None = None
    ) -> Iterator[BenchmarkSample]:
        """Generate synthetic OOLONG-Pairs samples."""
        rng = random.Random(seed)
        num_samples = num_samples or 50

        for i in range(num_samples):
            yield self._generate_sample(i, rng)

    def _parse_pairs(self, prediction: str) -> set[tuple[str, str]]:
        """Parse pairs from prediction string.

        Handles various formats:
        - (A, B)
        - A and B
        - A - B
        - A with B
        """
        pairs = set()

        # Try to find pairs in various formats
        patterns = [
            r"\(([^,]+),\s*([^)]+)\)",  # (A, B)
            r"([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)",  # A and B
            r"([A-Z][a-z]+)\s*[-–]\s*([A-Z][a-z]+)",  # A - B
            r"([A-Z][a-z]+)\s+with\s+([A-Z][a-z]+)",  # A with B
        ]

        for pattern in patterns:
            matches = re.findall(pattern, prediction)
            for match in matches:
                pairs.add((match[0].strip(), match[1].strip()))

        return pairs

    def evaluate(self, prediction: str, expected: str | list[Any]) -> dict[str, float]:
        """Evaluate pairwise F1 score."""
        predicted_pairs = self._parse_pairs(prediction)

        # Convert expected to set of tuples
        if isinstance(expected, str):
            expected_pairs = self._parse_pairs(expected)
        else:
            expected_pairs = {(a, b) for a, b in expected}

        metrics = Metrics.pairwise_f1(predicted_pairs, expected_pairs)

        # Add "correct" for compatibility (1.0 if F1 > 0.5)
        metrics["correct"] = 1.0 if metrics["f1"] > 0.5 else 0.0

        return metrics

    def default_metrics(self) -> list[str]:
        return ["correct", "precision", "recall", "f1"]
