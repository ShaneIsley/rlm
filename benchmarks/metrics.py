"""
Evaluation metrics for RLM benchmarks.

Provides common metrics used across benchmark evaluations:
- Exact match
- Containment (answer in prediction or vice versa)
- Token-level F1
- Normalized string comparison
"""

import re
import string
from collections import Counter


class Metrics:
    """Collection of evaluation metrics for benchmark scoring."""

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for comparison.

        - Lowercase
        - Remove punctuation
        - Collapse whitespace
        - Strip leading/trailing whitespace
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def exact_match(prediction: str, expected: str | list[str]) -> bool:
        """Check if prediction exactly matches expected (after normalization).

        Args:
            prediction: Model prediction.
            expected: Single expected answer or list of valid answers.

        Returns:
            True if normalized prediction matches any expected answer.
        """
        pred_norm = Metrics.normalize(prediction)

        if isinstance(expected, str):
            expected = [expected]

        return any(pred_norm == Metrics.normalize(exp) for exp in expected)

    @staticmethod
    def containment(prediction: str, expected: str | list[str]) -> bool:
        """Check if expected is contained in prediction or vice versa.

        More lenient than exact_match - useful when answers may be
        embedded in longer responses.

        Args:
            prediction: Model prediction.
            expected: Single expected answer or list of valid answers.

        Returns:
            True if any containment relationship exists.
        """
        pred_norm = Metrics.normalize(prediction)

        if isinstance(expected, str):
            expected = [expected]

        for exp in expected:
            exp_norm = Metrics.normalize(exp)
            if exp_norm in pred_norm or pred_norm in exp_norm:
                return True

        return False

    @staticmethod
    def token_f1(prediction: str, expected: str | list[str]) -> float:
        """Compute token-level F1 score.

        Treats prediction and expected as bags of tokens and computes
        precision, recall, and F1.

        Args:
            prediction: Model prediction.
            expected: Single expected answer or list of valid answers.
                      If list, returns max F1 across all expected answers.

        Returns:
            F1 score between 0.0 and 1.0.
        """
        if isinstance(expected, str):
            expected = [expected]

        pred_tokens = Metrics.normalize(prediction).split()

        if not pred_tokens:
            return 0.0

        max_f1 = 0.0
        for exp in expected:
            exp_tokens = Metrics.normalize(exp).split()

            if not exp_tokens:
                continue

            pred_counter = Counter(pred_tokens)
            exp_counter = Counter(exp_tokens)

            common = sum((pred_counter & exp_counter).values())

            if common == 0:
                continue

            precision = common / len(pred_tokens)
            recall = common / len(exp_tokens)
            f1 = 2 * precision * recall / (precision + recall)

            max_f1 = max(max_f1, f1)

        return max_f1

    @staticmethod
    def evaluate_standard(prediction: str, expected: str | list[str]) -> dict[str, float]:
        """Standard evaluation combining multiple metrics.

        Returns:
            Dictionary with:
                - "correct": 1.0 if exact_match or containment, else 0.0
                - "exact_match": 1.0 if exact match, else 0.0
                - "containment": 1.0 if containment, else 0.0
                - "f1": Token-level F1 score
        """
        exact = Metrics.exact_match(prediction, expected)
        contained = Metrics.containment(prediction, expected)
        f1 = Metrics.token_f1(prediction, expected)

        return {
            "correct": 1.0 if (exact or contained) else 0.0,
            "exact_match": 1.0 if exact else 0.0,
            "containment": 1.0 if contained else 0.0,
            "f1": f1,
        }

    @staticmethod
    def pairwise_f1(
        predicted_pairs: set[tuple[str, str]],
        expected_pairs: set[tuple[str, str]],
    ) -> dict[str, float]:
        """Compute F1 for pairwise predictions (e.g., OOLONG-Pairs).

        Both predicted and expected should be sets of (item1, item2) tuples.
        Pairs are treated as unordered (a, b) == (b, a).

        Args:
            predicted_pairs: Set of predicted pairs.
            expected_pairs: Set of expected pairs.

        Returns:
            Dictionary with precision, recall, and f1.
        """

        # Normalize pairs to be order-independent
        def normalize_pair(p: tuple[str, str]) -> tuple[str, str]:
            return tuple(sorted([Metrics.normalize(p[0]), Metrics.normalize(p[1])]))

        pred_normalized = {normalize_pair(p) for p in predicted_pairs}
        exp_normalized = {normalize_pair(p) for p in expected_pairs}

        if not pred_normalized and not exp_normalized:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if not pred_normalized:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not exp_normalized:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        common = len(pred_normalized & exp_normalized)
        precision = common / len(pred_normalized)
        recall = common / len(exp_normalized)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}
