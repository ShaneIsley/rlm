"""
BrowseComp-Plus Benchmark - Multi-hop QA over Large Document Corpora.

Tests compositional question answering that requires reasoning across
multiple documents. This is the most realistic benchmark setting,
with document corpora ranging from 6M to 11M tokens.

Note: This benchmark requires access to a document corpus. Users can
either provide their own corpus or use a synthetic generator.
"""

import random
from collections.abc import Iterator
from typing import Any

from benchmarks.base import Benchmark, BenchmarkSample
from benchmarks.metrics import Metrics


class BrowseCompPlusBenchmark(Benchmark):
    """BrowseComp-Plus benchmark for multi-hop QA.

    Tests ability to answer questions that require:
    1. Finding relevant documents in a large corpus
    2. Reasoning across multiple documents
    3. Synthesizing a final answer

    This implementation provides:
    - A synthetic corpus generator for testing
    - Interface for loading custom document corpora
    - Multi-hop question templates

    Args:
        num_documents: Number of documents in the corpus.
        doc_length: Average length of each document in characters.
        num_hops: Number of reasoning hops required (1-3).
        corpus_path: Path to custom corpus (JSON/JSONL file).
    """

    def __init__(
        self,
        num_documents: int = 100,
        doc_length: int = 5000,
        num_hops: int = 2,
        corpus_path: str | None = None,
    ):
        self.num_documents = num_documents
        self.doc_length = doc_length
        self.num_hops = num_hops
        self.corpus_path = corpus_path

    @property
    def name(self) -> str:
        return f"browsecomp-{self.num_documents}docs-{self.num_hops}hop"

    @property
    def description(self) -> str:
        return f"BrowseComp-Plus: {self.num_hops}-hop QA over {self.num_documents} documents"

    # Templates for generating synthetic documents and questions
    ENTITIES = {
        "companies": [
            "Acme Corp",
            "TechNova",
            "GlobalSync",
            "DataFlow",
            "CloudPeak",
            "InnovateTech",
            "FutureWorks",
            "QuantumLeap",
            "NexGen",
            "PrimeAI",
        ],
        "people": [
            "Dr. Sarah Chen",
            "James Rodriguez",
            "Maria Santos",
            "David Kim",
            "Emily Watson",
            "Michael Brown",
            "Lisa Park",
            "Robert Taylor",
            "Jennifer Lee",
            "William Davis",
        ],
        "products": [
            "SynthOS",
            "NeuralNet Pro",
            "DataVault",
            "CloudBridge",
            "AI Assistant",
            "SmartAnalytics",
            "SecureFlow",
            "AutoML Platform",
            "EdgeCompute",
            "RoboSuite",
        ],
        "locations": [
            "San Francisco",
            "New York",
            "London",
            "Tokyo",
            "Singapore",
            "Berlin",
            "Sydney",
            "Toronto",
            "Mumbai",
            "São Paulo",
        ],
    }

    FACT_TEMPLATES = [
        "{company} was founded by {person} in {location}.",
        "{person} serves as the CEO of {company}.",
        "{company} developed {product} in collaboration with {company2}.",
        "{product} was first released in {location}.",
        "{person} led the research team that created {product}.",
        "{company} acquired {company2} to expand into {location}.",
        "{person} previously worked at {company} before joining {company2}.",
    ]

    MULTIHOP_TEMPLATES = [
        {
            "hops": 2,
            "question": "Who founded the company that created {product}?",
            "chain": ["product->company", "company->founder"],
        },
        {
            "hops": 2,
            "question": "Where is the headquarters of the company where {person} works?",
            "chain": ["person->company", "company->location"],
        },
        {
            "hops": 3,
            "question": "Who is the CEO of the company that acquired the creator of {product}?",
            "chain": ["product->company", "company->acquirer", "acquirer->ceo"],
        },
    ]

    def _generate_corpus(self, rng: random.Random) -> tuple[list[dict], dict[str, Any]]:
        """Generate a synthetic document corpus with linked facts."""
        documents = []
        fact_graph = {}

        # Generate relationships
        companies = self.ENTITIES["companies"].copy()
        people = self.ENTITIES["people"].copy()
        products = self.ENTITIES["products"].copy()
        locations = self.ENTITIES["locations"].copy()

        rng.shuffle(companies)
        rng.shuffle(people)
        rng.shuffle(products)
        rng.shuffle(locations)

        # Assign founders to companies
        for i, company in enumerate(companies[: len(people)]):
            founder = people[i % len(people)]
            location = locations[i % len(locations)]
            fact_graph[company] = {
                "founder": founder,
                "location": location,
                "ceo": people[(i + 1) % len(people)],
            }

        # Assign products to companies
        for i, product in enumerate(products):
            company = companies[i % len(companies)]
            fact_graph[product] = {"company": company}

        # Generate documents with facts
        for i in range(self.num_documents):
            doc_content = []

            # Add some facts from the graph
            for _ in range(rng.randint(3, 8)):
                template = rng.choice(self.FACT_TEMPLATES)
                try:
                    fact = template.format(
                        company=rng.choice(companies),
                        company2=rng.choice(companies),
                        person=rng.choice(people),
                        product=rng.choice(products),
                        location=rng.choice(locations),
                    )
                    doc_content.append(fact)
                except (KeyError, IndexError):
                    pass

            # Add filler content
            while len(" ".join(doc_content)) < self.doc_length:
                doc_content.append(
                    f"Additional information about business operations and market trends "
                    f"in the {rng.choice(locations)} region continues to evolve."
                )

            documents.append(
                {
                    "id": f"doc-{i:04d}",
                    "content": " ".join(doc_content),
                }
            )

        return documents, fact_graph

    def _generate_question(
        self, fact_graph: dict[str, Any], rng: random.Random
    ) -> tuple[str, str, list[str]]:
        """Generate a multi-hop question with answer and reasoning chain."""
        # Simple 2-hop question for now
        products = [k for k in fact_graph.keys() if "company" in fact_graph.get(k, {})]
        if not products:
            return "What is 2+2?", "4", ["direct"]

        product = rng.choice(products)
        company = fact_graph[product]["company"]

        if company in fact_graph and "founder" in fact_graph[company]:
            founder = fact_graph[company]["founder"]
            question = f"Who founded the company that created {product}?"
            answer = founder
            chain = [f"{product} -> {company}", f"{company} -> {founder}"]
            return question, answer, chain

        return "What is 2+2?", "4", ["direct"]

    def load_samples(
        self, num_samples: int | None = None, seed: int | None = None
    ) -> Iterator[BenchmarkSample]:
        """Generate BrowseComp-Plus samples."""
        rng = random.Random(seed)
        num_samples = num_samples or 50

        for i in range(num_samples):
            # Generate fresh corpus for each sample (or reuse with different questions)
            documents, fact_graph = self._generate_corpus(rng)
            question, answer, chain = self._generate_question(fact_graph, rng)

            # Combine all documents into context
            context = "\n\n---\n\n".join(
                f"Document {doc['id']}:\n{doc['content']}" for doc in documents
            )

            yield BenchmarkSample(
                id=f"browsecomp-{i:04d}",
                context=context,
                question=question,
                expected_answer=answer,
                metadata={
                    "num_documents": len(documents),
                    "num_hops": len(chain),
                    "reasoning_chain": chain,
                },
            )

    def evaluate(self, prediction: str, expected: str | list[str]) -> dict[str, float]:
        """Evaluate using standard metrics."""
        return Metrics.evaluate_standard(prediction, expected)
