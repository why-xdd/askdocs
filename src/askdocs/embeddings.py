"""Embedding backends.

Three of them, behind one protocol:

* :class:`OllamaEmbedder` — a real local model, if Ollama is running.
* :class:`SentenceTransformerEmbedder` — a real local model, if torch is installed.
* :class:`TfidfEmbedder` — no model at all, and the reason ``askdocs`` runs the
  moment it is installed.

That last one matters more than it looks. A retrieval system whose test suite
needs a 400 MB download and a running daemon does not get tested, and a demo
that cannot start is not a demo. The TF-IDF backend is genuinely weaker at
paraphrase, and the eval harness reports exactly how much weaker — a number is
more honest than an apology.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .bm25 import tokenize


def stable_bucket(kind: str, token: str, dimensions: int) -> int:
    """Hash a feature to a bucket, identically in every process, forever.

    Python's built-in ``hash()`` is seeded randomly per process (PEP 456, on by
    default since 3.3). Using it here silently destroys the whole point of a
    saved index: the process that indexes and the process that queries bucket
    the same word differently, so query vectors land in a different space and
    ranking degenerates to noise. It looks like it works, because results still
    come back — they are just wrong, differently on every run.

    blake2b is deterministic across processes, machines and versions.
    """
    digest = hashlib.blake2b(f"{kind}:{token}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit-length vectors."""

    name: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        ...


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine similarity is a plain dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class TfidfEmbedder:
    """Hashed TF-IDF with character n-grams. Fits in memory, needs no model.

    Two design choices carry it:

    *Hashing* instead of a learned vocabulary, so a vector can be produced for a
    query containing words the corpus never had, without rebuilding anything.

    *Character n-grams alongside word tokens*, which is what lets it survive
    morphology. Russian inflects heavily — "запрос", "запроса", "запросу" are one
    concept and three tokens — and a word-only model treats them as unrelated.
    Overlapping 4-grams share most of their surface, so they land near each other.
    """

    def __init__(self, dimensions: int = 512, ngram: int = 4) -> None:
        self.name = f"tfidf-hash-{dimensions}"
        self.dimensions = dimensions
        self.ngram = ngram
        self._idf: dict[int, float] = {}
        self._documents = 0

    def _features(self, text: str) -> Counter[int]:
        """Bucket word tokens and character n-grams into the same vector space."""
        features: Counter[int] = Counter()
        lowered = text.lower()

        for token in tokenize(lowered):
            features[stable_bucket("w", token, self.dimensions)] += 1

        stripped = " ".join(tokenize(lowered))
        for i in range(max(0, len(stripped) - self.ngram + 1)):
            gram = stripped[i : i + self.ngram]
            if " " not in gram:  # n-grams spanning a word boundary are mostly noise
                features[stable_bucket("c", gram, self.dimensions)] += 1

        return features

    def fit(self, texts: Sequence[str]) -> TfidfEmbedder:
        """Learn document frequencies from the corpus.

        Called once at index time. Queries are embedded with the IDF learned
        here, never re-fitted — a query is one document and would give every
        term an identical, useless weight.
        """
        document_frequency: Counter[int] = Counter()
        for text in texts:
            document_frequency.update(self._features(text).keys())

        self._documents = len(texts)
        self._idf = {
            bucket: math.log(1 + self._documents / (frequency + 1))
            for bucket, frequency in document_frequency.items()
        }
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for bucket, count in self._features(text).items():
                # Sub-linear term frequency: the tenth occurrence says far less
                # than the second.
                vectors[row, bucket] = (1 + math.log(count)) * self._idf.get(bucket, 1.0)
        return normalise(vectors)

    def state(self) -> dict:
        return {
            "dimensions": self.dimensions,
            "ngram": self.ngram,
            "documents": self._documents,
            "idf": {str(k): v for k, v in self._idf.items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> TfidfEmbedder:
        embedder = cls(dimensions=state["dimensions"], ngram=state["ngram"])
        embedder._documents = state["documents"]
        embedder._idf = {int(k): v for k, v in state["idf"].items()}
        return embedder


class OllamaEmbedder:
    """Embeddings from a local Ollama daemon.

    Nothing leaves the machine, which is the entire reason to run a local model
    over documents you would not paste into a hosted API.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._dimensions = 0

    @property
    def dimensions(self) -> int:
        if not self._dimensions:
            self._dimensions = len(self.embed(["dimension probe"])[0])
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        vectors = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                response = client.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                vectors.append(response.json()["embedding"])
        return normalise(np.array(vectors, dtype=np.float32))

    def available(self) -> bool:
        """Whether the daemon is actually reachable, so the CLI can fall back."""
        try:
            import httpx

            with httpx.Client(timeout=2.0) as client:
                return client.get(f"{self.host}/api/tags").status_code == 200
        except Exception:
            return False


class SentenceTransformerEmbedder:
    """A sentence-transformers model, when torch is already in the environment."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model}"
        self._model = SentenceTransformer(model)
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return normalise(
            np.array(self._model.encode(list(texts), show_progress_bar=False))
        )


def resolve(backend: str = "auto", **kwargs) -> Embedder:
    """Pick a backend by name, or the best available one.

    ``auto`` prefers a real model and falls back rather than failing: a tool that
    refuses to start because a daemon is down is less useful than one that starts
    and says which backend it used.
    """
    if backend == "tfidf":
        return TfidfEmbedder(**kwargs)
    if backend == "ollama":
        return OllamaEmbedder(**kwargs)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(**kwargs)
    if backend != "auto":
        raise ValueError(f"unknown embedding backend {backend!r}")

    ollama = OllamaEmbedder()
    if ollama.available():
        return ollama
    return TfidfEmbedder()
