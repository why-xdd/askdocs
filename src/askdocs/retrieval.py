"""Hybrid retrieval: BM25 and vectors, fused, then reranked.

The fusion step is the one worth reading. The obvious approach — normalise both
score sets and add them — is subtly broken, because BM25 scores and cosine
similarities are not on comparable scales and their *distributions* differ per
query. A query with one rare term produces a huge BM25 spread; a query of common
words produces a flat one. Min-max normalising both then averaging means the
weighting silently changes from query to query, and nobody can tell you what
"0.6 lexical, 0.4 dense" actually did.

Reciprocal Rank Fusion sidesteps this by throwing the scores away and keeping
only the ranks. A document's contribution is ``1 / (k + rank)`` from each
retriever. It is scale-free, needs no tuning, and in the eval below it beats
either retriever alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25, tokenize
from .chunking import Chunk

# The constant from Cormack et al. Its job is to flatten the difference between
# rank 1 and rank 2 so that one retriever's confident-but-wrong top hit cannot
# dominate the other's agreement further down.
RRF_K = 60


@dataclass(slots=True)
class Hit:
    """A retrieved chunk with the evidence for why it was retrieved."""

    chunk: Chunk
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None

    def as_dict(self) -> dict:
        return {
            "text": self.chunk.text,
            "citation": self.chunk.citation,
            "source": self.chunk.source,
            "section": self.chunk.section,
            "score": round(self.score, 5),
            "lexical_rank": self.lexical_rank,
            "dense_rank": self.dense_rank,
        }


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked ID lists into one ranking.

    Args:
        rankings: One list of document indices per retriever, best first.
        k: Rank damping. Larger values flatten the head of each list.
        weights: Optional per-retriever weight. Defaults to equal.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + weight / (k + rank)

    return sorted(fused.items(), key=lambda pair: -pair[1])


def lexical_overlap_rerank(query: str, hits: list[Hit], weight: float = 0.3) -> list[Hit]:
    """Nudge results by how much of the query actually appears in the chunk.

    A cheap stand-in for a cross-encoder. Fusion works on ranks and therefore
    cannot see *how* well a chunk matched; this puts a little of that signal
    back, favouring a chunk that contains four of the five query terms over one
    that contains two but ranked well in both lists.

    Deliberately gentle. A large weight turns hybrid retrieval back into keyword
    search and loses the paraphrase matching the dense side was there to provide.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return hits

    best = max((hit.score for hit in hits), default=1.0) or 1.0
    for hit in hits:
        chunk_terms = set(tokenize(hit.chunk.text))
        coverage = len(query_terms & chunk_terms) / len(query_terms)
        hit.score += weight * coverage * best

    return sorted(hits, key=lambda hit: -hit.score)


class HybridRetriever:
    """BM25 + dense vectors over one corpus of chunks."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embed_query,
        rrf_k: int = RRF_K,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        self.chunks = chunks
        self.vectors = vectors
        self.embed_query = embed_query
        self.rrf_k = rrf_k
        self.bm25 = BM25([chunk.text for chunk in chunks])

    def _dense_ranking(self, query: str, top_k: int) -> list[int]:
        if not len(self.vectors):
            return []
        similarity = self.vectors @ self.embed_query(query)
        # argpartition then sort: O(n) to find the candidates, and only the
        # candidates get sorted. Sorting the whole corpus per query is wasted
        # work once it is more than a few thousand chunks.
        count = min(top_k, len(similarity))
        candidates = np.argpartition(-similarity, count - 1)[:count]
        return [int(i) for i in candidates[np.argsort(-similarity[candidates])]]

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidates: int = 30,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> list[Hit]:
        """Retrieve for a query.

        ``candidates`` is the depth each retriever contributes before fusion. It
        should comfortably exceed ``top_k``: fusion can only promote a document
        that at least one retriever surfaced, so a shallow candidate pool throws
        away the recall hybrid retrieval exists to gain.
        """
        lexical = [index for index, _ in self.bm25.search(query, candidates)]
        dense = self._dense_ranking(query, candidates) if mode != "lexical" else []

        if mode == "lexical":
            rankings = [lexical]
        elif mode == "dense":
            rankings = [dense]
        elif mode == "hybrid":
            rankings = [lexical, dense]
        else:
            raise ValueError(f"unknown mode {mode!r}")

        lexical_positions = {index: rank for rank, index in enumerate(lexical, 1)}
        dense_positions = {index: rank for rank, index in enumerate(dense, 1)}

        hits = [
            Hit(
                chunk=self.chunks[index],
                score=score,
                lexical_rank=lexical_positions.get(index),
                dense_rank=dense_positions.get(index),
            )
            for index, score in reciprocal_rank_fusion(rankings, self.rrf_k)
        ]

        if rerank:
            hits = lexical_overlap_rerank(query, hits)

        return hits[:top_k]
