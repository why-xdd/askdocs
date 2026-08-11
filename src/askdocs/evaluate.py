"""Retrieval evaluation.

Every RAG project claims hybrid retrieval helps. Almost none of them measure it
on their own corpus, which is the only place the claim can be true or false —
lexical search wins on identifier-heavy documentation and loses on prose, and
you cannot know which you have without running it.

So this ships as a first-class command, not a notebook. ``askdocs eval`` runs the
same questions through lexical, dense and hybrid retrieval and prints the three
side by side. If hybrid does not win on your documents, the tool says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .index import Index


@dataclass(slots=True)
class Question:
    """A query and the sources that should come back for it.

    Relevance is judged at source level rather than chunk level on purpose:
    labelling chunks means relabelling every time the chunk size changes, and the
    labels would silently rot. A source is stable.
    """

    query: str
    relevant: list[str]
    note: str = ""

    def matches(self, source: str) -> bool:
        """Substring match, so labels can be filenames rather than full paths."""
        return any(marker in source for marker in self.relevant)


@dataclass(slots=True)
class ModeResult:
    mode: str
    recall_at_k: float
    mrr: float
    hit_rate: float
    misses: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "misses": self.misses,
        }


def load_questions(path: Path) -> list[Question]:
    """Read a JSON list of ``{query, relevant, note}``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Question(
            query=item["query"],
            relevant=item["relevant"],
            note=item.get("note", ""),
        )
        for item in payload
    ]


def evaluate_mode(
    index: Index,
    questions: list[Question],
    mode: str,
    top_k: int = 5,
    rerank: bool = True,
) -> ModeResult:
    """Score one retrieval mode over the question set.

    *Recall@k* — did any relevant source appear in the top k? The number that
    matters most, because a generator cannot cite what retrieval never returned.

    *MRR* — how high did the first relevant result land? Distinguishes "found it
    at rank 1" from "found it at rank 5", which recall alone cannot see.
    """
    if not questions:
        return ModeResult(mode, 0.0, 0.0, 0.0)

    hits = 0
    reciprocal_total = 0.0
    misses: list[str] = []

    for question in questions:
        results = index.search(question.query, top_k=top_k, mode=mode, rerank=rerank)
        rank = next(
            (
                position
                for position, hit in enumerate(results, start=1)
                if question.matches(hit.chunk.source)
            ),
            None,
        )
        if rank is None:
            misses.append(question.query)
        else:
            hits += 1
            reciprocal_total += 1 / rank

    count = len(questions)
    return ModeResult(
        mode=mode,
        recall_at_k=hits / count,
        mrr=reciprocal_total / count,
        hit_rate=hits / count,
        misses=misses,
    )


def compare(
    index: Index,
    questions: list[Question],
    top_k: int = 5,
    rerank: bool = True,
) -> list[ModeResult]:
    """Run all three modes and return their results in a stable order."""
    return [
        evaluate_mode(index, questions, mode, top_k, rerank)
        for mode in ("lexical", "dense", "hybrid")
    ]
