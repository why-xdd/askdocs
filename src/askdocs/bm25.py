"""BM25 lexical retrieval.

Dense embeddings are the fashionable half of RAG and the weaker half on their
own. They cannot reliably retrieve an exact token they have never seen — an
error code, a config key, a person's surname, a version number. Those are
precisely the things people search technical documents for, and BM25 finds them
by construction.

This is Robertson's BM25 Okapi with the standard parameters, written out rather
than pulled from a library: it is forty lines, and owning it means the tokeniser
can match the one used for embeddings instead of quietly disagreeing with it.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Unicode-aware so Cyrillic tokenises as well as Latin. ``\w`` under ``re.UNICODE``
# would also swallow digits inside identifiers, which is what we want here:
# "utf-8" and "python3" should stay whole.
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

# Term-frequency saturation. Above ~k1 occurrences, more repeats stop adding
# much — a chunk that says "timeout" twenty times is not ten times more about
# timeouts than one that says it twice.
K1 = 1.5

# Length normalisation. At b=0.75 a long chunk is penalised, but not so hard
# that a genuinely thorough section loses to a one-line heading.
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. The same function indexes and queries."""
    return TOKEN.findall(text.lower())


class BM25:
    """An in-memory BM25 index over a fixed corpus.

    Built once at index time and kept whole in memory. For the corpus sizes this
    tool targets — a team's documentation, a codebase's docs directory, a few
    thousand pages of PDF — that is a few megabytes and far simpler than
    maintaining an inverted index on disk.
    """

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.term_frequencies: list[Counter[str]] = []
        self.lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for document in documents:
            tokens = tokenize(document)
            counts = Counter(tokens)
            self.term_frequencies.append(counts)
            self.lengths.append(len(tokens))
            document_frequency.update(counts.keys())

        self.count = len(documents)
        self.average_length = (
            sum(self.lengths) / self.count if self.count else 0.0
        )
        self.idf = {
            term: self._idf(frequency) for term, frequency in document_frequency.items()
        }

    def _idf(self, document_frequency: int) -> float:
        """Robertson-Sparck Jones IDF with the +0.5 smoothing.

        The ``+1`` inside the log keeps this non-negative. Without it, a term
        appearing in more than half the corpus gets a negative weight, and
        matching a common word actively *hurts* a chunk's score — which produces
        baffling results on small corpora where "the" is in every document.
        """
        return math.log(
            1 + (self.count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def score(self, query: str) -> list[float]:
        """Score every document against the query."""
        scores = [0.0] * self.count
        if not self.count:
            return scores

        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, counts in enumerate(self.term_frequencies):
                frequency = counts.get(term)
                if not frequency:
                    continue
                norm = 1 - B + B * (self.lengths[index] / self.average_length)
                scores[index] += idf * (
                    frequency * (K1 + 1) / (frequency + K1 * norm)
                )
        return scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ``(index, score)`` for the best matches, best first."""
        scored = [
            (index, score) for index, score in enumerate(self.score(query)) if score > 0
        ]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]
