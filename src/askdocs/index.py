"""Building an index and persisting it.

One SQLite file holds chunks, vectors and the embedder's own state. Keeping the
embedder state in the index is not incidental bookkeeping: a TF-IDF model that
re-learned its document frequencies at query time would weight query terms
differently than it weighted them at index time, and the two vector spaces would
no longer be comparable. Storing it makes an index self-contained and
reproducible — you can hand someone the file.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_file, find_documents
from .embeddings import Embedder, TfidfEmbedder, resolve
from .retrieval import HybridRetriever

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    source     TEXT NOT NULL,
    section    TEXT NOT NULL DEFAULT '',
    ordinal    INTEGER NOT NULL DEFAULT 0,
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end   INTEGER NOT NULL DEFAULT 0,
    metadata   TEXT NOT NULL DEFAULT '{}',
    vector     BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
"""


@dataclass(slots=True)
class IndexStats:
    documents: int
    chunks: int
    embedder: str
    dimensions: int

    def as_dict(self) -> dict:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "embedder": self.embedder,
            "dimensions": self.dimensions,
        }


class Index:
    """A queryable corpus: chunks, their vectors, and the embedder that made them."""

    def __init__(
        self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder
    ) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.retriever = HybridRetriever(chunks, vectors, self._embed_query)

    def _embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed([query])[0]

    @property
    def stats(self) -> IndexStats:
        return IndexStats(
            documents=len({chunk.source for chunk in self.chunks}),
            chunks=len(self.chunks),
            embedder=self.embedder.name,
            dimensions=self.vectors.shape[1] if len(self.vectors) else 0,
        )

    def search(self, query: str, top_k: int = 5, **kwargs):
        return self.retriever.search(query, top_k=top_k, **kwargs)

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

        with sqlite3.connect(path) as connection:
            connection.executescript(SCHEMA)
            connection.executemany(
                "INSERT INTO chunks "
                "(text, source, section, ordinal, char_start, char_end, metadata, vector) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.text,
                        chunk.source,
                        chunk.section,
                        chunk.ordinal,
                        chunk.char_start,
                        chunk.char_end,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        # float32 keeps the file half the size of float64 at a
                        # precision far below what cosine ranking can notice.
                        vector.astype(np.float32).tobytes(),
                    )
                    for chunk, vector in zip(self.chunks, self.vectors, strict=True)
                ],
            )

            meta = {
                "embedder": self.embedder.name,
                "dimensions": str(self.vectors.shape[1] if len(self.vectors) else 0),
            }
            if isinstance(self.embedder, TfidfEmbedder):
                meta["embedder_state"] = json.dumps(self.embedder.state())

            connection.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                list(meta.items()),
            )

    @classmethod
    def load(cls, path: Path, embedder: Embedder | None = None) -> Index:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no index at {path}")

        with sqlite3.connect(path) as connection:
            meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
            rows = connection.execute(
                "SELECT text, source, section, ordinal, char_start, char_end, "
                "metadata, vector FROM chunks ORDER BY id"
            ).fetchall()

        dimensions = int(meta.get("dimensions", 0))
        chunks = [
            Chunk(
                text=row[0],
                source=row[1],
                section=row[2],
                ordinal=row[3],
                char_start=row[4],
                char_end=row[5],
                metadata=json.loads(row[6]),
            )
            for row in rows
        ]
        vectors = (
            np.array([np.frombuffer(row[7], dtype=np.float32) for row in rows])
            if rows
            else np.zeros((0, dimensions), dtype=np.float32)
        )

        if embedder is None:
            state = meta.get("embedder_state")
            if state:
                embedder = TfidfEmbedder.from_state(json.loads(state))
            else:
                # A remote or model-backed embedder cannot be restored from the
                # file; rebuild the same kind and let it reconnect.
                embedder = resolve("auto")

        return cls(chunks, vectors, embedder)


def build(
    paths: list[Path] | Path,
    embedder: Embedder | None = None,
    max_chars: int = 1200,
    overlap: int = 120,
    progress: Callable[[Path], None] | None = None,
) -> Index:
    """Chunk documents, embed them, and return a queryable index."""
    roots = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]

    documents: list[Path] = []
    for root in roots:
        documents.extend(find_documents(root))

    chunks: list[Chunk] = []
    for document in documents:
        if progress is not None:
            progress(document)
        chunks.extend(chunk_file(document, max_chars, overlap))

    embedder = embedder or resolve("auto")
    texts = [chunk.text for chunk in chunks]

    # TF-IDF needs the corpus before it can weight anything; model backends do not.
    if isinstance(embedder, TfidfEmbedder):
        embedder.fit(texts)

    vectors = (
        embedder.embed(texts)
        if texts
        else np.zeros((0, embedder.dimensions), dtype=np.float32)
    )
    return Index(chunks, vectors, embedder)
