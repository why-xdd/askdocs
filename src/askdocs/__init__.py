"""askdocs — ask questions about your own documents, locally.

Heading-aware chunking, BM25 and dense retrieval fused by reciprocal rank,
cited answers, and an eval harness that checks whether the hybrid actually helps
on *your* corpus.
"""

from .answer import Answer, OllamaGenerator, answer
from .bm25 import BM25, tokenize
from .chunking import Chunk, chunk_file, chunk_markdown, find_documents
from .embeddings import (
    Embedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
    resolve,
)
from .evaluate import ModeResult, Question, compare, evaluate_mode, load_questions
from .index import Index, IndexStats, build
from .retrieval import Hit, HybridRetriever, reciprocal_rank_fusion

__version__ = "0.1.0"

__all__ = [
    "BM25",
    "Answer",
    "Chunk",
    "Embedder",
    "Hit",
    "HybridRetriever",
    "Index",
    "IndexStats",
    "ModeResult",
    "OllamaEmbedder",
    "OllamaGenerator",
    "Question",
    "SentenceTransformerEmbedder",
    "TfidfEmbedder",
    "answer",
    "build",
    "chunk_file",
    "chunk_markdown",
    "compare",
    "evaluate_mode",
    "find_documents",
    "load_questions",
    "reciprocal_rank_fusion",
    "resolve",
    "tokenize",
]
