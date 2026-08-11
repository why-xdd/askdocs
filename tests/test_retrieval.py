from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from askdocs.bm25 import BM25, tokenize
from askdocs.chunking import chunk_markdown
from askdocs.embeddings import TfidfEmbedder
from askdocs.evaluate import Question, compare, evaluate_mode, load_questions
from askdocs.index import Index, build
from askdocs.retrieval import reciprocal_rank_fusion

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def corpus_index() -> Index:
    return build(EXAMPLES / "docs", TfidfEmbedder())


@pytest.fixture(scope="module")
def questions() -> list[Question]:
    return load_questions(EXAMPLES / "questions.json")


# -- tokenising and BM25 ----------------------------------------------------


def test_tokenizer_keeps_identifiers_and_cyrillic_whole():
    assert tokenize("PAY_1004 and utf8") == ["pay", "1004", "and", "utf8"]
    assert tokenize("Запрос к базе") == ["запрос", "к", "базе"]


def test_bm25_ranks_the_document_containing_the_term():
    index = BM25(
        [
            "the deployment guide covers blue-green releases",
            "error PAY_1004 means the idempotency key was reused",
            "testing philosophy and flaky tests",
        ]
    )
    assert index.search("PAY_1004")[0][0] == 1


def test_bm25_idf_is_never_negative():
    """A term in every document must not actively penalise the documents it is in.

    Without the +1 inside the log, any term appearing in more than half the
    corpus gets a negative weight — which produces genuinely baffling rankings
    on small corpora where 'the' is everywhere.
    """
    index = BM25(["the cat sat", "the dog sat", "the bird sat"])
    assert all(value >= 0 for value in index.idf.values())
    assert all(score >= 0 for score in index.score("the"))


def test_bm25_saturates_on_repetition():
    """Twenty mentions must not score ten times a document with two."""
    index = BM25(["timeout " * 20, "timeout timeout", "unrelated content here"])
    scores = index.score("timeout")
    assert scores[0] > scores[1]
    assert scores[0] < scores[1] * 3


def test_empty_corpus_does_not_crash():
    assert BM25([]).search("anything") == []


# -- fusion -----------------------------------------------------------------


def test_rrf_rewards_agreement_between_retrievers():
    """A document both retrievers like beats one that only one ranks first."""
    fused = dict(reciprocal_rank_fusion([[7, 1, 2], [3, 7, 4]]))
    assert max(fused, key=fused.get) == 7


def test_rrf_is_scale_free():
    """Ranks only — the magnitude of the underlying scores cannot leak in."""
    a = reciprocal_rank_fusion([[1, 2, 3]])
    b = reciprocal_rank_fusion([[1, 2, 3]])
    assert a == b


def test_rrf_weights_shift_the_balance():
    lexical, dense = [1, 2], [2, 1]
    even = dict(reciprocal_rank_fusion([lexical, dense]))
    tilted = dict(reciprocal_rank_fusion([lexical, dense], weights=[3.0, 1.0]))

    assert even[1] == pytest.approx(even[2])
    assert tilted[1] > tilted[2]


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])


# -- chunking ---------------------------------------------------------------


def test_chunking_splits_on_headings_and_keeps_them():
    chunks = chunk_markdown(
        "# Title\n\nintro text\n\n## Section A\n\nbody a\n\n## Section B\n\nbody b",
        source="doc.md",
    )
    sections = [chunk.section for chunk in chunks]
    assert "Section A" in sections
    assert "Section B" in sections


def test_oversized_sections_are_split_with_overlap():
    body = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(60))
    chunks = chunk_markdown(f"# Big\n\n{body}", source="doc.md", max_chars=400, overlap=60)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 400 + 60 + 20 for chunk in chunks)
    assert all(chunk.section == "Big" for chunk in chunks)


def test_citation_includes_the_section():
    chunk = chunk_markdown("## Draining\n\nOn SIGTERM the service drains.", "deploy.md")[0]
    assert chunk.citation == "deploy.md#Draining"


def test_document_without_headings_still_chunks():
    assert len(chunk_markdown("just some prose with no headings", "notes.txt")) == 1


# -- embeddings -------------------------------------------------------------


def test_tfidf_vectors_are_unit_length():
    embedder = TfidfEmbedder(dimensions=128).fit(["alpha beta", "gamma delta"])
    norms = np.linalg.norm(embedder.embed(["alpha beta"]), axis=1)
    assert norms == pytest.approx([1.0], abs=1e-5)


def test_character_ngrams_survive_russian_inflection():
    """'запроса' and 'запросу' are one concept; a word-only model sees two.

    This is why the fallback embedder mixes character n-grams into the same
    space — Russian documentation is full of the same noun in five cases.
    """
    embedder = TfidfEmbedder(dimensions=1024).fit(
        ["обработка запроса к базе данных", "совершенно другая тема про графику"]
    )
    vectors = embedder.embed(["обработка запросу к базе данных", "тема про графику"])
    related = float(vectors[0] @ embedder.embed(["обработка запроса к базе данных"])[0])
    unrelated = float(vectors[1] @ embedder.embed(["обработка запроса к базе данных"])[0])

    assert related > unrelated
    assert related > 0.5


# -- end to end -------------------------------------------------------------


def test_index_builds_over_the_example_corpus(corpus_index):
    stats = corpus_index.stats
    assert stats.documents == 8
    assert stats.chunks > 15


def test_exact_identifier_is_retrieved(corpus_index):
    hits = corpus_index.search("PAY_1004", top_k=3)
    assert any("errors.md" in hit.chunk.source for hit in hits)


def test_paraphrased_question_is_retrieved(corpus_index):
    hits = corpus_index.search("how do we ship a new release without downtime", top_k=5)
    assert any("deployment.md" in hit.chunk.source for hit in hits)


def test_hits_carry_both_retriever_ranks(corpus_index):
    hits = corpus_index.search("idempotency key reused", top_k=5)
    assert any(hit.lexical_rank is not None for hit in hits)
    assert any(hit.dense_rank is not None for hit in hits)


def test_unknown_mode_is_rejected(corpus_index):
    with pytest.raises(ValueError):
        corpus_index.search("anything", mode="magic")


def test_index_round_trips_through_sqlite(corpus_index, tmp_path):
    path = tmp_path / "index.db"
    corpus_index.save(path)
    reloaded = Index.load(path)

    assert reloaded.stats.chunks == corpus_index.stats.chunks
    assert np.allclose(reloaded.vectors, corpus_index.vectors, atol=1e-6)

    # The embedder state must survive too, or query vectors would be weighted
    # differently from the indexed ones and land in a different space.
    original = corpus_index.search("PAY_1004", top_k=3)
    restored = reloaded.search("PAY_1004", top_k=3)
    assert [h.chunk.citation for h in original] == [h.chunk.citation for h in restored]


def test_loading_a_missing_index_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Index.load(tmp_path / "nope.db")


def test_feature_buckets_are_stable_across_processes(corpus_index, tmp_path):
    """Regression: a saved index must rank identically in a fresh interpreter.

    The TF-IDF backend once bucketed features with Python's built-in ``hash()``,
    which is seeded randomly per process. The index still returned results — just
    different, wrong ones on every run, because query vectors landed in a
    different space than the indexed ones. Only a separate interpreter can catch
    this; an in-process round trip shares the seed and passes happily.
    """
    import subprocess
    import sys

    path = tmp_path / "index.db"
    corpus_index.save(path)
    query = "how do we ship a new release without downtime"
    expected = [h.chunk.citation for h in corpus_index.search(query, top_k=3, mode="dense")]

    script = (
        "from askdocs.index import Index;"
        f"ix = Index.load(r'{path}');"
        f"print('|'.join(h.chunk.citation for h in ix.search({query!r}, top_k=3, mode='dense')))"
    )

    # Two runs, each with a different hash seed: identical output is only
    # possible if bucketing does not depend on the seed at all.
    outputs = set()
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.add(result.stdout.strip())

    assert len(outputs) == 1, f"ranking changed with the hash seed: {outputs}"
    assert outputs.pop() == "|".join(expected)


# -- evaluation -------------------------------------------------------------


def test_hybrid_is_never_worse_than_either_retriever_alone(corpus_index, questions):
    """The claim the whole design rests on, measured rather than asserted.

    The question set contains both kinds of query: bare identifiers that only
    lexical search finds, and paraphrases that only the dense side reaches.

    Note what this does *not* assert. Hybrid ties dense on recall for this
    corpus; it does not sweep. Writing the assertion as a strict win would mean
    either a false claim or a question set quietly tuned until it passed.
    """
    results = {r.mode: r for r in compare(corpus_index, questions, top_k=5)}

    assert results["hybrid"].recall_at_k >= results["lexical"].recall_at_k
    assert results["hybrid"].recall_at_k >= results["dense"].recall_at_k
    assert results["hybrid"].recall_at_k > 0.9


def test_hybrid_ranks_the_answer_higher_than_either_alone(corpus_index, questions):
    """Where fusion actually pays: MRR, not recall.

    With a fixed context window, a correct chunk at rank 5 often does not
    survive truncation, and a model weights the first passage far above the
    last. Moving the right answer up is the win, even when the same set of
    answers was already being found.
    """
    results = {r.mode: r for r in compare(corpus_index, questions, top_k=5)}

    assert results["hybrid"].mrr > results["lexical"].mrr
    assert results["hybrid"].mrr > results["dense"].mrr


def test_lexical_retrieval_wins_on_bare_identifiers(corpus_index):
    identifiers = [
        Question("PAY_1004", ["errors.md"]),
        Question("PAYMENTS_SHUTDOWN_GRACE", ["deployment.md"]),
        Question("pg_stat_statements", ["oncall.md"]),
    ]
    lexical = evaluate_mode(corpus_index, identifiers, "lexical", top_k=3)
    assert lexical.recall_at_k == 1.0


def test_evaluation_reports_what_it_missed(corpus_index):
    impossible = [Question("what is the airspeed velocity of a swallow", ["nothing.md"])]
    result = evaluate_mode(corpus_index, impossible, "hybrid", top_k=5)

    assert result.recall_at_k == 0.0
    assert result.misses == ["what is the airspeed velocity of a swallow"]


def test_mrr_rewards_ranking_the_answer_first(corpus_index, questions):
    result = evaluate_mode(corpus_index, questions, "hybrid", top_k=5)
    assert 0.0 < result.mrr <= 1.0
    assert result.mrr <= result.recall_at_k


def test_empty_question_set_scores_zero(corpus_index):
    assert evaluate_mode(corpus_index, [], "hybrid").recall_at_k == 0.0
