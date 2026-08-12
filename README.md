<div align="center">

# askdocs

**Ask questions about your own documents. Locally.**
Hybrid retrieval that finds both `PAY_1004` and "what happens when I reuse an idempotency key" — and an eval command that proves it on *your* corpus.

[![CI](https://github.com/why-xdd/askdocs/actions/workflows/ci.yml/badge.svg)](https://github.com/why-xdd/askdocs/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

<img width="580" src="https://raw.githubusercontent.com/why-xdd/askdocs/main/docs/banner.svg" alt="askdocs — local hybrid retrieval, measured: hybrid MRR 0.81"/>

</div>

---

```bash
pip install askdocs

askdocs index ./docs
askdocs ask "how do we ship a new release without downtime"
askdocs eval questions.json          # does any of this actually work?
```

Nothing leaves the machine. It runs with no model download at all, and picks up
a local Ollama model automatically if one is there.

---

## The number this project exists for

Most RAG projects assert that hybrid retrieval helps. This one ships the
measurement as a command, because whether it helps depends entirely on your
documents — lexical search wins on identifier-heavy technical writing and loses
on prose, and you cannot know which you have without running it.

On the [example corpus](examples/) — eight pages of internal documentation, 16
questions, using the built-in no-download embedder:

<img src="https://raw.githubusercontent.com/why-xdd/askdocs/main/docs/terminal.png" alt="askdocs eval: lexical 88% MRR 0.724, dense 94% MRR 0.731, hybrid 94% MRR 0.809" width="100%"/>

Read that carefully, because it is more interesting than a clean sweep. **Hybrid
does not beat dense on recall here — it ties.** What it wins is MRR, by a clear
margin: the same answers, ranked higher. For a system that feeds a fixed context
window to a model, rank is most of what matters — a correct chunk at position 5
often does not survive truncation, and a model given six passages weights the
first far more than the last.

Each retriever also fails differently. Lexical search misses *"I'm new here and
don't know who to talk to"*, which shares almost no vocabulary with the document
that answers it. Dense retrieval misses `PAYMENTS_SHUTDOWN_GRACE`, because an
opaque identifier has no semantics to embed. Fusion covers both, and the one
question all three modes still miss is [listed in the output](#commands) rather
than swept up.

*(An earlier version of this table claimed 100% for hybrid. That number was an
artefact of a bug — the TF-IDF backend bucketed features with Python's
`hash()`, which is seeded per process, so a saved index scored differently on
every run. The fix is described [below](#embeddings-in-order-of-preference);
these are the numbers after it.)*

The question set is [checked in](examples/questions.json) with a note on each
question explaining which retriever it is designed to defeat, so the result can
be argued with rather than taken on trust.

Reproduce it:

```bash
askdocs index examples/docs --embedder tfidf
askdocs eval examples/questions.json
```

If hybrid does *not* win on your corpus, the command says so and tells you which
mode to use instead.

---

## How it works

```
documents ──▶ heading-aware chunks ──┬──▶ BM25 ────────┐
                                     │                  ├──▶ RRF ──▶ rerank ──▶ cited answer
                                     └──▶ embeddings ───┘
```

### Chunking on structure, not character count

Chunk size quietly caps a RAG system's ceiling. Too small and an answer spreads
across three chunks that each retrieve poorly; too large and the embedding
averages five topics into a vector that matches nothing in particular.

So splits follow Markdown headings first — a heading is the author's own
statement about where one idea ends. Only sections too large to embed usefully
are split further, on paragraph then sentence boundaries, with overlap so a fact
sitting on a seam survives in one piece. The section name rides along and becomes
the citation: `deployment.md#Rolling out a new version`.

### Fusion by rank, not by score

The obvious way to combine two retrievers — normalise both score sets and add
them — is subtly broken. BM25 scores and cosine similarities are not on
comparable scales, and their *distributions change per query*: a query with one
rare term produces a huge BM25 spread, a query of common words a flat one.
Min-max normalising then averaging means the effective weighting silently drifts
from query to query, and "0.6 lexical, 0.4 dense" describes nothing reproducible.

**Reciprocal Rank Fusion** throws the scores away and keeps the ranks. Each
retriever contributes `1 / (k + rank)`. Scale-free, nothing to tune, and it wins
in the table above.

### Then a gentle rerank

Fusion works on ranks, so it cannot see *how well* a chunk matched. A light
lexical-overlap pass puts a little of that back, favouring a chunk containing
four of five query terms over one containing two. Deliberately gentle — turn it
up and hybrid retrieval collapses back into keyword search.

### Embeddings, in order of preference

| backend | needs | notes |
|---|---|---|
| `ollama` | Ollama running | `nomic-embed-text` by default. Nothing leaves the machine. |
| `sentence-transformers` | torch | If it is already in your environment. |
| `tfidf` | nothing | Hashed TF-IDF with character n-grams. Always available. |

`--embedder auto` (the default) uses the best one present and tells you which.

The TF-IDF fallback is why the test suite runs in under a second on a cold
checkout, and why the eval numbers above are reproducible by anyone. It mixes
**character n-grams** with word tokens on purpose: Russian inflects heavily —
*запрос / запроса / запросу* is one concept and three tokens — and a word-only
model treats them as unrelated. It is genuinely weaker at paraphrase than a real
model, and the eval reports exactly how much weaker rather than apologising.

Features are bucketed with **blake2b, not Python's `hash()`**. That distinction
is the difference between a working index and a broken one: `hash()` is seeded
randomly per process, so the process that builds an index and the process that
queries it bucket the same word differently. Results still come back — they are
simply wrong, and differently wrong on every run, which is the hardest kind of
bug to notice. There is a regression test that queries a saved index from a
separate interpreter under two different hash seeds and requires identical
rankings; an in-process round trip shares the seed and passes happily either way.

---

## Commands

```bash
# Index. PDFs need: pip install askdocs[pdf]
askdocs index ./docs ./handbook.pdf --out .askdocs/index.db

# Retrieve. --show-ranks reveals which retriever found what.
askdocs ask "what must never appear in logs" -k 5 --show-ranks
askdocs ask "PAY_1004" --mode lexical
askdocs ask "why is the ledger separate" --json

# Generate a written answer with a local model.
askdocs ask "how do I roll back a deploy" --generate --model qwen2.5:7b

# Measure.
askdocs eval questions.json -k 5 --json results.json
```

```
[1] docs/deployment.md#Rolling out a new version  bm25 #3 · vector #9
    Deployments are blue-green. The new colour is brought up alongside the old
    one, health-checked, then the load balancer is flipped in a single operation…
```

---

## Answers are cited, and allowed to say no

When `--generate` is on, the prompt instructs the model to answer only from the
retrieved context and to say plainly when the context does not contain the
answer. Models strongly prefer to fill the gap from memory, and a plausible
answer carrying a citation that does not support it is worse than no answer —
the citation is what makes it look verified.

Without `--generate`, the retrieved passages *are* the output. For "where is
this documented", that is the whole answer and no model is needed.

---

## As a library

```python
from askdocs import build, Index, compare, load_questions

index = build("./docs")
index.save(".askdocs/index.db")

for hit in Index.load(".askdocs/index.db").search("idempotency", top_k=5):
    print(hit.chunk.citation, hit.lexical_rank, hit.dense_rank)

for result in compare(index, load_questions("questions.json")):
    print(result.mode, result.recall_at_k, result.mrr)
```

An index is a single SQLite file containing chunks, vectors *and the embedder's
own state* — so you can hand someone the file and their queries will land in the
same vector space yours did.

---

## Tests

```bash
pytest        # 29 tests
ruff check .
```

They pin behaviour that is easy to get quietly wrong: BM25 IDF never going
negative (without the `+1` inside the log, a common term actively penalises the
documents containing it), term-frequency saturation, RRF rewarding agreement
between retrievers, an index surviving a save/load round trip with identical
rankings, and character n-grams surviving Russian inflection.

---

## What this is not

Not a vector database, and not a framework. There is no agent loop, no chain
abstraction, and no plugin system — about 900 lines of retrieval you can read in
one sitting. If you need billion-scale ANN search, use Qdrant or pgvector; this
is for the case that is actually common, which is a few thousand pages of
documentation on one machine.

MIT © [why-xdd](https://github.com/why-xdd)
