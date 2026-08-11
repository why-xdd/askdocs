"""Turning retrieved chunks into a cited answer.

The generation step is optional and deliberately thin. Retrieval is the part
that decides whether an answer can be correct at all, and it is the part this
project is about; swapping the model behind :class:`OllamaGenerator` should not
require touching anything else.

The prompt does one unusual thing: it instructs the model to answer *only* from
the context and to say so when the context does not contain the answer. Models
are strongly inclined to fill the gap from memory, and a plausible answer with a
citation pointing at a chunk that does not support it is worse than no answer —
the citation makes it look verified.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import Hit

SYSTEM_PROMPT = """You answer questions using only the numbered context below.

Rules:
- Use only the context. Do not add facts from your own knowledge.
- Cite the sources you used as [1], [2] inline, matching the context numbers.
- If the context does not contain the answer, say exactly that and stop. Do not
  guess, and do not pad the answer with related information that was not asked for.
- Answer in the language of the question."""


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[str]
    hits: list[Hit]

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "citations": self.citations,
            "hits": [hit.as_dict() for hit in self.hits],
        }


def build_context(hits: list[Hit], max_chars: int = 6000) -> str:
    """Number the chunks so the model has something concrete to cite.

    Truncated by total length rather than count: ten short chunks are cheaper and
    more useful than three long ones, and a fixed count would waste the budget
    on whichever the chunker happened to produce.
    """
    parts: list[str] = []
    used = 0
    for position, hit in enumerate(hits, start=1):
        block = f"[{position}] ({hit.chunk.citation})\n{hit.chunk.text}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


class OllamaGenerator:
    """Answer generation via a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, query: str, context: str) -> str:
        import httpx

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {query}",
                        },
                    ],
                    # Near-greedy: this is an extraction task, and sampling
                    # variety here just means variety in how wrong it can be.
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()


def answer(query: str, hits: list[Hit], generator=None, max_chars: int = 6000) -> Answer:
    """Answer from retrieved chunks, generating prose only if a generator is given.

    Without a generator this returns the retrieved passages themselves. That is
    a legitimate mode, not a degraded one: for "where is this documented", the
    citations *are* the answer, and no model is needed to produce them.
    """
    citations = [hit.chunk.citation for hit in hits]

    if generator is None:
        return Answer(text="", citations=citations, hits=hits)

    return Answer(
        text=generator.generate(query, build_context(hits, max_chars)),
        citations=citations,
        hits=hits,
    )
