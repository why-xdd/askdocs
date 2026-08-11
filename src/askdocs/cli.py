"""Command line interface: index, ask, eval."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .answer import OllamaGenerator, answer
from .embeddings import resolve
from .evaluate import compare, load_questions
from .index import Index, build

app = typer.Typer(
    add_completion=False,
    help="Ask questions about your own documents. Local, hybrid retrieval, cited answers.",
)
console = Console()

DEFAULT_INDEX = Path(".askdocs/index.db")


def _version(value: bool) -> None:
    if value:
        console.print(f"askdocs {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show version."
    ),
) -> None:
    pass


@app.command()
def index(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or directories."),
    out: Path = typer.Option(DEFAULT_INDEX, "--out", "-o", help="Where to write it."),
    embedder: str = typer.Option(
        "auto", help="auto | tfidf | ollama | sentence-transformers"
    ),
    max_chars: int = typer.Option(1200, help="Largest chunk, in characters."),
    overlap: int = typer.Option(120, help="Characters repeated across a boundary."),
) -> None:
    """Chunk, embed and store a corpus."""
    backend = resolve(embedder)

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True
    ) as progress:
        task = progress.add_task("indexing...")
        built = build(
            paths,
            backend,
            max_chars,
            overlap,
            progress=lambda p: progress.update(task, description=f"reading {p.name}"),
        )

    if not built.chunks:
        console.print("[yellow]No indexable documents found.[/]")
        raise typer.Exit(code=1)

    built.save(out)
    stats = built.stats

    # ASCII rather than an arrow glyph: the Windows legacy console is cp1251 by
    # default and raises UnicodeEncodeError on anything outside it, which turns
    # a successful index into a crash after the work is already done.
    console.print(
        f"[green]Indexed {stats.chunks} chunks from {stats.documents} documents[/]\n"
        f"[dim]embedder: {stats.embedder} ({stats.dimensions}d)  ->  {out}[/]"
    )
    if stats.embedder.startswith("tfidf"):
        console.print(
            "[dim]Using the built-in TF-IDF backend. Run Ollama with "
            "nomic-embed-text for better paraphrase matching.[/]"
        )


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question."),
    index_path: Path = typer.Option(DEFAULT_INDEX, "--index", "-i"),
    top_k: int = typer.Option(5, "-k", help="Passages to retrieve."),
    mode: str = typer.Option("hybrid", help="hybrid | lexical | dense"),
    generate: bool = typer.Option(
        False, "--generate/--no-generate", help="Write an answer with a local LLM."
    ),
    model: str = typer.Option("qwen2.5:7b", help="Ollama model for --generate."),
    show_ranks: bool = typer.Option(False, help="Show each retriever's rank."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Retrieve passages, and optionally have a local model answer from them."""
    loaded = Index.load(index_path)
    hits = loaded.search(query, top_k=top_k, mode=mode)

    if not hits:
        console.print("[yellow]Nothing matched.[/]")
        raise typer.Exit()

    generator = OllamaGenerator(model=model) if generate else None
    result = answer(query, hits, generator)

    if json_out:
        console.print_json(json.dumps(result.as_dict(), ensure_ascii=False))
        raise typer.Exit()

    if result.text:
        console.print(Panel(Markdown(result.text), title="answer", border_style="green"))
        console.print()

    for position, hit in enumerate(hits, start=1):
        rank_note = ""
        if show_ranks:
            lexical = hit.lexical_rank or "-"
            dense = hit.dense_rank or "-"
            rank_note = f"  [dim]bm25 #{lexical} · vector #{dense}[/]"

        console.print(f"[cyan]\\[{position}][/] [bold]{hit.chunk.citation}[/]{rank_note}")
        snippet = hit.chunk.text.strip().replace("\n", " ")
        console.print(f"    [dim]{snippet[:280]}{'…' if len(snippet) > 280 else ''}[/]\n")


@app.command(name="eval")
def eval_cmd(
    questions: Path = typer.Argument(..., exists=True, help="JSON question set."),
    index_path: Path = typer.Option(DEFAULT_INDEX, "--index", "-i"),
    top_k: int = typer.Option(5, "-k"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    json_out: Path = typer.Option(None, "--json", help="Write results here."),
) -> None:
    """Compare lexical, dense and hybrid retrieval on your own questions."""
    loaded = Index.load(index_path)
    question_set = load_questions(questions)
    results = compare(loaded, question_set, top_k=top_k, rerank=rerank)

    table = Table(
        title=f"{len(question_set)} questions · top-{top_k} · {loaded.stats.embedder}",
        title_justify="left",
    )
    table.add_column("mode")
    table.add_column(f"recall@{top_k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("misses", justify="right")

    best = max(results, key=lambda r: (r.recall_at_k, r.mrr))
    for result in results:
        style = "bold green" if result is best else None
        table.add_row(
            result.mode,
            f"{result.recall_at_k:.0%}",
            f"{result.mrr:.3f}",
            str(len(result.misses)),
            style=style,
        )

    console.print(table)

    if best.mode != "hybrid":
        console.print(
            f"\n[yellow]On this corpus {best.mode} retrieval wins. "
            f"Use --mode {best.mode}.[/]"
        )

    hybrid = next(r for r in results if r.mode == "hybrid")
    if hybrid.misses:
        console.print("\n[dim]Questions hybrid retrieval missed:[/]")
        for miss in hybrid.misses[:10]:
            console.print(f"  [dim]· {miss}[/]")

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps([r.as_dict() for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"\n[dim]Written to {json_out}[/]")


if __name__ == "__main__":
    app()
