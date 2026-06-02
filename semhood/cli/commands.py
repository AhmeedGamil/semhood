"""
CLI commands for semhood — a thin client over the warm daemon.

Every command resolves the current project root and calls the daemon (auto-
starting it on first use). Nothing here loads the embedding model: that lives in
the daemon, loaded once and kept warm, so repeated searches are instant.

Usage:
  semhood index ./my-project              # Stage 1: structural index (fast, free)
  semhood index ./my-project --changed    # Incremental (git diff HEAD~1)
  semhood enrich                          # Stage 2: drain pending → LLM
  semhood query "How does auth work?"     # Full RAG answer
  semhood search "retry backoff"          # Pure retrieval, no LLM
  semhood status                          # Counts per enrichment_state
  semhood serve                           # Run the daemon in the foreground
  semhood stop                            # Stop a background daemon
  semhood projects                        # List indexed projects
  semhood doctor                          # Daemon + config health check
"""

from __future__ import annotations

import json as _json
import logging

import typer
from rich.console import Console
from rich.table import Table

from semhood import client

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
console = Console()
app = typer.Typer(name="semhood", help="AST-based semantic code search that knows the neighborhood — code RAG for humans and AI")


def _root(root: str | None, start: str | None = None) -> str:
    """Resolve the project root a command should act on."""
    return root or client.current_root(start)


def _fail(exc: Exception) -> None:
    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Indexing / enrichment
# ---------------------------------------------------------------------------


@app.command()
def index(
    path: str = typer.Argument(".", help="Path to project directory"),
    root: str = typer.Option(None, "--root", "-r", help="Override the project root key"),
    changed: bool = typer.Option(False, "--changed", help="Index only git-changed files"),
    reset: bool = typer.Option(
        False, "--reset",
        help="Rebuild the index from scratch (use after changing the embedding "
             "model). Enrichment is restored from .semhood/enrichment.jsonl — no LLM cost.",
    ),
):
    """Stage 1: structural indexing. Fast, free, no LLM calls."""
    import os

    abspath = os.path.abspath(path)
    proj = _root(root, abspath)
    try:
        with console.status("[bold green]Indexing (daemon)..."):
            result = client.request(
                "index",
                {"root": proj, "path": abspath, "changed_only": changed, "reset": reset},
            )
    except client.DaemonError as exc:
        _fail(exc)

    table = Table(title="Indexing Results (Stage 1: structural)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Files indexed", str(result.get("files_indexed", 0)))
    table.add_row("Chunks indexed", str(result.get("chunks_indexed", 0)))
    table.add_row("Chunks skipped", str(result.get("chunks_skipped", 0)))
    table.add_row("Stale callers cleaned", str(result.get("stale_callers_cleaned", 0)))
    console.print(table)
    console.print("\n[dim]Run [bold]semhood enrich[/bold] to add LLM enrichment.[/dim]")


@app.command()
def enrich(
    root: str = typer.Option(None, "--root", "-r"),
    force: bool = typer.Option(False, "--force", help="Re-enrich every chunk"),
):
    """Stage 2: drain the pending queue with LLM-generated summaries + queries."""
    proj = _root(root)
    try:
        with console.status("[bold green]Enriching (daemon)..."):
            result = client.request("enrich", {"root": proj, "force": force})
    except client.DaemonError as exc:
        _fail(exc)

    table = Table(title="Enrichment Results (Stage 2)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Total processed", str(result.get("total", 0)))
    table.add_row("Enriched (fresh)", str(result.get("enriched", 0)))
    table.add_row("Skipped (trivial)", str(result.get("skipped_trivial", 0)))
    table.add_row("Failed", str(result.get("failed", 0)))
    console.print(table)


@app.command()
def compact(root: str = typer.Option(None, "--root", "-r")):
    """Prune .semhood/enrichment.jsonl records no longer referenced by any chunk.

    Orphans (from changed/deleted code) are kept by default as a cross-version
    cache; run this when you want to shrink the committed file.
    """
    proj = _root(root)
    try:
        with console.status("[bold green]Compacting enrichment store..."):
            result = client.request("compact_enrichment", {"root": proj})
    except client.DaemonError as exc:
        _fail(exc)
    console.print(
        f"[green]Removed {result.get('removed', 0)} orphaned record(s).[/green] "
        f"{result.get('kept', 0)} kept."
    )


# ---------------------------------------------------------------------------
# Query / search
# ---------------------------------------------------------------------------


@app.command()
def query(
    question: str = typer.Argument(help="Your question about the codebase"),
    root: str = typer.Option(None, "--root", "-r"),
):
    """Full RAG answer — retrieve, optionally rerank, and generate."""
    proj = _root(root)
    try:
        with console.status("[bold green]Searching..."):
            result = client.request("query", {"root": proj, "question": question})
    except client.DaemonError as exc:
        _fail(exc)

    console.print("\n[bold cyan]Answer:[/bold cyan]")
    console.print(result.get("answer", ""))

    sources = result.get("sources") or []
    if sources:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        for source in sources:
            name = source.get("name", "")
            file = source.get("file", "")
            line = source.get("line", "")
            cls = source.get("class")
            label = f"{cls}.{name}" if cls else name
            console.print(f"  • {label} — {file}:{line}")


@app.command()
def search(
    question: str = typer.Argument(help="Search query"),
    root: str = typer.Option(None, "--root", "-r"),
    top_k: int = typer.Option(8, "--top-k", "-k", help="Results to return"),
    vectors: str = typer.Option(
        "code,description,developer_queries", "--vectors",
        help="Comma-separated vector names to search",
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output: table | json | paths | compact"
    ),
):
    """Pure retrieval — embed the query and search. No LLM required."""
    proj = _root(root)
    vector_names = [v.strip() for v in vectors.split(",") if v.strip()]
    fmt = (format or "table").lower()
    if fmt not in {"table", "json", "paths", "compact"}:
        typer.echo(f"Unknown --format {fmt!r}. Use: table, json, paths, compact", err=True)
        raise typer.Exit(2)

    try:
        out = client.request(
            "search", {"root": proj, "query": question, "top_k": top_k,
                       "vectors": vector_names}
        )
    except client.DaemonError as exc:
        _fail(exc)

    results = out.get("results", [])

    if fmt == "json":
        print(_json.dumps(out, indent=2, default=str))
        return
    if fmt == "paths":
        for r in results:
            sym = f"{r['class_name']}.{r['name']}" if r.get("class_name") else (r.get("name") or "")
            print(f"{r.get('file')}:{r.get('line_start')}\t{sym}")
        return
    if fmt == "compact":
        for i, r in enumerate(results, start=1):
            sym = f"{r['class_name']}.{r['name']}" if r.get("class_name") else (r.get("name") or "")
            loc = f"{r.get('file')}:{r.get('line_start')}"
            doc = (r.get("summary") or "").replace("\n", " ")
            doc = doc[:77] + "..." if len(doc) > 80 else doc
            print(f"{i:2d}. {r.get('score', 0):.3f}  {loc:<40}  {sym:<35}  {doc}")
        return

    # Default: rich table
    console.print(
        f"\n[bold cyan]Query:[/bold cyan] {question}  "
        f"[dim]dim={out.get('embedding_dim')}, vectors={vector_names}, top_k={top_k}[/dim]\n"
    )
    table = Table(title="Results (RRF across vectors)", title_style="bold green")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Symbol")
    table.add_column("File:Line", style="cyan")
    table.add_column("Doc/Summary", style="dim", overflow="ellipsis")
    if not results:
        table.add_row("-", "-", "[dim]no results[/dim]", "-", "-")
    for i, r in enumerate(results, start=1):
        sym = f"{r['class_name']}.{r['name']}" if r.get("class_name") else (r.get("name") or "?")
        loc = f"{r.get('file', '?')}:{r.get('line_start', '?')}"
        doc = (r.get("summary") or "").replace("\n", " ")[:80]
        table.add_row(str(i), f"{r.get('score', 0):.3f}", sym, loc, doc)
    console.print(table)


# ---------------------------------------------------------------------------
# Status / projects / daemon lifecycle
# ---------------------------------------------------------------------------


@app.command()
def status(root: str = typer.Option(None, "--root", "-r")):
    """Counts per enrichment_state for the current project's index."""
    proj = _root(root)
    try:
        out = client.request("index_status", {"root": proj})
    except client.DaemonError as exc:
        _fail(exc)

    counts = out.get("counts", {})
    total = out.get("total", 0)
    table = Table(title=f"semhood Index Status — {proj}")
    table.add_column("State", style="cyan")
    table.add_column("Count", style="bold", justify="right")
    table.add_column("Percent", justify="right")
    for state, n in counts.items():
        if n < 0:
            continue
        pct = (n / total * 100) if total else 0
        table.add_row(state, str(n), f"{pct:.1f}%")
    table.add_row("[bold]total[/bold]", str(total), "100.0%")
    console.print(table)


@app.command()
def projects():
    """List every project that has an index in ~/.semhood/indexes/."""
    try:
        out = client.request("projects")
    except client.DaemonError as exc:
        _fail(exc)
    items = out.get("projects", [])
    if not items:
        console.print("[dim]No projects indexed yet.[/dim]")
        return
    table = Table(title="Indexed projects")
    table.add_column("ID", style="dim")
    table.add_column("Root", style="cyan")
    for p in items:
        table.add_row(p.get("id", ""), p.get("root", "") or "[dim]?[/dim]")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option(client.DEFAULT_HOST, "--host"),
    port: int = typer.Option(client.DEFAULT_PORT, "--port"),
):
    """Run the daemon in the foreground (Ctrl-C to stop)."""
    from semhood.daemon import run

    console.print(f"[bold green]Starting semhood daemon[/bold green] on {host}:{port} "
                  f"[dim](loads the embedding model once)…[/dim]")
    run(host=host, port=port)


@app.command()
def stop():
    """Stop a background daemon."""
    if not client.is_running():
        console.print("[dim]No daemon running.[/dim]")
        return
    try:
        client.request("shutdown", {}, autostart=False)
    except client.DaemonError:
        pass  # connection drops as it exits — expected
    console.print("[green]Daemon stopped.[/green]")


@app.command("install-skill")
def install_skill(
    target: str = typer.Option(
        None, "--target", "-t",
        help="Comma list: claude,cursor,windsurf,kiro,codex — or 'all'. "
             "Default: auto-detect installed editors.",
    ),
):
    """Install semhood's usage guidance into your AI editors (user-level, once).

    Teaches the agent *when* to prefer semantic search over grep/read. Writes to
    each editor's own convention (Claude skill, Cursor/Windsurf rules, Kiro
    steering, Codex AGENTS.md). Re-run any time to update in place.
    """
    from semhood.skill_install import TARGET_KEYS
    from semhood.skill_install import install_skill as _install

    selected: list[str] | None = None
    if target:
        selected = TARGET_KEYS if target.strip() == "all" else [
            t.strip() for t in target.split(",") if t.strip()
        ]
        unknown = [t for t in selected if t not in TARGET_KEYS]
        if unknown:
            typer.echo(f"Unknown target(s): {', '.join(unknown)}. "
                       f"Choose from: {', '.join(TARGET_KEYS)}, all", err=True)
            raise typer.Exit(2)

    results = _install(selected)
    installed = [r for r in results if r["status"] == "installed"]

    table = Table(title="Install semhood skill")
    table.add_column("Editor", style="cyan")
    table.add_column("Status")
    table.add_column("Location", style="dim")
    for r in results:
        if r["status"] == "installed":
            table.add_row(r["label"], "[green]installed[/green]", r.get("path", ""))
        elif r["status"] == "skipped":
            table.add_row(r["label"], "[dim]skipped[/dim]", f"[dim]{r.get('reason', '')}[/dim]")
        else:
            table.add_row(r["label"], "[red]error[/red]", r.get("error", ""))
    console.print(table)

    if not installed:
        console.print(
            "\n[yellow]Nothing installed.[/yellow] No editor configs detected. "
            "Force one with [bold]--target claude[/bold] (or cursor/windsurf/kiro/codex/all)."
        )
    else:
        console.print(
            f"\n[green]Installed to {len(installed)} editor(s).[/green] "
            "Restart the editor to pick it up. The MCP server provides the tools; "
            "this just teaches the agent when to use them."
        )


@app.command()
def doctor():
    """Show daemon + config health."""
    from semhood.paths import global_config_path, semhood_home

    console.print(f"[bold]semhood home:[/bold] {semhood_home()}")
    console.print(f"[bold]Global config:[/bold] {global_config_path()}")
    out = _safe_health()
    if out:
        console.print(
            f"[green]Daemon: running[/green] — embedder="
            f"{out.get('embedder')} dim={out.get('embedding_dim')} "
            f"llm={out.get('llm')} projects_loaded={out.get('projects_loaded')}"
        )
    else:
        console.print("[yellow]Daemon: not running[/yellow] (starts on first command)")


def _safe_health() -> dict | None:
    info = client._read_info()
    if not info:
        return None
    try:
        return client._http(client._base_url(info) + "/health", None, 2.0)
    except Exception:
        return None


if __name__ == "__main__":
    app()
