"""Model migration: re-embedding, burn-in trigger, version tracking."""

from __future__ import annotations

from typing import Optional

from rich.console import Console

from chicory.config import ChicoryConfig
from chicory.orchestrator.orchestrator import Orchestrator

console = Console()


def run_migration(
    config: ChicoryConfig,
    new_llm_model: str | None = None,
    new_embedding_model: str | None = None,
) -> None:
    """Run a full model migration."""
    o = Orchestrator(config)

    try:
        # 1. Record new model version
        llm_model = new_llm_model or config.llm_model
        emb_model = new_embedding_model or config.embedding_model

        console.print(f"[bold]Migrating to LLM: {llm_model}, Embedding: {emb_model}[/bold]")

        o.db.execute(
            "INSERT INTO model_versions (model_name, embedding_model, notes) VALUES (?, ?, ?)",
            (llm_model, emb_model, "Migration"),
        )
        o.db.connection.commit()

        # 2. Re-embed all memories if embedding model changed
        if new_embedding_model:
            console.print("[bold]Re-embedding all memories...[/bold]")
            count = o._embedding_engine.reembed_all(new_model=new_embedding_model)
            console.print(f"  Re-embedded {count} memories.")
        else:
            console.print("[dim]Embedding model unchanged, skipping re-embedding.[/dim]")

        # 3. Enter burn-in mode on adaptive thresholds
        console.print("[bold]Entering burn-in mode for adaptive thresholds...[/bold]")
        o._adaptive_thresholds.enter_burn_in(llm_model)
        console.print(
            f"  Burn-in active for {config.burn_in_hours} hours. "
            f"Thresholds widened by {config.burn_in_threshold_multiplier}x."
        )

        # 4. Snapshot current trends (for baseline comparison)
        console.print("[bold]Snapshotting current trends...[/bold]")
        o.trend_engine.snapshot_trends()

        console.print("[green bold]Migration complete.[/green bold]")

    finally:
        o.close()
