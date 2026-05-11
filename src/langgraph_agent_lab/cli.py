"""CLI for the lab."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        t0 = time.perf_counter()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        metrics.append(
            metric_from_state(
                final_state, scenario.expected_route.value, scenario.requires_approval, latency_ms
            )
        )
    resume_success = False
    if checkpointer is not None and scenarios:
        try:
            first_thread_cfg = {"configurable": {"thread_id": f"thread-{scenarios[0].id}"}}
            history = list(graph.get_state_history(first_thread_cfg))
            resume_success = len(history) > 0
        except Exception:
            resume_success = False
    report = summarize_metrics(metrics, resume_success=resume_success)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    # Bonus extension: export Mermaid diagram of the compiled graph.
    try:
        diagram = graph.get_graph().draw_mermaid()
        diagram_path = output.parent / "graph.mmd"
        diagram_path.parent.mkdir(parents=True, exist_ok=True)
        diagram_path.write_text(diagram, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort
        typer.echo(f"Mermaid export skipped: {exc}")
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("demo-persistence")
def demo_persistence(
    db_path: Annotated[str, typer.Option("--db")] = "outputs/checkpoints.sqlite",
    thread_id: Annotated[str, typer.Option("--thread-id")] = "demo-persist",
    query: Annotated[str, typer.Option("--query")] = "Please lookup order status for order 999",
) -> None:
    """Demo crash recovery with SqliteSaver.

    First call: runs a scenario, writes checkpoints to SQLite, prints history count.
    Second call (same --db and --thread-id): reloads state from SQLite WITHOUT rerun
    and prints the same history — proving checkpoints survive across processes.
    """
    from .state import Route, Scenario

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    checkpointer = build_checkpointer("sqlite", database_url=db_path)
    graph = build_graph(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": thread_id}}

    existing = list(graph.get_state_history(cfg))
    typer.echo(f"[pre] DB={db_path} thread_id={thread_id}")
    typer.echo(f"[pre] existing checkpoints: {len(existing)}")

    if not existing:
        typer.echo("[run] No prior state — running scenario fresh.")
        scenario = Scenario(id=thread_id, query=query, expected_route=Route.TOOL)
        state = initial_state(scenario)
        state["thread_id"] = thread_id  # align with cfg
        graph.invoke(state, config=cfg)
    else:
        typer.echo("[run] Found prior state in SQLite — skipping rerun.")
        typer.echo("      (This proves checkpoints survived across process restarts.)")

    history = list(graph.get_state_history(cfg))
    typer.echo(f"[post] checkpoints in DB: {len(history)}")
    if history:
        latest = history[0]
        typer.echo(f"[post] latest route: {latest.values.get('route')}")
        typer.echo(f"[post] latest final_answer: {latest.values.get('final_answer')}")
        typer.echo(f"[post] nodes visited: {len(latest.values.get('events') or [])}")
    typer.echo("\nTip: run this command a second time; you will see 'Found prior state'.")


@app.command("demo-time-travel")
def demo_time_travel(
    query: Annotated[str, typer.Option("--query")] = "Timeout failure while processing request",
    pick: Annotated[int, typer.Option("--pick", help="Checkpoint index (from newest=0) to replay from")] = 3,
) -> None:
    """Demo time-travel by replaying from a past checkpoint.

    1. Run an error scenario with MemorySaver so we collect many checkpoints in the retry loop.
    2. Pick a checkpoint mid-run and replay from it via graph.invoke(None, cfg_with_checkpoint_id).
    3. Print both timelines so you can compare original vs replayed execution.
    """
    from .state import Route, Scenario

    checkpointer = build_checkpointer("memory")
    graph = build_graph(checkpointer=checkpointer)
    thread_id = "demo-time-travel"
    cfg = {"configurable": {"thread_id": thread_id}}

    scenario = Scenario(id=thread_id, query=query, expected_route=Route.ERROR)
    state = initial_state(scenario)
    state["thread_id"] = thread_id
    original = graph.invoke(state, config=cfg)
    original_events = [e.get("node") for e in (original.get("events") or [])]
    typer.echo(f"[original] path: {' -> '.join(original_events)}")
    typer.echo(f"[original] final_answer: {original.get('final_answer')}")

    history = list(graph.get_state_history(cfg))
    typer.echo(f"\n[history] total checkpoints: {len(history)}")
    for i, snap in enumerate(history[:8]):
        typer.echo(f"  [{i}] next={snap.next} route={snap.values.get('route')} attempt={snap.values.get('attempt')}")

    if pick >= len(history):
        pick = len(history) - 1
    target = history[pick]
    typer.echo(f"\n[replay] resuming from checkpoint index {pick}: next={target.next}")
    replay_cfg = target.config
    replayed = graph.invoke(None, config=replay_cfg)
    replayed_events = [e.get("node") for e in (replayed.get("events") or [])]
    typer.echo(f"[replay] path: {' -> '.join(replayed_events)}")
    typer.echo(f"[replay] final_answer: {replayed.get('final_answer')}")


if __name__ == "__main__":
    app()
