"""Streamlit UI for demoing the LangGraph agent lab.

Run with:
    streamlit run src/langgraph_agent_lab/ui.py

Features:
- Pick a built-in scenario or type a custom query
- Live execution through the compiled graph (MemorySaver checkpointer)
- Real HITL path: when "Enable real HITL" is on, approval_node pauses via
  interrupt(); the UI shows an Approve/Reject form and resumes with Command().
- Route classification, audit-event timeline, final answer, retry/interrupt counts
- Mermaid diagram of the compiled graph
- Per-scenario state history (time-travel evidence)
"""
# ruff: noqa: E501
from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import metric_from_state
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state

st.set_page_config(page_title="Day 08 LangGraph Agent Demo", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph(checkpointer=build_checkpointer("memory"))


@st.cache_data
def get_scenarios() -> list[Scenario]:
    return load_scenarios("data/sample/scenarios.jsonl")


def render_event_timeline(events: list[dict]) -> None:
    if not events:
        st.info("No events recorded.")
        return
    for i, ev in enumerate(events, 1):
        node = ev.get("node", "?")
        etype = ev.get("event_type", "?")
        msg = ev.get("message", "")
        meta = ev.get("metadata") or {}
        with st.container(border=True):
            st.markdown(f"**{i}. `{node}`** · _{etype}_")
            st.caption(msg)
            if meta:
                st.json(meta, expanded=False)


def main() -> None:
    st.title("Day 08 — LangGraph Agentic Orchestration Demo")
    st.caption(
        "Interactive UI for the support-ticket agent. Pick a scenario or write a custom query; "
        "the graph routes it through classify → tool/clarify/risky → evaluate → answer/retry/dead_letter → finalize."
    )

    graph = get_graph()
    scenarios = get_scenarios()

    with st.sidebar:
        st.header("Run configuration")
        mode = st.radio("Input mode", ["Built-in scenario", "Custom query"], index=0)
        if mode == "Built-in scenario":
            options = {f"{s.id} — {s.expected_route.value}": s for s in scenarios}
            label = st.selectbox("Scenario", list(options.keys()))
            chosen = options[label]
            query = chosen.query
            expected_route = chosen.expected_route
            requires_approval = chosen.requires_approval
            max_attempts = chosen.max_attempts
            scenario_id = chosen.id
        else:
            query = st.text_area("Query", value="Refund this customer and send confirmation email", height=100)
            expected_route = Route(st.selectbox(
                "Expected route (for grading display)",
                [r.value for r in [Route.SIMPLE, Route.TOOL, Route.MISSING_INFO, Route.RISKY, Route.ERROR]],
                index=3,
            ))
            requires_approval = st.checkbox("Requires approval (HITL)", value=True)
            max_attempts = st.number_input("max_attempts", min_value=1, max_value=10, value=3)
            scenario_id = st.text_input("Scenario id", value="custom-1")

        st.markdown("---")
        st.subheader("HITL mode")
        real_hitl = st.checkbox(
            "Enable real `interrupt()` HITL",
            value=False,
            help="When ON: approval_node pauses via interrupt(); you must approve/reject below to resume. When OFF: mock approval always returns True.",
        )
        st.markdown("---")
        st.subheader("Architecture")
        st.markdown(
            "- 11 nodes, conditional retry loop bounded by `max_attempts`.\n"
            "- `evaluate` is the *done?* check that drives retries.\n"
            "- `approval` supports real `interrupt()` HITL when enabled."
        )

        run_btn = st.button("Run scenario", type="primary", use_container_width=True)
        if st.button("Reset metrics & session", use_container_width=True):
            st.session_state.pop("last_run", None)
            get_graph.clear()  # drop cached graph so MemorySaver is fresh
            st.rerun()

    if run_btn:
        # Toggle the env var read by approval_node on each invoke
        os.environ["LANGGRAPH_INTERRUPT"] = "true" if real_hitl else "false"
        scenario = Scenario(
            id=scenario_id,
            query=query,
            expected_route=expected_route,
            requires_approval=requires_approval,
            max_attempts=int(max_attempts),
        )
        state = initial_state(scenario)
        run_cfg = {"configurable": {"thread_id": state["thread_id"]}}
        t0 = time.perf_counter()
        final_state = graph.invoke(state, config=run_cfg)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        st.session_state["last_run"] = {
            "scenario": scenario,
            "final_state": final_state,
            "latency_ms": latency_ms,
            "run_cfg": run_cfg,
            "real_hitl": real_hitl,
        }

    last = st.session_state.get("last_run")
    if not last:
        st.info("Configure a run in the sidebar and press **Run scenario**.")
        _render_diagram(graph)
        return

    # HITL: detect a pending interrupt and render an approval form before metrics.
    _maybe_render_hitl(graph, last)

    scenario: Scenario = last["scenario"]
    final_state = last["final_state"]
    latency_ms = last["latency_ms"]
    run_cfg = last["run_cfg"]

    metric = metric_from_state(
        final_state,
        expected_route=scenario.expected_route.value,
        approval_required=scenario.requires_approval,
        latency_ms=latency_ms,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Expected", scenario.expected_route.value)
    col2.metric("Actual", metric.actual_route or "—")
    col3.metric("Success", "✅" if metric.success else "❌")
    col4.metric("Retries", metric.retry_count)
    col5.metric("Interrupts", metric.interrupt_count)

    st.subheader("Final answer")
    st.success(final_state.get("final_answer") or final_state.get("pending_question") or "(no answer)")

    tab_events, tab_state, tab_history, tab_diag = st.tabs(
        ["Event timeline", "Final state", "Checkpoint history", "Graph diagram"]
    )

    with tab_events:
        st.write(f"**Nodes visited:** {metric.nodes_visited} · **Latency:** {latency_ms} ms")
        render_event_timeline(final_state.get("events", []))

    with tab_state:
        # Strip very large/redundant fields for readability
        view = {k: v for k, v in final_state.items() if k != "events"}
        st.json(view, expanded=True)

    with tab_history:
        try:
            history = list(graph.get_state_history(run_cfg))
            st.write(f"Checkpoints captured: **{len(history)}** (proves resume / time-travel).")
            for i, snap in enumerate(history[:10]):
                with st.expander(f"Checkpoint {i} — next: {snap.next}", expanded=False):
                    st.write({"values": dict(snap.values), "next": snap.next})
        except Exception as exc:  # pragma: no cover - best-effort UI
            st.warning(f"State history unavailable: {exc}")

    with tab_diag:
        _render_diagram(graph)


def _get_pending_interrupt(graph, run_cfg):
    """Return the first pending Interrupt payload if graph is paused, else None."""
    try:
        snapshot = graph.get_state(run_cfg)
    except Exception:
        return None, ()
    if not snapshot.next:
        return None, ()
    # tasks may carry interrupts; find the first interrupt payload
    for task in getattr(snapshot, "tasks", []) or []:
        interrupts = getattr(task, "interrupts", None) or []
        for intr in interrupts:
            return intr, snapshot.next
    return None, snapshot.next


def _maybe_render_hitl(graph, last: dict) -> None:
    run_cfg = last["run_cfg"]
    intr, pending = _get_pending_interrupt(graph, run_cfg)
    if intr is None:
        return
    st.warning(f"🟡 Graph paused at node(s) **{pending}** — human approval required.")
    payload = getattr(intr, "value", intr)
    with st.container(border=True):
        st.markdown("**Proposed action awaiting review:**")
        st.json(payload if isinstance(payload, dict) else {"payload": str(payload)})
        with st.form("hitl-form"):
            decision = st.radio("Decision", ["Approve", "Reject"], horizontal=True)
            comment = st.text_input("Reviewer comment", value="looks fine")
            submitted = st.form_submit_button("Submit decision & resume", type="primary")
        if submitted:
            resume_value = {
                "approved": decision == "Approve",
                "reviewer": "streamlit-user",
                "comment": comment,
            }
            t0 = time.perf_counter()
            new_state = graph.invoke(Command(resume=resume_value), config=run_cfg)
            extra_latency = int((time.perf_counter() - t0) * 1000)
            last["final_state"] = new_state
            last["latency_ms"] = last["latency_ms"] + extra_latency
            st.session_state["last_run"] = last
            st.success(f"Resumed with decision **{decision}** — graph continued.")
            st.rerun()


def _render_diagram(graph) -> None:
    st.subheader("Compiled graph (Mermaid)")
    try:
        diagram = graph.get_graph().draw_mermaid()
    except Exception as exc:  # pragma: no cover
        st.warning(f"Could not render diagram: {exc}")
        return
    # Streamlit doesn't render mermaid natively; show source + offer download.
    st.code(diagram, language="mermaid")
    out = Path("outputs/graph.mmd")
    if out.exists():
        st.download_button("Download outputs/graph.mmd", out.read_bytes(), file_name="graph.mmd")


if __name__ == "__main__":
    main()
