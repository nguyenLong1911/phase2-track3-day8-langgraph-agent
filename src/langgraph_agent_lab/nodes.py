"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

from .state import AgentState, ApprovalDecision, Route, make_event


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    Implemented: trim whitespace, log truncated query, emit audit event.
    """
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using prioritized keyword heuristics.

    Priority order (per README): risky > tool > missing_info > error > simple.
    Word-boundary matching is used so e.g. "it" does not match "item"/"iteration".
    """
    raw = state.get("query", "")
    query = raw.lower()
    # Word-boundary tokens (strip punctuation) for precise matching
    clean_words = [w.strip("?!.,;:'\"()[]") for w in query.split()]
    token_set = {w for w in clean_words if w}

    risky_kw = {
        "refund", "refunds", "delete", "remove", "cancel", "revoke",
        "send", "wire", "deactivate", "terminate", "purge", "drop",
    }
    tool_kw = {
        "status", "order", "lookup", "check", "track",
        "find", "search", "fetch", "query", "retrieve", "get",
    }
    error_kw = {
        "timeout", "timed", "fail", "failed", "failure",
        "error", "errors", "crash", "crashed", "unavailable", "unresponsive",
    }
    vague_pronouns = {"it", "this", "that", "thing", "stuff"}

    route = Route.SIMPLE
    risk_level = "low"
    if token_set & risky_kw:
        route = Route.RISKY
        risk_level = "high"
    elif token_set & tool_kw:
        route = Route.TOOL
    elif len(clean_words) < 5 and (token_set & vague_pronouns):
        route = Route.MISSING_INFO
    elif token_set & error_kw:
        route = Route.ERROR
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"route={route.value}")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Implemented: return a fixed clarification surfacing the missing-context need.
    """
    question = "Can you provide the order id or the missing context?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    Implemented: deterministic mock that simulates transient failure when route=='error' and attempt<2, then succeeds (drives the bounded retry loop).
    """
    attempt = int(state.get("attempt", 0))
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={state.get('scenario_id', 'unknown')}"
    else:
        result = f"mock-tool-result for scenario={state.get('scenario_id', 'unknown')}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval.

    Implemented: produce a proposed_action string and emit pending_approval event.
    """
    return {
        "proposed_action": "prepare refund or external action; approval required",
        "events": [make_event("risky_action", "pending_approval", "approval required")],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.

    Implemented: real HITL via interrupt() when LANGGRAPH_INTERRUPT=true; mock approval otherwise so CI runs offline.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision.

    Implemented: increment attempt counter and append an error message; bound is enforced in routing.route_after_retry.
    """
    attempt = int(state.get("attempt", 0)) + 1
    errors = [f"transient failure attempt={attempt}"]
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [make_event("retry", "completed", "retry attempt recorded", attempt=attempt)],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response.

    Implemented: ground the answer in the latest tool_results entry, fallback to a safe default.
    """
    if state.get("tool_results"):
        answer = f"I found: {state['tool_results'][-1]}"
    else:
        answer = "This is a safe mock answer. Replace with your agent response."
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.

    Implemented: substring check for 'ERROR' in latest tool result -> evaluation_result='needs_retry' else 'success'.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    if "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "tool result indicates failure, retry needed")],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    Implemented: set a final_answer indicating the run was logged for manual review and emit an audit event.
    """
    return {
        "final_answer": "Request could not be completed after maximum retry attempts. Logged for manual review.",
        "events": [make_event("dead_letter", "completed", f"max retries exceeded, attempt={state.get('attempt', 0)}")],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
