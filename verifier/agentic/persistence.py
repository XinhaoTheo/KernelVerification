"""Persistence helpers for agentic verification runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import ArtifactRef, Claim, Evidence, RunState, ToolCall, ToolEvent, Turn


@dataclass(slots=True)
class PersistedRun:
    run_dir: Path
    run_json: Path
    tool_events_jsonl: Path
    claims_json: Path
    transcript_md: Path
    verdict_json: Path | None


def persist_run(state: RunState, run_dir: Path, *, stop_reason: str | None = None) -> PersistedRun:
    run_dir.mkdir(parents=True, exist_ok=True)

    run_json = run_dir / "run.json"
    tool_events_jsonl = run_dir / "tool_events.jsonl"
    claims_json = run_dir / "claims.json"
    transcript_md = run_dir / "transcript.md"
    verdict_json = run_dir / "verdict.json" if state.verdict is not None else None

    run_json.write_text(_json_dumps(state.to_dict()), encoding="utf-8")
    tool_events_jsonl.write_text(
        "".join(_json_dumps(event.to_dict(), indent=None) + "\n" for event in state.tool_events),
        encoding="utf-8",
    )
    claims_json.write_text(
        _json_dumps([claim.to_dict() for claim in state.claims]),
        encoding="utf-8",
    )
    transcript_md.write_text(_render_transcript(state, stop_reason=stop_reason), encoding="utf-8")
    if verdict_json is not None:
        verdict_json.write_text(_json_dumps(state.verdict), encoding="utf-8")

    return PersistedRun(
        run_dir=run_dir,
        run_json=run_json,
        tool_events_jsonl=tool_events_jsonl,
        claims_json=claims_json,
        transcript_md=transcript_md,
        verdict_json=verdict_json,
    )


def load_run_state(run_json: Path) -> RunState:
    """Load a persisted run.json into structured RunState objects."""
    raw = json.loads(Path(run_json).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("run.json must contain a JSON object")
    return RunState(
        entry=raw.get("entry"),
        artifact=raw.get("artifact"),
        skills=list(raw.get("skills") or []),
        history=[_turn_from_dict(item) for item in raw.get("history") or []],
        tool_events=[_tool_event_from_dict(item) for item in raw.get("tool_events") or []],
        claims=[_claim_from_dict(item) for item in raw.get("claims") or []],
        verdict=raw.get("verdict"),
        convergence=raw.get("convergence"),
        skeptic_review=raw.get("skeptic_review"),
    )


def load_tool_events_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _turn_from_dict(raw: dict[str, Any]) -> Turn:
    return Turn(
        role=raw.get("role", "unknown"),
        round=int(raw.get("round", 0)),
        text=str(raw.get("text", "")),
        tool_calls=[_tool_call_from_dict(item) for item in raw.get("tool_calls") or []],
    )


def _tool_call_from_dict(raw: dict[str, Any]) -> ToolCall:
    return ToolCall(tool=str(raw.get("tool", "")), args=dict(raw.get("args") or {}))


def _tool_event_from_dict(raw: dict[str, Any]) -> ToolEvent:
    return ToolEvent(
        id=str(raw.get("id", "")),
        tool=str(raw.get("tool", "")),
        args=dict(raw.get("args") or {}),
        status=str(raw.get("status", "")),
        output=dict(raw.get("output") or {}),
        created_at=str(raw.get("created_at", "")),
    )


def _claim_from_dict(raw: dict[str, Any]) -> Claim:
    return Claim(
        id=str(raw.get("id", "")),
        statement=str(raw.get("statement", "")),
        rationale=str(raw.get("rationale", "")),
        status=str(raw.get("status", "open")),
        raised_by=str(raw.get("raised_by", "unknown")),
        scope=str(raw.get("scope", "unknown")),
        scope_rationale=str(raw.get("scope_rationale", "")),
        scope_evidence=list(raw.get("scope_evidence") or []),
        evidence=[_evidence_from_dict(item) for item in raw.get("evidence") or []],
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
    )


def _evidence_from_dict(raw: dict[str, Any]) -> Evidence:
    return Evidence(
        id=str(raw.get("id", "")),
        kind=str(raw.get("kind", "agent_analysis")),
        tool_event_id=raw.get("tool_event_id"),
        summary=str(raw.get("summary", "")),
        supports=str(raw.get("supports", "inconclusive")),
        data=dict(raw.get("data") or {}),
        artifacts=[_artifact_ref_from_dict(item) for item in raw.get("artifacts") or []],
        created_at=str(raw.get("created_at", "")),
    )


def _artifact_ref_from_dict(raw: dict[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        kind=raw.get("kind", "other"),
        path=str(raw.get("path", "")),
        description=str(raw.get("description", "")),
        sha256=raw.get("sha256"),
    )


_MAX_TRANSCRIPT_TEXT = 1200
_MAX_TRANSCRIPT_JSON = 2000


def _render_transcript(state: RunState, *, stop_reason: str | None = None) -> str:
    lines: list[str] = []
    lines.append("# Agentic Verification Transcript")
    lines.append("")
    lines.append(f"- Entry: {_md_inline(state.entry or 'unknown')}")
    lines.append(f"- Turns: {len(state.history)}")
    lines.append(f"- Tool events: {len(state.tool_events)}")
    lines.append(f"- Claims: {len(state.claims)}")
    if stop_reason:
        lines.append(f"- Stop reason: {_md_inline(stop_reason)}")
    if state.verdict:
        lines.append(f"- Verdict: {_md_inline(str(state.verdict.get('verdict', 'unknown')))}")
        lines.append(f"- Confidence: {state.verdict.get('confidence', 'unknown')}")
    if state.convergence:
        lines.append(f"- Convergence request: {_md_inline(str(state.convergence.get('request', 'unknown')))}")
    if state.skeptic_review:
        lines.append(f"- Skeptic review: {_md_inline(str(state.skeptic_review.get('decision', 'unknown')))}")
    lines.append("")

    _append_timeline(lines, state)
    _append_claims(lines, state)
    _append_tool_events(lines, state)
    _append_verdict(lines, state)
    return "\n".join(lines).rstrip() + "\n"


def _append_timeline(lines: list[str], state: RunState) -> None:
    lines.append("## Timeline")
    lines.append("")
    event_index = 0
    for turn in state.history:
        lines.append(f"### Turn {turn.round} - {_md_inline(_value_text(turn.role))}")
        lines.append("")
        if turn.text.strip():
            lines.append("Message:")
            lines.append("")
            lines.append(_md_block(turn.text.strip(), limit=_MAX_TRANSCRIPT_TEXT))
            lines.append("")
        if not turn.tool_calls:
            lines.append("Tool calls: none")
            lines.append("")
            continue
        lines.append("Tool calls:")
        lines.append("")
        for call in turn.tool_calls:
            event = state.tool_events[event_index] if event_index < len(state.tool_events) else None
            event_index += 1
            label = f"{call.tool}"
            if event is not None:
                label += f" -> {event.id} {_value_text(event.status)}"
            lines.append(f"- `{label}`")
            if call.args:
                lines.append("  Args:")
                lines.extend(_indented_json(call.args, indent="  ", limit=800))
            if event is not None:
                summary = _tool_output_summary(event)
                if summary:
                    lines.append("  Output summary:")
                    lines.extend(_indented_json(summary, indent="  ", limit=1200))
        lines.append("")


def _append_claims(lines: list[str], state: RunState) -> None:
    lines.append("## Claims")
    lines.append("")
    if not state.claims:
        lines.append("No claims recorded.")
        lines.append("")
        return
    for claim in state.claims:
        lines.append(f"### {claim.id} - {_md_inline(_value_text(claim.status))}")
        lines.append("")
        lines.append(f"Statement: {claim.statement}")
        lines.append("")
        lines.append(f"Scope: `{_value_text(claim.scope)}`")
        if claim.scope_rationale:
            lines.append("")
            lines.append(f"Scope rationale: {claim.scope_rationale}")
        if claim.scope_evidence:
            lines.append("")
            lines.append("Scope evidence:")
            for item in claim.scope_evidence:
                source = item.get("source", "unknown") if isinstance(item, dict) else "unknown"
                summary = item.get("summary", "") if isinstance(item, dict) else str(item)
                lines.append(f"- `{source}`: {summary}")
        lines.append("")
        lines.append(f"Rationale: {claim.rationale}")
        lines.append("")
        if not claim.evidence:
            lines.append("Evidence: none")
            lines.append("")
            continue
        lines.append("Evidence:")
        for evidence in claim.evidence:
            event = f", tool_event_id={evidence.tool_event_id}" if evidence.tool_event_id else ""
            lines.append(
                f"- `{evidence.id}` {_value_text(evidence.kind)} supports `{_value_text(evidence.supports)}`{event}: {evidence.summary}"
            )
        lines.append("")


def _append_tool_events(lines: list[str], state: RunState) -> None:
    lines.append("## Tool Events")
    lines.append("")
    if not state.tool_events:
        lines.append("No tool events recorded.")
        lines.append("")
        return
    for event in state.tool_events:
        lines.append(f"### {event.id} - {event.tool} - {_value_text(event.status)}")
        lines.append("")
        summary = _tool_output_summary(event)
        if summary:
            lines.extend(_indented_json(summary, indent="", limit=_MAX_TRANSCRIPT_JSON))
            lines.append("")


def _append_verdict(lines: list[str], state: RunState) -> None:
    lines.append("## Verdict")
    lines.append("")
    if not state.verdict:
        lines.append("No final verdict recorded.")
        lines.append("")
        return
    lines.extend(_indented_json(state.verdict, indent="", limit=_MAX_TRANSCRIPT_JSON))
    lines.append("")


def _tool_output_summary(event: ToolEvent) -> dict[str, Any]:
    output = event.output or {}
    if _value_text(event.status) == "error":
        return {"error_type": output.get("error_type"), "message": output.get("message")}

    summary: dict[str, Any] = {}
    for key in (
        "entry",
        "claim_id",
        "claim_statement",
        "expected_signal",
        "event_id",
        "exit_code",
        "timed_out",
        "timeout_s",
        "duration_s",
        "json_result",
        "json_parse_error",
        "request",
        "verdict",
        "confidence",
        "decisive_claims",
        "reason",
    ):
        if key in output and output[key] is not None:
            summary[key] = output[key]

    if event.tool == "record_claim":
        summary = {
            key: output.get(key)
            for key in ("id", "status", "scope", "scope_rationale", "scope_evidence", "statement", "rationale")
            if key in output
        }
    elif event.tool in {"append_evidence", "update_claim_status"}:
        summary = {key: output.get(key) for key in ("id", "status", "summary", "supports") if key in output}
    elif event.tool == "record_no_new_claims":
        summary = {
            key: output.get(key)
            for key in ("decision", "reason", "reviewed_claims", "reviewed_tool_event_count", "turn")
            if key in output
        }
    elif event.tool == "finalize_probe_evidence":
        claim = output.get("claim") if isinstance(output.get("claim"), dict) else {}
        evidence = output.get("evidence") if isinstance(output.get("evidence"), dict) else {}
        summary = {
            "claim_id": claim.get("id"),
            "claim_status": claim.get("status"),
            "evidence_id": evidence.get("id"),
            "supports": evidence.get("supports"),
            "summary": evidence.get("summary"),
            "tool_event_id": evidence.get("tool_event_id"),
        }
    elif event.tool in {"run_python_probe", "run_claim_probe"}:
        summary.setdefault("stdout", _truncate(str(output.get("stdout", "")), 500))
        summary.setdefault("stderr", _truncate(str(output.get("stderr", "")), 500))
        if "artifacts" in output:
            summary["artifacts"] = output.get("artifacts")
        if "evidence_draft" in output:
            summary["evidence_draft"] = output.get("evidence_draft")
    elif not summary:
        summary = _compact_mapping(output)
    return {key: value for key, value in summary.items() if value is not None}


def _compact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, str):
            compact[key] = _truncate(item, 500)
        elif isinstance(item, (int, float, bool)) or item is None:
            compact[key] = item
        elif key in {"artifacts", "files"}:
            compact[key] = item
    return compact


def _indented_json(value: Any, *, indent: str, limit: int) -> list[str]:
    rendered = _truncate(_json_dumps(value), limit)
    return [indent + line for line in rendered.splitlines()]


def _value_text(value: Any) -> str:
    return str(value.value) if hasattr(value, "value") else str(value)


def _md_block(value: str, *, limit: int) -> str:
    return "```text\n" + _truncate(value, limit) + "\n```"


def _md_inline(value: str) -> str:
    return "`" + value.replace("`", "'") + "`"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def _json_dumps(value, *, indent: int | None = 2) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, default=str)
