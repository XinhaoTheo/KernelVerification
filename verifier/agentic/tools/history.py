"""Experiment history tools for replay-aware agents."""

from __future__ import annotations

import json
from pathlib import Path

from .registry import ToolContext


def retrieve_experiment_history_schema() -> dict:
    return {
        "type": "object",
        "required": [],
        "properties": {
            "limit": {"type": ["integer", "null"], "minimum": 1},
        },
        "additionalProperties": False,
    }


def retrieve_experiment_history(context: ToolContext, args: dict) -> dict:
    limit = int(args.get("limit") or 20)
    if limit < 1:
        raise ValueError("limit must be >= 1")

    run_dir = Path(context.run_dir) if context.run_dir is not None else None
    if run_dir is None:
        return {"events": [], "source": None, "exists": False}

    path = run_dir / "tool_events.jsonl"
    if not path.exists():
        return {"events": [], "source": str(path), "exists": False}

    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("tool") in {"run_python_probe", "run_claim_probe"}:
            events.append(event)
    return {"events": events[-limit:], "source": str(path), "exists": True}
