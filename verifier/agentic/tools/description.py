"""Description model tools for the Describer agent."""

from __future__ import annotations

from typing import cast

from verifier.agentic.state import (
    DescriptionTask,
    DescriptionTaskStatus,
    DescriptionUpdate,
    JsonValue,
    Role,
    utc_now_iso,
)

from .registry import ToolContext

_REASON_KINDS = [
    "contract_scope",
    "source_interpretation",
    "probe_anomaly",
    "verdict_blocker",
    "other",
]
_MODEL_FIELDS = [
    "contract_model",
    "kernel_model",
    "risk_map",
    "scope_notes",
    "open_questions",
]
_MAX_DESCRIPTION_REQUESTS_PER_TURN = 2
_MAX_OPEN_DESCRIPTION_TASKS = 4
_MAX_TOTAL_DESCRIPTION_TASKS = 20


def request_description_schema() -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], {
        "type": "object",
        "required": ["reason_kind", "question"],
        "properties": {
            "reason_kind": {"type": "string", "enum": _REASON_KINDS},
            "question": {"type": "string"},
            "related_claims": {"type": "array", "items": {"type": "string"}, "default": []},
            "source_refs": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        "additionalProperties": False,
    })


def record_description_update_schema() -> dict[str, JsonValue]:
    array_of_strings = {"type": "array", "items": {"type": "string"}, "default": []}
    return cast(dict[str, JsonValue], {
        "type": "object",
        "required": ["summary"],
        "properties": {
            "summary": {"type": "string"},
            "task_id": {"type": ["string", "null"]},
            "resolved_task_ids": array_of_strings,
            "contract_model": array_of_strings,
            "kernel_model": array_of_strings,
            "risk_map": array_of_strings,
            "scope_notes": array_of_strings,
            "open_questions": array_of_strings,
            "impact_on_claims": array_of_strings,
        },
        "additionalProperties": False,
    })


def request_description(context: ToolContext, args: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if context.current_role == Role.DESCRIBER.value:
        raise ValueError("Describer cannot request another description task")
    _enforce_request_limits(context)

    reason_kind = str(args["reason_kind"]).strip()
    if reason_kind not in _REASON_KINDS:
        raise ValueError(f"invalid reason_kind: {reason_kind}")
    question = str(args["question"]).strip()
    if not question:
        raise ValueError("question must be non-empty")

    task = DescriptionTask(
        id=f"d{len(context.state.description_tasks) + 1}",
        reason_kind=reason_kind,
        question=question,
        requested_by=context.current_role or "unknown",
        related_claims=_string_list(args.get("related_claims"), field="related_claims"),
        source_refs=_string_list(args.get("source_refs"), field="source_refs"),
        request_turn=context.current_turn,
    )
    context.state.description_tasks.append(task)
    return task.to_dict()


def record_description_update(context: ToolContext, args: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if context.current_role != Role.DESCRIBER.value:
        raise ValueError("record_description_update can only be called by the describer")

    summary = str(args["summary"]).strip()
    if not summary:
        raise ValueError("summary must be non-empty")

    task_ids = _resolved_task_ids(args)
    field_values = {field: _string_list(args.get(field), field=field) for field in _MODEL_FIELDS}
    impact_on_claims = _string_list(args.get("impact_on_claims"), field="impact_on_claims")

    for field, values in field_values.items():
        _append_new(getattr(context.state.description_model, field), values)

    update = DescriptionUpdate(
        id=f"du{len(context.state.description_updates) + 1}",
        summary=summary,
        task_ids=task_ids,
        contract_model=field_values["contract_model"],
        kernel_model=field_values["kernel_model"],
        risk_map=field_values["risk_map"],
        scope_notes=field_values["scope_notes"],
        open_questions=field_values["open_questions"],
        impact_on_claims=impact_on_claims,
        created_by=context.current_role or Role.DESCRIBER.value,
        turn=context.current_turn,
    )
    context.state.description_updates.append(update)

    resolved_tasks = []
    for task_id in task_ids:
        task = _get_description_task(context, task_id)
        task.status = DescriptionTaskStatus.RESOLVED
        task.response_summary = summary
        task.resolved_at = utc_now_iso()
        task.resolved_turn = context.current_turn
        resolved_tasks.append(task.to_dict())

    return {
        "update": update.to_dict(),
        "description_model": context.state.description_model.to_dict(),
        "resolved_tasks": resolved_tasks,
    }


def _resolved_task_ids(args: dict[str, JsonValue]) -> list[str]:
    ids = []
    task_id = args.get("task_id")
    if task_id is not None and str(task_id).strip():
        ids.append(str(task_id).strip())
    ids.extend(_string_list(args.get("resolved_task_ids"), field="resolved_task_ids"))
    deduped = []
    for item in ids:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _get_description_task(context: ToolContext, task_id: str) -> DescriptionTask:
    for task in context.state.description_tasks:
        if task.id == task_id:
            return task
    raise ValueError(f"unknown description task id: {task_id}")


def _string_list(value: JsonValue | None, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field}[{index}] must be a string")
        cleaned = item.strip()
        if cleaned:
            result.append(cleaned)
    return result


def _append_new(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _enforce_request_limits(context: ToolContext) -> None:
    if context.current_turn_tool_counts.get("request_description", 0) > _MAX_DESCRIPTION_REQUESTS_PER_TURN:
        raise ValueError(f"at most {_MAX_DESCRIPTION_REQUESTS_PER_TURN} description requests are allowed per turn")
    if len(context.state.description_tasks) >= _MAX_TOTAL_DESCRIPTION_TASKS:
        raise ValueError(f"at most {_MAX_TOTAL_DESCRIPTION_TASKS} description tasks are allowed per run")
    open_count = sum(
        1
        for task in context.state.description_tasks
        if _status_value(task.status) == DescriptionTaskStatus.OPEN.value
    )
    if open_count >= _MAX_OPEN_DESCRIPTION_TASKS:
        raise ValueError(f"at most {_MAX_OPEN_DESCRIPTION_TASKS} description tasks may be open at once")


def _status_value(status: DescriptionTaskStatus | str) -> str:
    return status.value if isinstance(status, DescriptionTaskStatus) else str(status)
