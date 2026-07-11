"""Local execution tools for agent-generated probes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from verifier.agentic.ledger import ClaimLedger, LedgerError
from verifier.agentic.state import ClaimStatus, EvidenceKind

from .registry import ToolContext

_DEFAULT_TIMEOUT_S = 60
_MAX_TIMEOUT_S = 600
_CLAIM_PROBE_SUPPORTS = [
    ClaimStatus.CONFIRMED.value,
    ClaimStatus.REBUTTED.value,
    ClaimStatus.INCONCLUSIVE.value,
]


def run_python_probe_schema() -> dict:
    return {
        "type": "object",
        "required": ["code"],
        "properties": {
            "code": {"type": "string"},
            "timeout_s": {"type": ["integer", "null"], "minimum": 1, "maximum": _MAX_TIMEOUT_S},
            "use_gpu": {"type": ["boolean", "null"], "default": True},
        },
        "additionalProperties": False,
    }


def run_claim_probe_schema() -> dict:
    return {
        "type": "object",
        "required": ["claim_id", "code"],
        "properties": {
            "claim_id": {"type": "string"},
            "code": {"type": "string"},
            "expected_signal": {"type": ["string", "null"]},
            "timeout_s": {"type": ["integer", "null"], "minimum": 1, "maximum": _MAX_TIMEOUT_S},
            "use_gpu": {"type": ["boolean", "null"], "default": True},
        },
        "additionalProperties": False,
    }


def finalize_probe_evidence_schema() -> dict:
    return {
        "type": "object",
        "required": ["event_id", "supports", "summary"],
        "properties": {
            "event_id": {"type": "string"},
            "supports": {"type": "string", "enum": _CLAIM_PROBE_SUPPORTS},
            "summary": {"type": "string"},
            "status": {"type": ["string", "null"], "enum": _CLAIM_PROBE_SUPPORTS + [None]},
            "data": {"type": ["object", "null"]},
        },
        "additionalProperties": False,
    }


def run_python_probe(context: ToolContext, args: dict) -> dict:
    return _execute_python_probe(context, args)


def run_claim_probe(context: ToolContext, args: dict) -> dict:
    claim_id = str(args["claim_id"])
    claim = ClaimLedger(context.state).get_claim(claim_id)
    result = _execute_python_probe(context, args)
    expected_signal = _optional_str(args.get("expected_signal"))
    result.update(
        {
            "claim_id": claim.id,
            "claim_statement": claim.statement,
            "expected_signal": expected_signal,
            "evidence_draft": {
                "claim_id": claim.id,
                "kind": EvidenceKind.RUNTIME_PROBE.value,
                "tool_event_id": result["event_id"],
                "supports": "needs_interpretation",
                "summary": _draft_summary(result, expected_signal),
                "data": _probe_evidence_data(result, expected_signal),
            },
        }
    )
    return result


def finalize_probe_evidence(context: ToolContext, args: dict) -> dict:
    event_id = str(args["event_id"])
    event = _tool_event_by_id(context, event_id)
    if event.tool != "run_claim_probe":
        raise LedgerError(f"tool event {event_id} is not a run_claim_probe event")
    output = event.output
    claim_id = str(output.get("claim_id") or "")
    if not claim_id:
        raise LedgerError(f"run_claim_probe event {event_id} has no claim_id")
    if _probe_event_consumed(context, event_id):
        raise LedgerError(f"probe event {event_id} already has evidence")

    data = args.get("data") or {}
    if not isinstance(data, dict):
        raise LedgerError("data must be an object")
    merged_data = _probe_evidence_data(output, _optional_str(output.get("expected_signal")))
    merged_data.update(data)

    ledger = ClaimLedger(context.state)
    evidence = ledger.append_evidence(
        claim_id=claim_id,
        kind=EvidenceKind.RUNTIME_PROBE.value,
        summary=str(args["summary"]),
        supports=str(args["supports"]),
        tool_event_id=event_id,
        data=merged_data,
    )
    status = str(args.get("status") or args["supports"])
    claim = ledger.update_claim_status(claim_id=claim_id, status=status)
    return {"claim": claim.to_dict(), "evidence": evidence.to_dict()}


def _execute_python_probe(context: ToolContext, args: dict) -> dict:
    code = str(args["code"])
    if not code.strip():
        raise ValueError("probe code must be non-empty")

    timeout_s = _optional_int(args.get("timeout_s")) or _DEFAULT_TIMEOUT_S
    if timeout_s < 1 or timeout_s > _MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be between 1 and {_MAX_TIMEOUT_S}")
    use_gpu = bool(args.get("use_gpu", True))

    event_id = context.current_tool_event_id or f"t{len(context.state.tool_events) + 1}"
    run_dir = _run_dir(context)
    probes_dir = run_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    probe_path = probes_dir / f"{event_id}_probe.py"
    stdout_path = probes_dir / f"{event_id}_stdout.txt"
    stderr_path = probes_dir / f"{event_id}_stderr.txt"
    json_path = probes_dir / f"{event_id}_json_result.json"

    probe_path.write_text(code, encoding="utf-8")

    cwd = _probe_cwd(context, run_dir)
    env = _probe_env(context, cwd=cwd, use_gpu=use_gpu)

    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, str(probe_path)],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = _coerce_output(exc.stdout)
        stderr = _coerce_output(exc.stderr)
    duration_s = time.monotonic() - started

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    json_result, json_parse_error = _parse_last_stdout_json(stdout)
    if json_result is not None:
        json_path.write_text(json.dumps(json_result, indent=2, sort_keys=True), encoding="utf-8")

    artifacts = [
        _artifact_ref(run_dir, probe_path, "probe_code", "Python probe code executed by local runtime."),
        _artifact_ref(run_dir, stdout_path, "stdout", "Captured stdout from the probe process."),
        _artifact_ref(run_dir, stderr_path, "stderr", "Captured stderr from the probe process."),
    ]
    if json_result is not None:
        artifacts.append(
            _artifact_ref(run_dir, json_path, "json_result", "Parsed JSON object from the last non-empty stdout line.")
        )

    return {
        "event_id": event_id,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "duration_s": round(duration_s, 6),
        "use_gpu": use_gpu,
        "cwd": str(cwd),
        "stdout": stdout,
        "stderr": stderr,
        "json_result": json_result,
        "json_parse_error": json_parse_error,
        "artifacts": artifacts,
    }


def _run_dir(context: ToolContext) -> Path:
    if context.run_dir is not None:
        run_dir = Path(context.run_dir)
    else:
        run_dir = Path("agentic_runs") / (context.state.entry or "adhoc")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def _probe_cwd(context: ToolContext, run_dir: Path) -> Path:
    artifact = context.state.artifact or {}
    session_dir = artifact.get("session_dir")
    if isinstance(session_dir, str) and session_dir:
        path = Path(session_dir)
        if path.exists() and path.is_dir():
            return path.resolve()
    return run_dir


def _probe_env(context: ToolContext, *, cwd: Path, use_gpu: bool) -> dict[str, str]:
    env = os.environ.copy()
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    repo_root = Path(__file__).resolve().parents[3]
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_parts = [str(cwd), str(repo_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["KV_AGENTIC_RUN_DIR"] = str(_run_dir(context))
    return env


def _parse_last_stdout_json(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None, "stdout has no non-empty lines"

    last_line = lines[-1]
    try:
        value = json.loads(last_line)
    except json.JSONDecodeError as exc:
        return None, f"last stdout line is not JSON: {exc.msg}"
    if not isinstance(value, dict):
        return None, "last stdout JSON value is not an object"
    return value, None


def _artifact_ref(run_dir: Path, path: Path, kind: str, description: str) -> dict[str, str]:
    return {
        "kind": kind,
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": _sha256(path),
        "description": description,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _draft_summary(result: dict[str, Any], expected_signal: str | None) -> str:
    if result.get("timed_out"):
        return "Probe timed out and needs Experimenter interpretation."
    if result.get("json_result") is not None:
        return "Probe produced a JSON result that needs Experimenter interpretation."
    if result.get("exit_code") == 0:
        return "Probe completed successfully and needs Experimenter interpretation."
    return "Probe failed or produced nonzero exit and needs Experimenter interpretation."


def _probe_evidence_data(result: dict[str, Any], expected_signal: str | None) -> dict[str, Any]:
    return {
        "expected_signal": expected_signal,
        "exit_code": result.get("exit_code"),
        "timed_out": result.get("timed_out"),
        "timeout_s": result.get("timeout_s"),
        "duration_s": result.get("duration_s"),
        "stdout": result.get("stdout"),
        "stderr": result.get("stderr"),
        "json_result": result.get("json_result"),
        "json_parse_error": result.get("json_parse_error"),
        "artifacts": result.get("artifacts", []),
    }


def _tool_event_by_id(context: ToolContext, event_id: str):
    for event in context.state.tool_events:
        if event.id == event_id:
            return event
    raise LedgerError(f"unknown tool event id: {event_id}")


def _probe_event_consumed(context: ToolContext, event_id: str) -> bool:
    return any(
        evidence.tool_event_id == event_id
        for claim in context.state.claims
        for evidence in claim.evidence
    )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
