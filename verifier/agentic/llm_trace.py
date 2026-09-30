"""Opt-in request/response capture for OpenAI-compatible chat completions.

Only API payloads are captured: client configuration, HTTP headers, and the
environment are never serialized. A request is saved before its API call, and
the raw SDK response is saved before the agent protocol is parsed.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterator
from uuid import uuid4


_id_lock = threading.Lock()
_last_id_ns = 0


def _ordered_call_id() -> str:
    global _last_id_ns
    with _id_lock:
        _last_id_ns = max(time.time_ns(), _last_id_ns + 1)
        return f"{_last_id_ns:020d}_{uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exception_details(error: BaseException) -> dict[str, Any]:
    # SDK exceptions can echo a provider's error body. Keep the diagnostic
    # message, but do not let an echoed credential leak into the trace.
    message = str(error)
    secrets = {
        value for name, value in os.environ.items()
        if value and re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", name, re.I)
    }
    for secret in sorted(secrets, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", message)
    message = re.sub(
        r"(?i)(\bauthorization\b[\"']?\s*[:=]\s*)[^\r\n]+",
        r"\1[REDACTED]", message,
    )
    result: dict[str, Any] = {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": message,
    }
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        result["status_code"] = status_code
    return result


class LLMCallTrace:
    def __init__(self, root: str, request: dict[str, Any]) -> None:
        self.started_monotonic = time.monotonic()
        self.path = Path(root) / _ordered_call_id()
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata: dict[str, Any] = {
            "schema_version": 1,
            "api": "chat.completions",
            "started_at": _utc_now(),
            "status": "started",
            "response_saved": False,
        }
        self._write("request.json", request)
        self._write("metadata.json", self.metadata)
        if os.getenv("AGENTIC_LLM_TRACE_PROGRESS") == "1":
            print(f"LLM trace started: {self.path.name}", flush=True)

    def _write(self, filename: str, value: Any) -> None:
        # Each call owns its directory; replace metadata atomically so an
        # interrupted run still has a readable starting or finishing record.
        temporary = self.path / f".{filename}.tmp"
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.path / filename)

    def record_response(self, response: Any) -> None:
        self._write("response.json", response.model_dump(mode="json"))
        self.metadata["response_saved"] = True
        self.metadata["response_received_at"] = _utc_now()
        if os.getenv("AGENTIC_LLM_TRACE_PROGRESS") == "1":
            raw = response.model_dump(mode="json")
            finish = [c.get("finish_reason") for c in raw.get("choices", [])]
            print(f"LLM trace response: {self.path.name} finish={finish} usage={raw.get('usage')}", flush=True)

    def finish(self, error: BaseException | None = None) -> None:
        self.metadata.update(
            finished_at=_utc_now(),
            duration_s=time.monotonic() - self.started_monotonic,
            status="error" if error is not None else "completed",
        )
        if error is not None:
            self._write("error.json", _exception_details(error))
        self._write("metadata.json", self.metadata)


@contextmanager
def capture_chat_completion(request: dict[str, Any]) -> Iterator[LLMCallTrace | None]:
    """Capture one SDK call only when AGENTIC_LLM_TRACE_DIR is set.

    Trace setup intentionally fails before making a paid request if the
    destination cannot be written. With tracing disabled no path is accessed.
    """
    root = os.getenv("AGENTIC_LLM_TRACE_DIR")
    if not root:
        yield None
        return
    trace = LLMCallTrace(root, request)
    try:
        yield trace
    except BaseException as error:
        trace.finish(error)
        raise
    else:
        trace.finish()
