"""Lossless trace storage; every trial has its own directory.

GLM runs share traces_glm/<case>/<arm>/<trial>/. Metadata preserves exact
model IDs and datasets. Existing flat traces remain readable; writes never
replace differing files from an earlier run.
"""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark_fn_fp"
TRACES_GLOB = "traces_*"


def new_trial_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative trace path: {value!r}")
    return path


def traces_root(traces_dir: str) -> Path:
    return BENCHMARK_DIR / _relative(traces_dir)


def all_traces_roots() -> list[Path]:
    return sorted(p for p in BENCHMARK_DIR.glob(TRACES_GLOB) if p.is_dir() and not p.is_symlink())


def trace_path(case_id: str, arm: str, *, traces_dir: str, trial: str) -> Path:
    for part in (case_id, arm, trial):
        if len(_relative(part).parts) != 1:
            raise ValueError(f"Trace identity components must be single names: {part!r}")
    return traces_root(traces_dir) / case_id / arm / trial


def update_trace_metadata(dest: Path, **updates: Any) -> None:
    path = dest / "trace_meta.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update(updates)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def reserve_trace(case_id: str, arm: str, *, traces_dir: str, trial: str,
                  metadata: dict[str, Any] | None = None) -> Path:
    try:
        from .models import pricing_snapshot
    except ImportError:
        from models import pricing_snapshot
    metadata = dict(metadata or {})
    if metadata.get("model") and "pricing_snapshot" not in metadata:
        metadata["pricing_snapshot"] = pricing_snapshot(metadata["model"])
    dest = trace_path(case_id, arm, traces_dir=traces_dir, trial=trial)
    dest.mkdir(parents=True, exist_ok=False)
    update_trace_metadata(dest, **{"schema_version": 2, "case": case_id, "arm": arm,
        "trial": trial, "dataset": "benchmark_fn_fp", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), **(metadata or {})})
    return dest


def pack_run_dir(run_dir: Path) -> bytes | None:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as tar:
        tar.add(str(run_dir), arcname=".")
    return blob.getvalue()


def write_trace(case_id: str, arm: str, *, traces_dir: str, tar: bytes | None = None,
                files: dict[str, str] | None = None, trial: str | None = None,
                metadata: dict[str, Any] | None = None) -> Path:
    """Save a whole run, including raw LLM calls and all probe files.

    A caller may reserve the trial before the paid call, then append its results.
    Identical files are idempotent; differing files raise instead of overwriting.
    Archive paths and all collisions are checked before any payload is written.
    """
    trial = trial or new_trial_id()
    dest = trace_path(case_id, arm, traces_dir=traces_dir, trial=trial)
    payload: dict[Path, bytes] = {}
    if tar:
        with tarfile.open(fileobj=io.BytesIO(tar), mode="r:gz") as archive:
            for member in archive.getmembers():
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe trace archive member: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Non-regular trace archive member: {member.name}")
                key = Path(*rel.parts)
                if not key.parts:
                    raise ValueError("Archive file cannot name the trace directory")
                data = archive.extractfile(member).read()
                if key in payload and payload[key] != data:
                    raise ValueError(f"Conflicting archive members: {key}")
                payload[key] = data
    for name, value in (files or {}).items():
        key = _relative(name)
        data = (value or "").encode("utf-8")
        if key in payload and payload[key] != data:
            raise ValueError(f"Conflicting trace payload: {key}")
        payload[key] = data
    if Path("trace_meta.json") in payload:
        raise ValueError("Trace metadata is managed separately from payload files")
    for name in payload:
        if any(parent in payload for parent in name.parents):
            raise ValueError(f"A trace file is also used as a directory: {name}")
    for name, value in payload.items():
        path = dest / name
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise FileExistsError(f"Refusing a symlink in trace destination: {parent}")
            if parent == dest:
                break
            if parent != path and parent.exists() and not parent.is_dir():
                raise FileExistsError(f"Trace parent is not a directory: {parent}")
        if path.exists() and path.read_bytes() != value:
            raise FileExistsError(f"Refusing to overwrite trace: {path}")
    dest.mkdir(parents=True, exist_ok=True)
    for name, value in payload.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(value)
    update_trace_metadata(dest, **{"schema_version": 2, "case": case_id, "arm": arm,
        "trial": trial, **(metadata or {})})
    return dest


def write_json(case_id: str, arm: str, name: str, payload: Any, *, traces_dir: str,
               trial: str | None = None) -> Path:
    dest = write_trace(case_id, arm, traces_dir=traces_dir, trial=trial,
        files={name: json.dumps(payload, indent=2, ensure_ascii=False, default=str)})
    return dest / name


def single_call_readable_files(*, system: str, user: str, response: str,
                               thinking: str = "", usage: dict | None = None) -> dict[str, str]:
    """Human transcript plus verdict view; raw response remains authoritative."""
    try:
        verdict = json.loads(response)
        if not isinstance(verdict, dict):
            verdict = {"verdict": None, "status": "invalid_response"}
    except ValueError:
        verdict = {"verdict": None, "status": "no_final_verdict"}
    return {"response_thinking.txt": thinking,
        "verdict.json": json.dumps(verdict, indent=2, ensure_ascii=False),
        "transcript.md": "\n\n".join(["# Single-call trace", "## System prompt", system,
            "## User prompt", user, "## Provider reasoning (verbatim)", thinking or "(not supplied)",
            "## Final response (verbatim)", response or "(no final text)",
            "## Verdict", json.dumps(verdict, indent=2, ensure_ascii=False),
            "## Usage and stop reason", json.dumps(usage or {}, indent=2, ensure_ascii=False)]) + "\n"}


def iter_trace_records(benchmark_dir: Path | None = None):
    """Discover unified trial leaves and legacy layouts, preserving provenance."""
    try:
        from .models import PROFILES
        from .case_registry import load_registry, resolve_trace_case
    except ImportError:
        from models import PROFILES
        from case_registry import load_registry, resolve_trace_case
    benchmark = Path(benchmark_dir) if benchmark_dir is not None else BENCHMARK_DIR
    registry = load_registry(benchmark)
    legacy_models = {"traces_glm": "z-ai/glm-5.3-flash",
                     "traces_glm_fireworks": "accounts/fireworks/models/glm-5p3",
                     "traces_opus5": "claude-opus-5"}
    for root in sorted(benchmark.glob(TRACES_GLOB)):
        if not root.is_dir() or root.is_symlink():
            continue
        candidates = {p.parent for name in ("run.json", "usage.json", "trace_meta.json")
                      for p in root.rglob(name)}
        for directory in sorted(candidates):
            parts = directory.relative_to(benchmark).parts
            indexes = [i for i, part in enumerate(parts) if part in ("solo", "debate", "single_call")]
            if not indexes:
                continue
            arm_index = indexes[0]
            if arm_index < 2 or len(parts) > arm_index + 2:
                continue  # nested API/probe files are not separate trials
            if len(parts)>arm_index+1 and parts[arm_index+1] in {"probes", "llm_calls"}:
                continue
            meta_path = directory / "trace_meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            model = meta.get("model")
            if not model:
                # Only historical layouts identify a provider without metadata.
                if root.name == "traces_glm" and len(parts) > 1 and parts[1] in {"fireworks", "openrouter"}:
                    model = {"fireworks": "accounts/fireworks/models/glm-5p3",
                             "openrouter": "z-ai/glm-5.3-flash"}[parts[1]]
                elif len(parts) == arm_index + 1:
                    model = legacy_models.get(root.name)
            if not model:
                matches = [p for p in PROFILES.values()
                           if directory.is_relative_to(benchmark / p.traces_dir)]
                # A shared family path alone cannot choose an API model or price.
                model = matches[0].model if len(matches) == 1 else None
            profile = PROFILES.get(model)
            case, original_case, dataset = resolve_trace_case(parts[arm_index-1], meta, registry)
            yield {"path": directory, "metadata": meta,
                "case": case, "original_case": original_case,
                "arm": meta.get("arm", parts[arm_index]),
                "trial": meta.get("trial", parts[arm_index+1] if len(parts)>arm_index+1 else "legacy"),
                "dataset": dataset, "model": model,
                "provider": meta.get("provider", profile.provider if profile else None)}
