"""Artifact access tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verifier.dataset import load_entry

from .registry import ToolContext

_DEFAULT_MAX_CHARS = 20000


def load_artifact_schema() -> dict:
    return {
        "type": "object",
        "required": ["entry"],
        "properties": {
            "entry": {"type": "string"},
        },
        "additionalProperties": False,
    }


def inspect_kernel_source_schema() -> dict:
    return {
        "type": "object",
        "required": ["entry"],
        "properties": {
            "entry": {"type": "string"},
            "start_line": {"type": ["integer", "null"], "minimum": 1},
            "end_line": {"type": ["integer", "null"], "minimum": 1},
        },
        "additionalProperties": False,
    }


def inspect_problem_schema() -> dict:
    return {
        "type": "object",
        "required": ["entry"],
        "properties": {
            "entry": {"type": "string"},
        },
        "additionalProperties": False,
    }


def list_artifact_files_schema() -> dict:
    return {
        "type": "object",
        "required": ["entry"],
        "properties": {
            "entry": {"type": "string"},
        },
        "additionalProperties": False,
    }


def read_artifact_file_schema() -> dict:
    return {
        "type": "object",
        "required": ["entry", "path"],
        "properties": {
            "entry": {"type": "string"},
            "path": {"type": "string"},
            "max_chars": {"type": ["integer", "null"], "minimum": 1},
        },
        "additionalProperties": False,
    }


def load_artifact(context: ToolContext, args: dict) -> dict:
    entry = str(args["entry"])
    dataset_dir = Path(str(context.dataset_dir)) if context.dataset_dir is not None else None
    artifact = load_entry(entry, dataset_dir=dataset_dir)

    context.state.entry = entry
    context.state.artifact = {
        "entry": entry,
        "session_dir": artifact.get("session_dir"),
        "passed": bool(artifact.get("passed", False)),
        "status": str(artifact.get("status", "unknown")),
        "rounds": artifact.get("rounds"),
        "kernel_code": artifact.get("kernel_code", ""),
        "test_code": artifact.get("test_code", ""),
        "has_error": bool(artifact.get("error")),
    }
    return context.state.artifact


def inspect_kernel_source(context: ToolContext, args: dict) -> dict:
    entry = str(args["entry"])
    path = _resolve_artifact_path(context, entry, "kernel.py")
    text = _read_text(path)
    lines = text.splitlines()
    total_lines = len(lines)

    start_line = _optional_int(args.get("start_line")) or 1
    end_line = _optional_int(args.get("end_line")) or total_lines
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")

    selected = []
    if total_lines:
        capped_end = min(end_line, total_lines)
        for line_no in range(start_line, capped_end + 1):
            selected.append({"line": line_no, "text": lines[line_no - 1]})

    return {
        "entry": entry,
        "path": "kernel.py",
        "start_line": start_line,
        "end_line": min(end_line, total_lines) if total_lines else 0,
        "total_lines": total_lines,
        "lines": selected,
        "content": _format_numbered_lines(selected),
    }


def inspect_problem(context: ToolContext, args: dict) -> dict:
    entry = str(args["entry"])
    path = _entry_dir(context, entry) / "problem.txt"
    if not path.exists():
        return {
            "entry": entry,
            "path": "problem.txt",
            "exists": False,
            "content": "",
        }
    return {
        "entry": entry,
        "path": "problem.txt",
        "exists": True,
        "content": _read_text(path),
    }


def list_artifact_files(context: ToolContext, args: dict) -> dict:
    entry = str(args["entry"])
    root = _entry_dir(context, entry)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        files.append(
            {
                "path": rel_path,
                "kind": _file_kind(rel_path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"entry": entry, "files": files}


def read_artifact_file(context: ToolContext, args: dict) -> dict:
    entry = str(args["entry"])
    rel_path = str(args["path"])
    max_chars = _optional_int(args.get("max_chars")) or _DEFAULT_MAX_CHARS
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    path = _resolve_artifact_path(context, entry, rel_path)
    if not path.is_file():
        raise FileNotFoundError(f"artifact file not found: {rel_path}")

    content = _read_text(path)
    truncated = len(content) > max_chars
    return {
        "entry": entry,
        "path": rel_path,
        "kind": _file_kind(rel_path),
        "content": content[:max_chars],
        "truncated": truncated,
        "size_chars": len(content),
    }


def _dataset_root(context: ToolContext) -> Path:
    return Path(str(context.dataset_dir)) if context.dataset_dir is not None else Path("dataset")


def _entry_dir(context: ToolContext, entry: str) -> Path:
    root = _dataset_root(context)
    entry_dir = root / entry
    if not entry_dir.exists() or not entry_dir.is_dir():
        raise FileNotFoundError(f"dataset entry not found: {entry_dir}")
    return entry_dir


def _resolve_artifact_path(context: ToolContext, entry: str, rel_path: str) -> Path:
    if not rel_path or Path(rel_path).is_absolute():
        raise ValueError("artifact path must be a non-empty relative path")

    entry_dir = _entry_dir(context, entry).resolve()
    path = (entry_dir / rel_path).resolve()
    if not path.is_relative_to(entry_dir):
        raise ValueError(f"artifact path escapes entry directory: {rel_path}")
    return path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _format_numbered_lines(lines: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item['line']}: {item['text']}" for item in lines)


def _file_kind(path: str) -> str:
    name = Path(path).name
    if name == "kernel.py":
        return "kernel_source"
    if name == "test.py":
        return "test_source"
    if name == "problem.txt":
        return "problem"
    if name == "meta.json":
        return "metadata"
    if name.startswith("seed_") and name.endswith(".py"):
        return "seed_source"
    if name == "error.txt":
        return "error_log"
    return "other"
