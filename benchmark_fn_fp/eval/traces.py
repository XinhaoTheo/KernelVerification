"""Keep the complete record of every run, always.

Fifteen-odd evaluation runs were made before this existed, and not one of them
left a complete record. The Modal runners kept the last 20,000 characters of
transcript.md and nothing else; run.json, tool_events.jsonl, claims.json and
every probe's source and captured output lived in the container and died with
it. Findings that later turned out to matter -- a Skeptic spending its turn
re-reading material already in its prompt, a tool rejecting the Describer's
first call in every single run, an agent reaching a correct verdict for three
wrong reasons -- were only visible once full traces existed, and every one of
them had been happening invisibly the whole time.

So capture is not a separate script to remember to use. Every runner writes its
trace here, and a run that produced no trace is a bug in the runner.

Layout, one directory per case per arm:

    benchmark_fn_fp/traces/<case_id>/<arm>/

`arm` is the configuration being measured -- "solo", "debate", "single_call" --
so the same case under two configurations sits side by side.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACES_DIR = REPO_ROOT / "benchmark_fn_fp" / "traces"


def pack_run_dir(run_dir: Path) -> bytes | None:
    """Tar a whole orchestrator run directory for transport out of a container."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as tar:
        tar.add(str(run_dir), arcname=".")
    return blob.getvalue()


def write_trace(case_id: str, arm: str, *, tar: bytes | None = None,
                files: dict[str, str] | None = None) -> Path:
    """Write one run's complete record under traces/<case_id>/<arm>/.

    `tar` is an archive from pack_run_dir; `files` is any additional plain-text
    content to drop alongside it (a runner log, a raw prompt and response for an
    arm that has no orchestrator run directory at all).
    """
    dest = TRACES_DIR / case_id / arm
    dest.mkdir(parents=True, exist_ok=True)
    if tar:
        with tarfile.open(fileobj=io.BytesIO(tar), mode="r:gz") as archive:
            archive.extractall(dest)
    for name, text in (files or {}).items():
        (dest / name).write_text(text or "", encoding="utf-8")
    return dest


def write_json(case_id: str, arm: str, name: str, payload: Any) -> Path:
    dest = TRACES_DIR / case_id / arm
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
