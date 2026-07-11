"""On-disk dataset of generated kernel artifacts for the agentic verifier.

Each entry is a self-contained directory:
    dataset/<name>/
        problem.txt        # the prompt fed to KernelAgent
        kernel.py          # final kernel code (best attempt even on failure)
        test.py            # primary test (test_0.py from session_dir)
        seed_0.py, ...     # initial seed kernels (for the author agent)
        meta.json          # { name, passed, status, rounds, source, raw, ... }
        error.txt          # only on failure: stderr + stdout from last attempt

Failed entries are saved too — downstream pipeline decides whether a failure
is "real bug we should learn from" or "test was wrong / spec ambiguous".
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DATASET_DIR = Path("dataset")


def save_entry(
    name: str,
    artifact: dict[str, Any],
    *,
    dataset_dir: Path | None = None,
) -> Path:
    """Persist a generator artifact as a dataset entry.

    Saves both passing and failing kernels. For failures, kernel.py contains
    the best (longest-fought) attempted kernel, and error.txt records the
    last round's stderr/stdout for human reading. meta.json carries a
    `status` field for programmatic filtering downstream.

    Overwrites an existing entry of the same name.
    """
    base = (dataset_dir or DEFAULT_DATASET_DIR) / name
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    session = Path(artifact["session_dir"])

    src_problem = session / "problem.txt"
    if src_problem.exists():
        shutil.copy(src_problem, base / "problem.txt")

    (base / "kernel.py").write_text(artifact.get("kernel_code", ""))
    (base / "test.py").write_text(artifact.get("test_code", ""))

    for seed in sorted(session.glob("seed_*.py")):
        shutil.copy(seed, base / seed.name)

    err = artifact.get("error") or {}
    err_stderr = err.get("stderr", "")
    err_stdout = err.get("stdout", "")

    meta = {
        "name": name,
        "passed": bool(artifact.get("passed", False)),
        "status": artifact.get("status", "unknown"),
        "rounds": artifact.get("rounds"),
        "source": "kernelagent",
        "error_stderr_len": len(err_stderr),
        "error_stdout_len": len(err_stdout),
        "raw": artifact.get("raw"),
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if not meta["passed"] and (err_stderr or err_stdout):
        (base / "error.txt").write_text(
            f"=== STATUS: {meta['status']} (rounds={meta['rounds']}) ===\n\n"
            f"=== STDERR ===\n{err_stderr}\n\n"
            f"=== STDOUT ===\n{err_stdout}\n"
        )

    return base


def load_entry(
    name: str,
    *,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    """Materialize a dataset entry as an artifact dict.

    The returned dict mirrors what verifier.generator.generate_kernel returns,
    `session_dir` points at the dataset entry directory itself so local tools can import and inspect artifacts.
    """
    base = (dataset_dir or DEFAULT_DATASET_DIR) / name
    if not base.exists() or not (base / "meta.json").exists():
        raise FileNotFoundError(f"dataset entry not found: {base}")

    meta = json.loads((base / "meta.json").read_text())
    kernel_code = _read_or_empty(base / "kernel.py")
    test_code = _read_or_empty(base / "test.py")
    error_txt = _read_or_empty(base / "error.txt")

    return {
        "kernel_code": kernel_code,
        "test_code": test_code,
        "passed": bool(meta.get("passed", False)),
        "status": meta.get("status", "unknown"),
        "rounds": meta.get("rounds"),
        "error": {"text": error_txt} if error_txt else {},
        "session_dir": str(base.resolve()),
        "raw": meta,
    }


def iter_entries(*, dataset_dir: Path | None = None) -> Iterator[str]:
    """Yield names of all entries in the dataset, sorted."""
    base = dataset_dir or DEFAULT_DATASET_DIR
    if not base.exists():
        return
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and (entry / "meta.json").exists():
            yield entry.name


def _read_or_empty(p: Path) -> str:
    return p.read_text() if p.exists() else ""
