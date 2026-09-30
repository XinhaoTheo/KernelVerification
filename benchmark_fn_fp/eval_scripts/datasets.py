"""Answer-free case paths and local, pre-payment frozen-source checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from .case_registry import validation_path
except ImportError:
    from case_registry import validation_path

DATASETS = ("benchmark_fn_fp", "correlation_pair", "numerical_challenges", "evidence_challenges",
            "numerical_pilot")
FROZEN_DATASETS = ("correlation_pair", "numerical_challenges", "evidence_challenges", "numerical_pilot")


def cases_dir(repo: Path, dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(repo) / "benchmark_fn_fp"
    if dataset == "benchmark_fn_fp":
        return root / "triton_eval_cases"
    return root / dataset / "eval_cases"


def checked_case_hashes(repo: Path, dataset: str, name: str) -> dict[str, str]:
    """Refuse an unvalidated or edited challenge before creating a paid job.

    Only hashes leave this helper. Ground-truth labels stay outside prompts and
    outside the image in which tool-enabled agents execute.
    """
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid case name: {name!r}")
    directory = cases_dir(repo, dataset)
    hashes = {f"{kind}_sha256": hashlib.sha256(
        (directory / name / filename).read_bytes()).hexdigest()
        for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt"))}
    if dataset in FROZEN_DATASETS:
        frozen_path = validation_path(Path(repo) / "benchmark_fn_fp", dataset)
        validated = json.loads(frozen_path.read_text())["cases"]
        if name not in validated:
            raise ValueError(f"No GPU validation for {dataset}/{name}")
        frozen = validated[name]
        if frozen.get("ground_truth") not in {"trust", "reject"}:
            raise ValueError(f"GPU validation has no usable label for {dataset}/{name}")
        for key, value in hashes.items():
            if frozen.get(key) != value:
                raise ValueError(f"Frozen case changed: {dataset}/{name}/{key}")
    return hashes
