"""Answer-free case paths and local, pre-payment frozen-source checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from .case_registry import (canonical_dataset, case_sort_key, dataset_members,
                                dataset_root, is_active_case, load_registry, validation_path)
except ImportError:
    from case_registry import (canonical_dataset, case_sort_key, dataset_members,
                               dataset_root, is_active_case, load_registry, validation_path)

DATASETS = ("benchmark_fn_fp", "correlation_pair", "numerical_challenges", "evidence_challenges",
            "numerical_pilot", "single_call_vs_tools_challenges", "solo_vs_debate_challenges",
            "real_kernel_challenges")
FROZEN_DATASETS = ("correlation_pair", "numerical_challenges", "evidence_challenges", "numerical_pilot",
                   "real_kernel_challenges")


def cases_dir(repo: Path, dataset: str) -> Path:
    dataset = canonical_dataset(dataset)
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(repo) / "benchmark_fn_fp"
    if dataset in {"benchmark_fn_fp", "correlation_pair", "numerical_challenges", "evidence_challenges",
                   "real_kernel_challenges"}:
        return root / "triton_eval_cases"
    return dataset_root(root, dataset) / "eval_cases"


def case_names(repo: Path, dataset: str) -> list[str]:
    """List a logical dataset without mixing cases that share a public directory."""
    directory = cases_dir(repo, dataset)
    details = load_registry(Path(repo) / "benchmark_fn_fp").get("case_details", {})
    if details:
        return sorted((name for name, row in details.items()
                       if row["dataset"] in dataset_members(dataset) and is_active_case(row)),
                      key=case_sort_key)
    # Older checkouts and isolated tests may not have a registry.
    return sorted((p.name for p in directory.iterdir()
                   if p.is_dir() and (p / "kernel.py").is_file()), key=case_sort_key)


def ensure_active_dataset(dataset: str) -> None:
    if canonical_dataset(dataset) == "numerical_pilot":
        raise ValueError("numerical_pilot is archived; case_82–105 are excluded from new experiments")


def case_dataset(repo: Path, dataset: str, name: str, *, require_active: bool = False) -> str:
    """Resolve each case's source before validating or recording a mixed collection."""
    details = load_registry(Path(repo) / "benchmark_fn_fp").get("case_details", {})
    if require_active:
        ensure_active_dataset(dataset)
    if not details:
        return canonical_dataset(dataset)
    if name not in details:
        raise ValueError(f"Case is absent from case_map.json: {name}")
    detail = details[name]
    expected = detail["dataset"]
    if expected not in dataset_members(dataset):
        raise ValueError(f"{name} belongs to {expected}; pass its dataset or collection")
    if require_active and not is_active_case(detail):
        raise ValueError(f"{name} is archived; excluded from new experiments")
    return expected


def checked_case_hashes(repo: Path, dataset: str, name: str) -> dict[str, str]:
    """Refuse an unvalidated or edited challenge before creating a paid job.

    Only hashes leave this helper. Ground-truth labels stay outside prompts and
    outside the image in which tool-enabled agents execute.
    """
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid case name: {name!r}")
    directory = cases_dir(repo, dataset)
    dataset = case_dataset(repo, dataset, name)
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
