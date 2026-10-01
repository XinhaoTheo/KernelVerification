"""Canonical numeric identities without rewriting historical experiment bytes."""
from __future__ import annotations

import json
from pathlib import Path
import re

DATASET_DIRECTORIES = {
    "numerical_challenges": "single_call_vs_tools_challenges",
    "evidence_challenges": "solo_vs_debate_challenges",
}


def canonical_dataset(dataset: str) -> str:
    """Keep trace identities stable while accepting readable directory names."""
    return {directory: identity for identity, directory in DATASET_DIRECTORIES.items()}.get(dataset, dataset)


def dataset_root(benchmark: Path, dataset: str) -> Path:
    identity = canonical_dataset(dataset)
    root = Path(benchmark) / DATASET_DIRECTORIES.get(identity, identity)
    legacy = Path(benchmark) / identity
    # Historical checkouts and test fixtures can still use their original layout.
    return legacy if not root.exists() and legacy.exists() else root


def case_sort_key(name: str):
    match = re.fullmatch(r"case_(\d+)", name)
    return (0, int(match.group(1))) if match else (1, name)


def load_registry(benchmark: Path) -> dict:
    path = Path(benchmark) / "case_map.json"
    return json.loads(path.read_text()) if path.exists() else {}


def validation_path(benchmark: Path, dataset: str) -> Path:
    dataset = canonical_dataset(dataset)
    root = dataset_root(benchmark, dataset)
    if dataset == "numerical_pilot":
        # The pilot's answer key already contains its T4 measurements and
        # frozen public-file hashes; no new GPU result is implied by this route.
        return root / "answer_key.json"
    if dataset in {"correlation_pair", "numerical_challenges", "evidence_challenges"}:
        private = root / "private_data" / "validation_gpu.json"
        if private.exists():
            return private
    # Read-only compatibility for old checkouts and isolated legacy fixtures.
    return root / "validation_gpu.json"


def resolve_trace_case(path_case: str, metadata: dict, registry: dict) -> tuple[str, str, str]:
    """The directory determines identity; registered aliases authenticate old metadata.

    Metadata and API transcripts are immutable historical artifacts. In migrated
    trees their old IDs remain valid aliases only for the registered dataset.
    Unregistered old fixtures preserve the former reader behavior.
    """
    original = metadata.get("case", path_case)
    dataset = metadata.get("dataset", "benchmark_fn_fp")
    details = registry.get("case_details", {})
    if not details:
        return original, original, dataset
    canonical = path_case
    detail = details.get(canonical)
    if detail is None:
        matches = [(name, row) for name, row in details.items()
                   if row.get("previous_id") == path_case and row.get("dataset") == dataset]
        if len(matches) == 1:
            canonical, detail = matches[0]
        elif path_case in registry.get("retired_ids", []):
            if original != path_case:
                raise ValueError(f"Retired trace case mismatch: {path_case} / {original}")
            return path_case, original, dataset
        else:
            raise ValueError(f"Trace case is absent from case_map.json: {path_case} ({dataset})")
    expected_dataset = detail["dataset"]
    if "dataset" in metadata and dataset != expected_dataset:
        raise ValueError(f"Trace dataset mismatch for {canonical}: {dataset} != {expected_dataset}")
    aliases = {canonical, detail.get("previous_id"), detail.get("source_name")}
    if original not in aliases:
        raise ValueError(f"Trace case mismatch for {canonical}: historical metadata names {original}")
    return canonical, original, expected_dataset
