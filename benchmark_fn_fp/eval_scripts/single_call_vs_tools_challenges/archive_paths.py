"""Paths for the retired input-specific authoring and reporting programs."""
from pathlib import Path
import os


BENCHMARK = Path(__file__).resolve().parents[2]
ARCHIVE = BENCHMARK / "archive" / "input_specific"
COLLECTIONS = {"single_call_vs_tools_challenges", "solo_vs_debate_challenges"}


def archived_collection(root):
    """Redirect old project defaults; preserve explicit external test fixtures."""
    root = Path(root)
    if root.name in COLLECTIONS and root.parent.resolve() == BENCHMARK:
        return ARCHIVE / root.name
    return root


def benchmark_root(collection):
    """Reports use the central registry even when their data are archived."""
    root = Path(collection)
    if root.parent.name == "input_specific" and root.parent.parent.name == "archive":
        return root.parents[2]
    return root.parent


def trace_link(collection, path, filename):
    return Path(os.path.relpath(Path(path) / filename, collection)).as_posix()


def canonical_case_rows(collection, dataset, rows):
    """Read historical answer keys through the registry without rewriting them."""
    from benchmark_fn_fp.eval_scripts.case_registry import load_registry, remap_validation_cases

    registry = load_registry(benchmark_root(collection))
    mapped = remap_validation_cases(rows, dataset, registry)
    if len(mapped) != len(rows):
        raise ValueError(f"Unresolved or ambiguous frozen case IDs for {dataset}")
    return mapped


def frozen_case_id(collection, dataset, case):
    """Keep names of frozen JSON keys and NPZ files separate from public paths."""
    from benchmark_fn_fp.eval_scripts.case_registry import canonical_dataset, load_registry

    details = load_registry(benchmark_root(collection)).get("case_details", {})
    if not details:
        return case
    detail = details.get(case, {})
    if detail.get("dataset") != canonical_dataset(dataset):
        raise ValueError(f"Unregistered case or dataset mismatch: {case}/{dataset}")
    return detail.get("validation_case_id", case)


def frozen_case_rows(collection, dataset, rows):
    """Serialize new validation rows under the exact registered evidence keys."""
    output = {frozen_case_id(collection, dataset, case): row for case, row in rows.items()}
    if len(output) != len(rows):
        raise ValueError(f"Ambiguous frozen case ownership for {dataset}")
    return output
