"""Resolve historical source names to the single registered numeric case ID."""
import json
from pathlib import Path


def source_case_path(root, source_name):
    registry = Path(__file__).resolve().parents[2] / "case_map.json"
    details = json.loads(registry.read_text())["case_details"]
    matches = [case for case, row in details.items()
               if row["dataset"] == "benchmark_fn_fp"
               and source_name in {case, row.get("source_name")}]
    if len(matches) != 1:
        raise ValueError(
            f"Source {source_name!r} is unregistered or retired; register a new "
            "global case ID before generating it. Retired IDs must not be reused."
        )
    return Path(root) / matches[0]
