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
