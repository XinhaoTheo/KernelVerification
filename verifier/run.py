"""Load a kernel artifact from the dataset and run debate on it.

This is the ONLINE / cheap stage. Generation (calling KernelAgent) happens
offline via `kv-build`. This script just reads an entry off disk, hands it
to author/skeptic/judge, and prints the verdict.

Usage:
    uv run kv-run                    # default entry: elem_add
    uv run kv-run softmax            # named entry from dataset/
    uv run kv-run --list             # list available entries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import dataset
from .debate import run_debate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?", default="elem_add", help="dataset entry name")
    ap.add_argument("--list", action="store_true", help="list entries and exit")
    ap.add_argument("--out", default=None, help="path to write debate transcript JSON")
    args = ap.parse_args()

    available = list(dataset.iter_entries())

    if args.list:
        if not available:
            print("(dataset is empty — run `uv run kv-build` first)")
            return 0
        for name in available:
            print(name)
        return 0

    try:
        artifact = dataset.load_entry(args.entry)
    except FileNotFoundError:
        print(
            f"entry '{args.entry}' not found. available: {available or '(empty)'}\n"
            f"run `uv run kv-build --problem {args.entry}` first.",
            file=sys.stderr,
        )
        return 2

    print(f"=== Loaded entry: {args.entry} ===")
    print(f"  passed (per dataset) = {artifact['passed']}")
    print(f"  session_dir          = {artifact['session_dir']}")
    print(f"  kernel_code          = {len(artifact['kernel_code'])} chars")
    print(f"  test_code            = {len(artifact['test_code'])} chars")

    if not artifact["passed"]:
        print("\nWarning: entry is marked as not passing the generator's own test.")
        print("Debate will still run — useful for testing skeptic recognition.")

    print("\n=== Running debate ===")
    verdict, history = run_debate(artifact)

    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))

    transcript_path = Path(args.out) if args.out else Path(artifact["session_dir"]) / "debate_history.json"
    transcript_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    print(f"\nFull transcript: {transcript_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
