"""One-shot: backfill `status` / `rounds` on pre-patch dataset entries.

Old entries' meta.json lacked these fields (they were built before the
failure-aware patch). This walks dataset/ and adds them where missing,
inferring from existing fields:
    passed=True  -> status="passed",  rounds = meta.raw.rounds (if any)
    passed=False -> status="unknown_failure" (we have no diagnostics for these
                    legacy entries; rebuild to get real status/error.txt)

Idempotent: entries that already have `status` are skipped.

Usage:
    uv run python tests/migrate_dataset.py            # dry-run, prints diff
    uv run python tests/migrate_dataset.py --write    # actually writes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verifier.dataset import DEFAULT_DATASET_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--write", action="store_true", help="Actually write changes (default: dry-run)")
    args = ap.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print(f"dataset dir does not exist: {root}", file=sys.stderr)
        return 2

    changed = 0
    skipped = 0
    for entry_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = entry_dir / "meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        if "status" in meta and "rounds" in meta:
            skipped += 1
            continue

        passed = bool(meta.get("passed", False))
        raw = meta.get("raw") or {}
        if passed:
            new_status = "passed"
            new_rounds = raw.get("rounds")
        else:
            new_status = "unknown_failure"
            new_rounds = raw.get("rounds")

        meta.setdefault("status", new_status)
        meta.setdefault("rounds", new_rounds)
        meta.setdefault("error_stderr_len", 0)
        meta.setdefault("error_stdout_len", 0)

        print(f"  {entry_dir.name}: status={new_status} rounds={new_rounds}")
        changed += 1

        if args.write:
            meta_path.write_text(json.dumps(meta, indent=2, default=str))

    print(f"\nchanged: {changed}   already-current: {skipped}")
    if not args.write and changed:
        print("(dry-run — pass --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
