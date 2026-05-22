"""Verify one kernel end-to-end: recheck (our own test) -> debate -> verdict.

This is the VERIFICATION entry point. It does NOT trust whatever test the
upstream generator may have produced (a different generator might produce none
at all). Instead it is self-contained:

    load kernel  ->  generate OUR allclose test + run it (recheck)  ->  debate

The recheck result (pass/fail + error) is what the debate agents see as the
authoritative correctness signal. recheck results are cached on the entry, so
iterating on debate prompts does NOT re-pay for test generation unless you pass
--force-recheck.

Usage:
    uv run kv-run                    # default entry: elem_add
    uv run kv-run softmax            # named entry from dataset/
    uv run kv-run --list             # list available entries
    uv run kv-run cumsum --force-recheck   # regenerate + rerun our test
"""

from __future__ import annotations

# recheck runs the kernel on GPU, so pin a usable device before torch loads.
from .gpu_pick import pin_freest_gpu

pin_freest_gpu()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from . import dataset, recheck  # noqa: E402
from .debate import run_debate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?", default="elem_add", help="dataset entry name")
    ap.add_argument("--list", action="store_true", help="list entries and exit")
    ap.add_argument(
        "--force-recheck",
        action="store_true",
        help="regenerate + rerun our test even if a cached recheck exists",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="stream the whole flow: recheck test+output, each agent's full text, verifier probes",
    )
    ap.add_argument("--out", default=None, help="path to write debate result JSON")
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
    print(f"  kernel_code = {len(artifact['kernel_code'])} chars")

    # Self-contained verification: our own test, not the generator's.
    print("\n=== Recheck (our independent allclose test) ===")
    rc = recheck.get_recheck(args.entry, force=args.force_recheck)
    print(f"  recheck status = {rc['status']}  (rtol={rc['rtol']}, atol={rc['atol']})")

    if args.verbose:
        bar = "─" * 72
        print(f"\n{bar}\n▶ RECHECK TEST (LLM-generated, our ground truth)\n{bar}")
        print(rc["test_code"].strip())
        print(f"\n{bar}\n▶ RECHECK OUTPUT\n{bar}")
        print(rc["output_text"].strip())

    # Fold OUR verification result into the artifact the debate sees, so author /
    # skeptic / judge reason about the kernel against the test WE trust.
    artifact["passed"] = rc["status"] == "passed"
    artifact["status"] = rc["status"]
    if rc["test_code"]:
        artifact["test_code"] = rc["test_code"]
    artifact["error"] = {"text": rc["error_text"]} if rc["error_text"] else {}

    print("\n=== Running debate ===")
    verdict, history, claims = run_debate(artifact, verbose=args.verbose)

    print("\n=== Claims ledger ===")
    for c in claims:
        print(f"  [{c.get('id')}] {c.get('status'):<12s} {c.get('statement', '')[:80]}")

    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))

    transcript_path = (
        Path(args.out)
        if args.out
        else Path(artifact["session_dir"]) / "debate_result.json"
    )
    transcript_path.write_text(
        json.dumps(
            {
                "entry": args.entry,
                "recheck_status": rc["status"],
                "verdict": verdict,
                "claims": claims,
                "history": history,
                "rounds": len(history) // 3,  # author + skeptic + verifier per round
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nFull result: {transcript_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
