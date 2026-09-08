"""Check every captured trace for the failure modes found by reading them.

Each check here exists because the defect it looks for was already shipped once
and was only caught by reading a trace line by line. Reading does not scale to
32 cases times three arms, so the checks run themselves.

Usage (from repo root):
    python benchmark_fn_fp/eval/audit_traces.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACES = REPO / "benchmark_fn_fp" / "traces"
TOOL_EVENT_WINDOW = 12  # must match base.py's state.tool_events[-12:]


def audit(run_dir: Path) -> list[str]:
    problems: list[str] = []
    run = json.loads((run_dir / "run.json").read_text())
    events = [json.loads(l) for l in (run_dir / "tool_events.jsonl").read_text().splitlines() if l.strip()]

    # 1. A tool call that failed cost a model call and produced nothing.
    for e in events:
        if e.get("status") == "error":
            problems.append(f"tool error: {e['tool']} -- {(e.get('output') or {}).get('message','')[:90]}")

    # 2. The contract must be present in the rendered state, not only as a tool
    #    event that scrolls out of the window. The Judge decides whether the
    #    kernel violates it; on case_33 it once ruled without it in context.
    artifact = run.get("artifact") or {}
    if not (artifact.get("problem_text") or "").strip():
        problems.append("artifact.problem_text is empty: the contract is not pinned into the prompt")

    # 3. "Nobody ran the tests" must not reach the prompt as "the tests failed".
    #    bool(None) is False, so every case once told every agent the kernel had
    #    failed; a Judge cited it while rejecting a trust case.
    if artifact.get("passed") is False and artifact.get("status") == "under_test":
        problems.append("artifact.passed is False on an under_test case: unknown rendered as failure")

    # 4. A turn that called no tool spent a model call on nothing. Turns that
    #    only think are how the run silently burns budget.
    for i, t in enumerate(run.get("history") or []):
        role = t.get("role")
        if role == "orchestrator":
            continue
        if not (t.get("tool_calls") or []):
            out = (t.get("usage") or {}).get("output_tokens") or 0
            problems.append(f"turn {i+1} ({role}) called no tool, output_tokens={out}")

    # 5. Re-reading material the prompt already carries is the waste that the
    #    preload fix was for; it should not come back.
    preloaded = {"load_artifact", "inspect_problem", "list_artifact_files"}
    for i, t in enumerate(run.get("history") or []):
        if t.get("role") == "orchestrator":
            continue
        for c in t.get("tool_calls") or []:
            if c.get("tool") in preloaded:
                problems.append(f"turn {i+1} ({t.get('role')}) re-fetched {c.get('tool')}, already in the prompt")

    # 6. A run that ends without a verdict answered nothing.
    if not (run.get("verdict") or {}).get("verdict"):
        problems.append("no verdict recorded")

    # 7. Every claim must end settled; an open claim at the end was never
    #    accounted for.
    for c in run.get("claims") or []:
        if c.get("status") in (None, "open"):
            problems.append(f"claim {c.get('id')} left open")

    # 8. A confirmed claim that supports a reject must carry scope evidence.
    verdict = run.get("verdict") or {}
    if verdict.get("verdict") == "reject":
        decisive = set(verdict.get("decisive_claims") or [])
        for c in run.get("claims") or []:
            if c.get("id") in decisive and c.get("scope") != "in_scope":
                problems.append(f"reject rests on {c.get('id')} with scope={c.get('scope')}")

    return problems


def main() -> int:
    dirs = sorted(p.parent for p in TRACES.glob("*/*/run.json"))
    if not dirs:
        print(f"no traces under {TRACES}", file=sys.stderr)
        return 1
    total = 0
    for d in dirs:
        name = f"{d.parent.name}/{d.name}"
        found = audit(d)
        total += len(found)
        if found:
            print(f"### {name}: {len(found)} issue(s)")
            for p in found:
                print(f"    - {p}")
        else:
            print(f"### {name}: clean")
    print(f"\n{total} issue(s) across {len(dirs)} trace(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
