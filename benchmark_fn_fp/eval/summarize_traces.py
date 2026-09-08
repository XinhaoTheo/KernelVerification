"""Score every arm from the traces on disk.

The scoreboard is derived, never authored. Each runner used to write its own
results_baselineN.json and overwrite it wholesale, so a batch of 14 cases
replaced a run of 32: results_baseline3.json ended up holding 14 of the 32
debate results, and the single-call arm was scattered across eight files that
only meant anything added together. A trace cannot be rebuilt from a summary,
but a summary can always be rebuilt from traces, so this reads traces/ and
writes the summary, and nothing else writes it.

Usage (from repo root):
    python benchmark_fn_fp/eval/summarize_traces.py
    python benchmark_fn_fp/eval/summarize_traces.py --json     # machine-readable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACES = REPO / "benchmark_fn_fp" / "traces"
SOURCE = REPO / "benchmark_fn_fp" / "triton"
MAP = REPO / "benchmark_fn_fp" / "case_map.json"
OUT = REPO / "benchmark_fn_fp" / "eval" / "scoreboard.json"

# Anthropic list price, USD per million tokens, for the model these runs used.
USD_IN, USD_OUT = 5.0, 25.0
CACHE_WRITE_MULT, CACHE_READ_MULT = 2.0, 0.1


def ground_truth() -> dict[str, str]:
    cases = json.loads(MAP.read_text())["cases"]
    out = {}
    for cid, real in cases.items():
        meta = json.loads((SOURCE / real / "meta.json").read_text())
        out[cid] = "reject" if meta["expected"]["correct_verdict"] == "BUGGY" else "trust"
    return out


def read_run(path: Path) -> dict:
    run = json.loads(path.read_text())
    usage = [t.get("usage") or {} for t in run.get("history", [])]

    def total(key: str) -> int:
        return sum(u.get(key) or 0 for u in usage)

    usd = (total("input_tokens") * USD_IN
           + total("output_tokens") * USD_OUT
           + total("cache_creation_input_tokens") * USD_IN * CACHE_WRITE_MULT
           + total("cache_read_input_tokens") * USD_IN * CACHE_READ_MULT) / 1e6
    verdict = run.get("verdict") or {}
    return {
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "turns": len(run.get("history", [])),
        "claims": len(run.get("claims") or []),
        "probes": sum(1 for e in run.get("tool_events") or []
                      if e.get("tool") == "run_claim_probe"),
        "usd": round(usd, 4),
    }


def main() -> int:
    gt = ground_truth()
    arms: dict[str, dict[str, dict]] = {}
    for run_json in sorted(TRACES.glob("*/*/run.json")):
        case, arm = run_json.parts[-3], run_json.parts[-2]
        arms.setdefault(arm, {})[case] = read_run(run_json)

    report: dict[str, object] = {"ground_truth": gt, "arms": {}}
    for arm in sorted(arms):
        rows = arms[arm]
        scored = {c: r for c, r in rows.items() if c in gt}
        correct = [c for c, r in scored.items() if r["verdict"] == gt[c]]
        fn = [c for c in scored if gt[c] == "reject"]
        fp = [c for c in scored if gt[c] == "trust"]
        report["arms"][arm] = {
            "cases": len(scored),
            "correct": len(correct),
            # A missed defect and a false alarm are different failures and are
            # never summed into one accuracy number without also being shown apart.
            "fn_correct": sum(1 for c in fn if scored[c]["verdict"] == gt[c]),
            "fn_total": len(fn),
            "fp_correct": sum(1 for c in fp if scored[c]["verdict"] == gt[c]),
            "fp_total": len(fp),
            "usd": round(sum(r["usd"] for r in scored.values()), 2),
            "wrong": sorted(c for c in scored if scored[c]["verdict"] != gt[c]),
            "per_case": scored,
        }

    if "--json" in sys.argv:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        names = sorted(arms)
        print(f"{'case':10s}{'truth':8s}" + "".join(f"{n:22s}" for n in names))
        for case in sorted(gt):
            line = f"{case:10s}{gt[case]:8s}"
            for n in names:
                r = arms[n].get(case)
                if not r:
                    line += f"{'-':22s}"
                else:
                    mark = "ok" if r["verdict"] == gt[case] else "WRONG"
                    line += f"{str(r['verdict']) + ' ' + str(r['confidence']) + ' ' + mark:22s}"
            print(line)
        print()
        for n in names:
            a = report["arms"][n]
            print(f"{n:10s} {a['correct']}/{a['cases']}   "
                  f"FN {a['fn_correct']}/{a['fn_total']}   "
                  f"FP {a['fp_correct']}/{a['fp_total']}   ${a['usd']}")
            if a["wrong"]:
                print(f"{'':10s} wrong: {', '.join(a['wrong'])}")

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
