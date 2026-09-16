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
# One tree per model: traces_opus5/, traces_glm/, ...
BENCHMARK = REPO / "benchmark_fn_fp"
SOURCE = REPO / "benchmark_fn_fp" / "triton"
MAP = REPO / "benchmark_fn_fp" / "case_map.json"
OUT = REPO / "benchmark_fn_fp" / "eval" / "scoreboard.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import PROFILES, label_for_traces_dir, profile_for  # noqa: E402


def profile_for_tree(traces_dir: str):
    """The model that produced a trace tree, from the directory name alone.

    Each model writes to its own tree (models.traces_dir_for), so the same table
    that set a run's max_tokens also prices it. Charging a $0.075/M open model at
    Opus rates would overstate its cost seventy-fold.
    """
    for profile in PROFILES.values():
        if profile.traces_dir == traces_dir:
            return profile
    return profile_for(f"unprofiled:{traces_dir}")


def ground_truth() -> dict[str, str]:
    cases = json.loads(MAP.read_text())["cases"]
    out = {}
    for cid, real in cases.items():
        meta = json.loads((SOURCE / real / "meta.json").read_text())
        out[cid] = "reject" if meta["expected"]["correct_verdict"] == "BUGGY" else "trust"
    return out


def read_run(path: Path, profile) -> dict:
    run = json.loads(path.read_text())
    usage = [t.get("usage") or {} for t in run.get("history", [])]

    def total(key: str) -> int:
        return sum(u.get(key) or 0 for u in usage)

    usd = (total("input_tokens") * profile.price_in
           + total("output_tokens") * profile.price_out
           + total("cache_creation_input_tokens") * profile.price_in * profile.cache_write
           + total("cache_read_input_tokens") * profile.price_in * profile.cache_read) / 1e6
    verdict = run.get("verdict") or {}
    return {
        # None, not 0.0, for a model with no profile: a run whose price is
        # unknown must not be summed into a total as if it were free.
        "model": profile.model or None,
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "turns": len(run.get("history", [])),
        "claims": len(run.get("claims") or []),
        "probes": sum(1 for e in run.get("tool_events") or []
                      if e.get("tool") == "run_claim_probe"),
        "usd": round(usd, 4) if profile.known else None,
    }


def main() -> int:
    gt = ground_truth()
    arms: dict[str, dict[str, dict]] = {}
    for run_json in sorted(BENCHMARK.glob("traces_*/*/*/run.json")):
        tree, case, arm = run_json.parts[-4], run_json.parts[-3], run_json.parts[-2]
        # `opus5/solo`, `glm/debate`: the model is part of the arm's name, so two
        # models' results for one case sit on separate rows and never merge.
        name = f"{label_for_traces_dir(tree)}/{arm}"
        arms.setdefault(name, {})[case] = read_run(run_json, profile_for_tree(tree))

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
            "usd": round(sum(r["usd"] or 0.0 for r in scored.values()), 2),
            "unpriced": sum(1 for r in scored.values() if r["usd"] is None),
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
            note = f"  ({a['unpriced']} unpriced)" if a["unpriced"] else ""
            print(f"{n:16s} {a['correct']}/{a['cases']}   "
                  f"FN {a['fn_correct']}/{a['fn_total']}   "
                  f"FP {a['fp_correct']}/{a['fp_total']}   ${a['usd']}{note}")
            if a["wrong"]:
                print(f"{'':16s} wrong: {', '.join(a['wrong'])}")

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
