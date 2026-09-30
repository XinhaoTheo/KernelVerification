"""Score every arm from the traces on disk.

The scoreboard is derived, never authored. Each runner used to write its own
results_baselineN.json and overwrite it wholesale, so a batch of 14 cases
replaced a run of 32: results_baseline3.json ended up holding 14 of the 32
debate results, and the single-call arm was scattered across eight files that
only meant anything added together. A trace cannot be rebuilt from a summary,
but a summary can always be rebuilt from traces, so this reads traces/ and
writes the summary, and nothing else writes it.

Usage (from repo root):
    python benchmark_fn_fp/eval_scripts/summarize_traces.py
    python benchmark_fn_fp/eval_scripts/summarize_traces.py --json     # machine-readable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Discovery handles shared GLM roots, provider leaves, and repeated trials.
BENCHMARK = REPO / "benchmark_fn_fp"
OUT = REPO / "benchmark_fn_fp" / "eval_scripts" / "scoreboard.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import PROFILES, profile_for, profile_for_trace  # noqa: E402
from case_registry import case_sort_key, validation_path  # noqa: E402


def profile_for_tree(traces_dir: str):
    """Legacy helper for a profile's full tree path, including provider leaves.

    New readers use per-run model metadata. A shared family root alone cannot
    identify a model or price and deliberately receives an unknown profile.
    """
    matches = [profile for profile in PROFILES.values() if profile.traces_dir == traces_dir]
    if len(matches) == 1:
        return matches[0]
    return profile_for(f"unprofiled:{traces_dir}")


def ground_truth(benchmark_dir: Path | None = None) -> dict[str, str]:
    root = Path(benchmark_dir) if benchmark_dir is not None else BENCHMARK
    cases = json.loads((root / "case_map.json").read_text())["cases"]
    out = {}
    for cid, real in cases.items():
        meta = json.loads((root / "triton" / real / "meta.json").read_text())
        out[cid] = "reject" if meta["expected"]["correct_verdict"] == "BUGGY" else "trust"
    return out


def read_run(path: Path, profile) -> dict:
    # Protocol parsing can fail before run.json is persisted. Raw calls still
    # document real provider usage, so an absent final state must not erase it.
    run = json.loads(path.read_text()) if path.exists() else {}
    history_usage = [t["usage"] for t in run.get("history", []) if t.get("usage")]
    raw_usage = []
    calls = []
    raw_root = path.parent / "llm_calls"
    for directory in sorted(raw_root.iterdir()) if raw_root.exists() else []:
        if not directory.is_dir():
            continue
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        response_path = directory / "response.json"
        response = json.loads(response_path.read_text()) if response_path.exists() else {}
        usage_record = response.get("usage") or {}
        has_usage = all(usage_record.get(key) is not None
                        for key in ("prompt_tokens", "completion_tokens"))
        if has_usage:
            raw_usage.append({"input_tokens": usage_record["prompt_tokens"] or 0,
                              "output_tokens": usage_record["completion_tokens"] or 0})
        calls.append({"call": directory.name, "status": metadata.get("status"),
                      "response_saved": response_path.exists(), "usage_saved": has_usage,
                      "finish_reasons": [choice.get("finish_reason")
                                         for choice in response.get("choices", [])]})
    responses_saved = sum(call["response_saved"] for call in calls)
    use_raw = responses_saved > 0
    usage = raw_usage if use_raw else history_usage
    coverage_status = ("partial" if calls and (
        len(raw_usage) != len(calls) or len(raw_usage) < len(history_usage))
        else "complete" if calls else "legacy_history")
    coverage = {"status": coverage_status, "captured_calls": len(calls),
                "responses_saved": responses_saved, "responses_with_usage": len(raw_usage),
                "history_calls_with_usage": len(history_usage),
                "missing_response_calls": [call["call"] for call in calls if not call["response_saved"]],
                "missing_usage_calls": [call["call"] for call in calls
                                        if call["response_saved"] and not call["usage_saved"]],
                "note": ("Coverage describes captured SDK calls; unrecorded responses or SDK retries "
                         "may have additional charges. Legacy history does not certify complete API capture.")}

    def total(key: str) -> int:
        return sum(u.get(key) or 0 for u in usage)

    usd = (total("input_tokens") * profile.price_in
           + total("output_tokens") * profile.price_out
           + total("cache_creation_input_tokens") * profile.price_in * profile.cache_write
           + total("cache_read_input_tokens") * profile.price_in * profile.cache_read) / 1e6
    verdict_path = path.parent / "verdict.json"
    verdict = run.get("verdict") or (
        json.loads(verdict_path.read_text()) if verdict_path.exists() else {})
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
        "usd": round(usd, 6) if profile.known and (usage or not calls) else None,
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "usage_source": "raw_chat_responses" if use_raw else "history",
        "usage_coverage": coverage,
        "usd_is_partial": coverage_status == "partial",
        "call_finish_reasons": calls,
    }


def read_single_call(run_dir: Path, profile) -> dict:
    """Read both legacy flat usage and newer raw-provider single-call traces."""
    def read(name: str) -> dict:
        path = run_dir / name
        return json.loads(path.read_text()) if path.exists() else {}

    usage_record = read("usage.json")
    raw = read("raw_response.json")
    # A failure after receiving the API response can precede the derived files.
    if not raw:
        responses = sorted((run_dir / "llm_calls").glob("*/response.json"))
        if responses:
            raw = json.loads(responses[0].read_text())
    choice = (raw.get("choices") or [{}])[0]
    verdict = read("verdict.json") or usage_record.get("response") or {}
    if not verdict:
        response_path = run_dir / "response_text.json"
        text = (response_path.read_text() if response_path.exists()
                else (choice.get("message") or {}).get("content") or "")
        try:
            candidate = json.loads(text)
            verdict = candidate if isinstance(candidate, dict) else {}
        except (ValueError, TypeError):
            verdict = {}
    usage = usage_record.get("usage") or raw.get("usage") or usage_record
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    usd = (input_tokens * profile.price_in + output_tokens * profile.price_out
           + cache_write * profile.price_in * profile.cache_write
           + cache_read * profile.price_in * profile.cache_read) / 1e6
    row = {
        "model": profile.model,
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "reason": verdict.get("reason"),
        "stop_reason": usage_record.get("stop_reason") or choice.get("finish_reason"),
        "turns": 1 if raw or usage_record else 0,
        "claims": 0,
        "probes": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(usd, 6) if profile.known and usage else None,
    }
    if (run_dir / "llm_calls").exists():
        api_usage = read_run(run_dir / "run.json", profile)
        for key in ("input_tokens", "output_tokens", "usd", "usage_source",
                    "usage_coverage", "usd_is_partial", "call_finish_reasons"):
            row[key] = api_usage[key]
    return row


def classify_outcome(row: dict, truth: str | None) -> str:
    """Separate unanswered or exhausted attempts from explicit wrong verdicts."""
    if row.get("stop_reason") in {"length", "max_tokens", "max_output_tokens"}:
        return "token_limit"
    if row.get("verdict") not in {"trust", "reject", "needs_more_evidence"} and any(
        reason in {"length", "max_tokens", "max_output_tokens"}
        for call in row.get("call_finish_reasons", []) for reason in call["finish_reasons"]
    ):
        return "token_limit"
    if row.get("verdict") == "needs_more_evidence":
        return "abstention"
    if row.get("verdict") not in {"trust", "reject"}:
        return "no_verdict"
    if truth is None:
        return "unscored"
    return "correct" if row["verdict"] == truth else "wrong_verdict"


def build_report(*, records=None, labels: dict[str, dict[str, str]] | None = None,
                 benchmark_dir: Path | None = None) -> dict:
    """Group complete attempts without conflating datasets, trials or endpoints."""
    root = Path(benchmark_dir) if benchmark_dir is not None else BENCHMARK
    if records is None:
        from traces import iter_trace_records
        records = iter_trace_records(benchmark_dir=root)
    if labels is None:
        labels = {"benchmark_fn_fp": ground_truth(root)}
        for dataset in ("correlation_pair", "numerical_challenges", "evidence_challenges", "numerical_pilot"):
            validated_labels = validation_path(root, dataset)
            if validated_labels.exists():
                labels[dataset] = {
                    case: row["ground_truth"]
                    for case, row in json.loads(validated_labels.read_text())["cases"].items()
                }
    groups = {}
    for record in records:
        run_dir = Path(record["path"])
        model = record.get("model")
        profile = profile_for_trace(model or "unprofiled:missing-model", record.get("metadata") or {})
        if (run_dir / "run.json").exists() or (
            record["arm"] != "single_call" and (run_dir / "llm_calls").exists()
        ):
            row = read_run(run_dir / "run.json", profile)
        elif record["arm"] == "single_call":
            row = read_single_call(run_dir, profile)
        else:
            row = {"verdict": None, "confidence": None, "usd": None,
                   "turns": 0, "claims": 0, "probes": 0}
        # Model identity is metadata, never a guess based on a common GLM family.
        row.update({key: record.get(key) for key in
                    ("case", "original_case", "arm", "trial", "dataset", "provider", "model")})
        row["path"] = str(run_dir.relative_to(root)) if run_dir.is_relative_to(root) else str(run_dir)
        truth = labels.get(record["dataset"], {}).get(record["case"])
        row["truth"] = truth
        row["outcome"] = classify_outcome(row, truth)
        row["correct"] = row["outcome"] == "correct" if truth is not None else None
        error_path = run_dir / "runner_error.txt"
        if error_path.exists() and error_path.read_text().strip():
            row["runner_error"] = error_path.read_text().strip()
        key = "/".join(str(record.get(part) or "unknown") for part in
                       ("dataset", "provider", "model", "arm", "trial"))
        group = groups.setdefault(key, {
            **{part: record.get(part) for part in ("dataset", "provider", "model", "arm", "trial")},
            "per_case": {},
        })
        if record["case"] in group["per_case"]:
            raise ValueError(f"Duplicate case/arm/model/provider/dataset/trial: {key}/{record['case']}")
        group["per_case"][record["case"]] = row

    for group in groups.values():
        rows = group["per_case"]
        scored = {case: row for case, row in rows.items() if row["truth"] is not None}
        fn = [row for row in scored.values() if row["truth"] == "reject"]
        fp = [row for row in scored.values() if row["truth"] == "trust"]
        priced = [row["usd"] for row in rows.values() if row["usd"] is not None]
        group.update({
            "attempts": len(rows), "cases": len(scored),
            "correct": sum(row["outcome"] == "correct" for row in scored.values()),
            "fn_correct": sum(row["outcome"] == "correct" for row in fn), "fn_total": len(fn),
            "fp_correct": sum(row["outcome"] == "correct" for row in fp), "fp_total": len(fp),
            "usd": round(sum(priced), 6) if priced else None,
            "unpriced": sum(row["usd"] is None for row in rows.values()),
            "partial_cost_attempts": sum(bool(row.get("usd_is_partial")) for row in rows.values()),
            "wrong": sorted((case for case, row in scored.items() if row["outcome"] == "wrong_verdict"), key=case_sort_key),
            "outcomes": {outcome: sum(row["outcome"] == outcome for row in rows.values())
                         for outcome in ("correct", "wrong_verdict", "abstention", "token_limit", "no_verdict", "unscored")},
        })
    return {"ground_truth": labels.get("benchmark_fn_fp", {}),
            "ground_truth_by_dataset": labels, "arms": dict(sorted(groups.items()))}


def main() -> int:
    report = build_report()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for name, group in report["arms"].items():
            cost = "unknown" if group["usd"] is None else f"${group['usd']:.4f}"
            print(f"{name}: {group['correct']}/{group['cases']}  "
                  f"FN {group['fn_correct']}/{group['fn_total']}  "
                  f"FP {group['fp_correct']}/{group['fp_total']}  {cost}")
            print("  outcomes: " + json.dumps(group["outcomes"]))
            if group["wrong"]:
                print("  wrong verdicts: " + ", ".join(group["wrong"]))
            if group["unpriced"]:
                print(f"  unpriced attempts: {group['unpriced']}")
            if group["partial_cost_attempts"]:
                print(f"  incomplete API usage coverage: {group['partial_cost_attempts']} attempt(s); "
                      "reported cost excludes unrecorded usage")
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
