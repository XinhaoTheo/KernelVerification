"""Read-only scoring of the prospective three-arm evidence pilot.
Update only the PILOT RESULTS block in solo_vs_debate_challenges/README.md.
Historical dataset IDs and source hashes are read unchanged; no model calls.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "solo_vs_debate_challenges"
sys.path.insert(0, str(ROOT.parent / "eval_scripts"))
from summarize_traces import build_report
from generated_results import compact_results, replace_results
from traces import iter_trace_records, experiment_trial, trial_sort_key

CASES = tuple(f"case_{i + 61:02d}" for i in range(1, 5))
ARMS = ("single_call", "solo", "debate")
TRIALS = ("ea_pilot_r1", "ea_pilot_r2")
MODEL = "accounts/fireworks/models/glm-5p3"
CAP = 32768
TERMINAL_STATUSES = {"completed", "error", "failed", "cancelled", "timeout"}
LENGTH_STOPS = {"length", "max_tokens", "max_output_tokens"}


def read(path):
    return json.loads(path.read_text())


def audit_row(row, metadata, path, frozen):
    """Validate raw evidence independently of the common summary's usage flag."""
    row = dict(row)
    row["status"] = metadata.get("status")
    issues = row["issues"] = []
    if row.get("model") != MODEL or row.get("provider") != "fireworks":
        issues.append("model/provider differs")
    label = frozen.get(row["case"], {})
    for key in ("kernel_sha256", "problem_sha256"):
        if not label.get(key) or metadata.get(key) != label[key]:
            issues.append(f"source mismatch: {key}")
    if row["arm"] != "single_call" and metadata.get("total_output_token_budget") != CAP:
        issues.append("shared budget differs")
    used, accounting_known = 0, True
    calls = sorted(p for p in (path / "llm_calls").glob("*") if p.is_dir())
    final_finish_reasons = []
    if not calls:
        issues.append("no raw API calls captured")
    if row["arm"] == "single_call" and len(calls) != 1:
        issues.append("single-call arm did not capture exactly one API call")
    for call in calls:
        request_path, response_path = call / "request.json", call / "response.json"
        if not request_path.exists():
            issues.append(f"missing request: {call.name}")
        else:
            request = read(request_path)
            if (request.get("model") != MODEL or request.get("reasoning_effort") != "low"
                    or not accounting_known or used >= CAP
                    or request.get("max_tokens") != CAP - used):
                issues.append(f"request setting/budget mismatch: {call.name}")
        metadata_path = call / "metadata.json"
        call_meta = read(metadata_path) if metadata_path.exists() else {}
        if call_meta.get("status") != "completed" or call_meta.get("response_saved") is not True:
            issues.append(f"raw call did not finish completely: {call.name}")
        if not response_path.exists():
            issues.append(f"missing response: {call.name}")
            accounting_known = False
            final_finish_reasons = []
            continue
        response = read(response_path)
        usage = response.get("usage") or {}
        valid_usage = all(isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool)
                          and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens"))
        if valid_usage:
            used += usage["completion_tokens"]
        else:
            issues.append(f"missing/invalid usage: {call.name}")
            accounting_known = False
        final_finish_reasons = [choice.get("finish_reason") for choice in response.get("choices", [])]
        if not final_finish_reasons or any(reason is None for reason in final_finish_reasons):
            issues.append(f"missing finish reason: {call.name}")
    row["recorded_output_budget_used"] = used
    if used > CAP:
        issues.append("cumulative output cap exceeded")
    coverage = row.get("usage_coverage", {})
    if (coverage.get("status") != "complete" or coverage.get("captured_calls") != len(calls)
            or coverage.get("responses_saved") != len(calls)
            or coverage.get("responses_with_usage") != len(calls)):
        issues.append("incomplete raw usage coverage")

    error = row.get("runner_error") or ""
    if row["status"] not in TERMINAL_STATUSES:
        row["outcome"] = "pending"
    elif "OutputTokenBudgetAccountingError" in error:
        row["outcome"] = "budget_accounting_error"
    elif "OutputTokenBudgetExhausted" in error:
        row["outcome"] = "budget_exhaustion"
    elif row["status"] == "cancelled":
        row["outcome"] = "cancelled"
    elif row["status"] == "timeout":
        row["outcome"] = "timeout"
    elif any(kind in error for kind in ("InternalServerError", "APIConnectionError", "APITimeoutError",
                                       "RateLimitError", "AuthenticationError", "PermissionDeniedError")):
        row["outcome"] = "transport_error"
    elif row["status"] in {"error", "failed"} or error:
        row["outcome"] = "runner_error"
    elif any(reason in LENGTH_STOPS for reason in final_finish_reasons):
        # A partial final JSON verdict must not become an explicit mistake.
        row["outcome"] = "token_limit"
    row["valid_complete"] = bool(row["status"] == "completed" and not issues and not error)
    row["correct"] = row["outcome"] == "correct" if row.get("truth") is not None else None
    row["trace_link"] = "../" + row["path"] + "/" + (
        "transcript.md" if (path / "transcript.md").exists() else "trace_meta.json")
    return row


def pilot_gate(rows, *, cases=CASES, trials=TRIALS):
    """Only the predeclared case/trial slots can activate the two-stage gate."""
    slots = {(experiment_trial(r), r["case"], r["arm"]): r for r in rows}
    if len(slots) != len(rows):
        raise ValueError("Duplicate designated slot")
    comparisons = []
    for trial in trials:
        for case in cases:
            solo, debate = (slots.get((trial, case, arm), {}) for arm in ("solo", "debate"))
            usable = solo.get("valid_complete") and debate.get("valid_complete")
            comparisons.append({"trial": trial, "case": case,
                "debate_corrects_solo": bool(usable and solo["outcome"] == "wrong_verdict" and debate["outcome"] == "correct"),
                "solo_corrects_debate": bool(usable and debate["outcome"] == "wrong_verdict" and solo["outcome"] == "correct")})

    def terminal(trials):
        return all((t, c, a) in slots and slots[(t, c, a)].get("status") in TERMINAL_STATUSES
                   and slots[(t, c, a)]["outcome"] != "pending"
                   for t in trials for c in cases for a in ARMS)

    r1 = terminal(trials[:1]) and any(x["debate_corrects_solo"] for x in comparisons if x["trial"] == trials[0])
    replicated = [case for case in cases if all(any(x["trial"] == t and x["case"] == case
        and x["debate_corrects_solo"] for x in comparisons) for t in trials)]
    net = sum(x["debate_corrects_solo"] - x["solo_corrects_debate"] for x in comparisons)
    full = terminal(trials)
    gate = {"r1_activates_r2": r1, "replicated_cases": replicated,
            "net_paired_improvements": net, "all_two_round_slots_terminal": full,
            "expansion_numerical_gate": bool(full and replicated and net > 0),
            "additional_requirement": "Audit actual independent GPU evidence before expansion."}
    return comparisons, gate


def derive(*, cases=CASES, trials=TRIALS):
    frozen = {case: row for case, row in read(ROOT / "private_data" / "validation_gpu.json")["cases"].items()
              if case in cases}
    for case, label in frozen.items():
        for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt")):
            actual = hashlib.sha256((ROOT.parent / "triton_eval_cases" / case / filename).read_bytes()).hexdigest()
            if label[f"{kind}_sha256"] != actual:
                raise ValueError(f"Frozen source changed: {case}/{kind}")
    records = [r for r in iter_trace_records(benchmark_dir=ROOT.parent)
               if r["dataset"] == "evidence_challenges" and r["case"] in cases]
    common = build_report(records=records, benchmark_dir=ROOT.parent, labels={
        "evidence_challenges": {case: row["ground_truth"] for case, row in frozen.items()}})
    meta = {str(Path(r["path"]).resolve()): r["metadata"] for r in records}
    rows = []
    for group in common["arms"].values():
        for row in group["per_case"].values():
            path = ROOT.parent / row["path"]
            m = meta[str(path.resolve())]
            rows.append(audit_row(row, m, path, frozen))
    rows.sort(key=lambda r: (r["case"], r["arm"], trial_sort_key(r["trial"])))
    comparisons, gate = pilot_gate(rows, cases=cases, trials=trials)
    groups = []
    audited_by_path = {row["path"]: row for row in rows}
    for group in common["arms"].values():
        selected = [audited_by_path[row["path"]] for row in group["per_case"].values()]
        groups.append({"trial": group["original_trial"], "original_trial": group["original_trial"],
            "arm": group["arm"], "provider": group["provider"], "model": group["model"],
            "protocol": group["protocol"], "attempts": len(selected),
            "outcomes": dict(Counter(r["outcome"] for r in selected)),
            "api_estimate_usd": round(sum(r.get("usd") or 0 for r in selected), 6)})
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "cases": frozen,
            "rows": rows, "groups": groups, "comparisons": comparisons, "gate": gate,
            "api_estimate_usd": round(sum(r.get("usd") or 0 for r in rows), 6),
            "unknown_cost_attempts": sum(r.get("usd") is None for r in rows),
            "partial_cost_attempts": sum(bool(r.get("usd_is_partial")) for r in rows)}


def render(result):
    """Summarize outcomes while retaining all rows in the optional JSON export."""
    gate = result["gate"]
    return "\n".join([
        f"生成时间：{result['generated_at']}", "", compact_results(result["rows"]),
        f"指定窗口扩展门槛： **{'通过' if gate['expansion_numerical_gate'] else '未达到'}**; "
        f"复验中重复纠错的题目： {', '.join(gate['replicated_cases']) or 'none'}; "
        f"配对明确错误净改善： {gate['net_paired_improvements']}.", "",
        "纠正明确错误与解决弃答分别统计。达到数值门槛后仍需独立审核关键运行证据；该报告不会启动新实验。",
    ]) + "\n"


def main(*, export_json=False):
    result = derive()
    replace_results(ROOT, "PILOT RESULTS", render(result))
    if export_json:
        output = ROOT / "private_data" / "reports" / "scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"groups": result["groups"], "gate": result["gate"],
                      "api_estimate_usd": result["api_estimate_usd"]}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    main(export_json=parser.parse_args().json)
