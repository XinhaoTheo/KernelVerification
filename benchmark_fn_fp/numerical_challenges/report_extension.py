"""Derive the I--N extension report from canonical traces, without API calls.

Writes EXTENSION_REPORT.md; --json also exports a scoreboard under private_data/reports/. Existing cases,
traces, protocols and the earlier C--H report are read-only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "eval_scripts"))
from summarize_traces import build_report
from traces import iter_trace_records, experiment_trial, trial_sort_key

CASES = tuple(f"case_{i:02d}" for i in range(44, 50))
ARMS = ("single_call", "solo", "debate")
MODEL = "accounts/fireworks/models/glm-5p3"
DATASET = "numerical_challenges"
LOW_TRIALS = ("extension_low32_r1", "extension_low32_r2", "extension_low32_r3")
PLANNED = {
    LOW_TRIALS[0]: {"reasoning_effort": "low", "max_tokens": 32768, "arms": ARMS},
    LOW_TRIALS[1]: {"reasoning_effort": "low", "max_tokens": 32768, "arms": ARMS},
    LOW_TRIALS[2]: {"reasoning_effort": "low", "max_tokens": 32768, "arms": ("single_call",)},
    "extension_default64_r1": {"reasoning_effort": "default", "max_tokens": 65536,
                               "arms": ("single_call",)},
}
OUTCOMES = ("correct", "wrong_verdict", "abstention", "token_limit", "no_verdict", "unscored", "pending")
TERMINAL = {"completed", "error", "failed", "aborted", "cancelled", "timeout"}


def _read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _counts(rows):
    counts = Counter(row["outcome"] for row in rows)
    return {name: counts[name] for name in OUTCOMES}


def _cost(rows):
    priced = [row for row in rows if row.get("usd") is not None]
    return {
        "recorded_api_estimate_usd": round(sum(row["usd"] for row in priced), 6) if priced else None,
        "unpriced_attempts": sum(row.get("usd") is None for row in rows),
        "partial_cost_attempts": sum(bool(row.get("usd_is_partial")) for row in rows),
        "input_tokens_recorded": sum(row.get("input_tokens", 0) for row in rows),
        "output_tokens_recorded": sum(row.get("output_tokens", 0) for row in rows),
        "raw_coverage_counts": dict(Counter(row["raw_coverage"] for row in rows)),
        "captured_calls": sum(row.get("usage_coverage", {}).get("captured_calls", 0) for row in rows),
        "responses_saved": sum(row.get("usage_coverage", {}).get("responses_saved", 0) for row in rows),
        "responses_with_usage": sum(row.get("usage_coverage", {}).get("responses_with_usage", 0) for row in rows),
    }


def _labels(root, cases=CASES):
    gpu = _read(root / "private_data" / "validation_gpu.json").get("cases", {})
    cpu = {}
    for path in sorted((root / "private_data").glob("answer_key_*.json")):
        data = _read(path)
        for case, row in data.get("cases", {}).items():
            if case in cases:
                if case in cpu:
                    raise ValueError(f"Duplicate CPU key: {case}")
                cpu[case] = {**row, "family": data["family"]}
    labels = {}
    for case in cases:
        candidate, validated = cpu.get(case, {}), gpu.get(case, {})
        if validated and not candidate:
            raise ValueError(f"GPU label without CPU key: {case}")
        for kind in ("kernel", "problem"):
            path = root / "eval_cases" / case / ("kernel.py" if kind == "kernel" else "problem.txt")
            key = f"{kind}_sha256"
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            if candidate.get(key) and actual != candidate[key]:
                raise ValueError(f"Frozen public source changed: {case}/{kind}")
            if validated and validated.get(key) != candidate.get(key):
                raise ValueError(f"CPU/GPU source mismatch: {case}/{kind}")
        if validated and validated.get("ground_truth") != candidate.get("cpu_ground_truth"):
            raise ValueError(f"CPU/GPU label mismatch: {case}")
        if validated and validated.get("input_sha256") != candidate.get("input_sha256"):
            raise ValueError(f"CPU/GPU input mismatch: {case}")
        labels[case] = {
            "family": candidate.get("family", "pending"),
            "truth": validated.get("ground_truth", candidate.get("cpu_ground_truth")),
            "truth_status": "gpu_verified" if validated else "cpu_only" if candidate else "pending",
            "budget": candidate.get("budget"), "cpu_error": candidate.get("error"),
            "gpu_errors": validated.get("errors", []),
            "gpu_max_error": max(validated["errors"]) if validated.get("errors") else None,
            "kernel_sha256": candidate.get("kernel_sha256"),
            "problem_sha256": candidate.get("problem_sha256"),
            "gpu_bitwise_repeatable": validated.get("bitwise_repeatable"),
            "gpu_environment": validated.get("environment"),
        }
    return labels


def _request_config(path, metadata):
    requests = sorted((path / "llm_calls").glob("*/request.json"))
    if not requests and (path / "request.json").exists():
        requests = [path / "request.json"]
    configs = []
    for source in requests:
        request = _read(source)
        configs.append({"reasoning_effort": request.get("reasoning_effort", "default"),
                        "max_tokens": request.get("max_tokens"), "model": request.get("model")})
    unique = {json.dumps(config, sort_keys=True) for config in configs}
    if len(unique) == 1:
        return configs[0], configs
    if unique:
        return {"reasoning_effort": "mixed", "max_tokens": "mixed", "model": "mixed"}, configs
    return {"reasoning_effort": metadata.get("reasoning_effort", "unknown"),
            "max_tokens": metadata.get("max_tokens"), "model": metadata.get("model")}, configs


def _protocol_issues(row, frozen, planned=PLANNED):
    expected = planned.get(experiment_trial(row))
    issues = []
    if not expected:
        issues.append("outside prespecified extension trials")
    elif row["arm"] not in expected["arms"]:
        issues.append("arm outside planned trial")
    if row["model"] != MODEL or row["provider"] != "fireworks":
        issues.append("model/provider mismatch")
    if expected:
        for key in ("reasoning_effort", "max_tokens"):
            if row[key] != expected[key]:
                issues.append(f"{key} differs from {expected[key]}")
    if row["arm"] in {"solo", "debate"}:
        rounds = 10 if row["arm"] == "solo" else 4
        if row.get("max_rounds") != rounds:
            issues.append(f"max_rounds differs from {rounds}")
    for key in ("kernel_sha256", "problem_sha256"):
        if not frozen.get(key) or row.get(key) != frozen[key]:
            issues.append(f"unverified or changed {key}")
    if any(config["model"] != row["model"] for config in row["request_configs"]):
        issues.append("captured request model mismatch")
    if row["status"] == "completed" and not row["request_configs"]:
        issues.append("completed run has no raw request")
    return issues


def _valid(row):
    if not row:
        return False
    coverage = row.get("usage_coverage", {})
    captured = coverage.get("captured_calls", 0)
    capture_complete = (coverage.get("status") == "complete" and captured > 0
                        and coverage.get("responses_saved") == captured
                        and coverage.get("responses_with_usage") == captured
                        and len(row.get("request_configs", [])) == captured)
    return bool(row["status"] == "completed" and not row["protocol_issues"]
                and not row.get("runner_error") and capture_complete)


def replication_gate(rows, labels, *, cases=CASES, low_trials=LOW_TRIALS, required_families=2):
    slots = defaultdict(list)
    for row in rows:
        if experiment_trial(row) in low_trials:
            slots[(row["case"], row["arm"], experiment_trial(row))].append(row)
    details = {}
    for case in cases:
        windows = {arm: [slots[(case, arm, trial)] for trial in
                         (low_trials if arm == "single_call" else low_trials[:2])]
                   for arm in ARMS}
        unique = {arm: [slot[0] if len(slot) == 1 else None for slot in window]
                  for arm, window in windows.items()}
        finished = all(row and row["status"] in TERMINAL for window in unique.values() for row in window)
        source_valid = all(_valid(row) and row["outcome"] in {"correct", "wrong_verdict"}
                           for row in unique["single_call"])
        wrong = sum(_valid(row) and row["outcome"] == "wrong_verdict" for row in unique["single_call"])
        tool_correct = {arm: sum(_valid(row) and row["outcome"] == "correct" for row in unique[arm])
                        for arm in ("solo", "debate")}
        verified = labels[case]["truth_status"] == "gpu_verified"
        passed = verified and source_valid and wrong >= 2 and all(value == 2 for value in tool_correct.values())
        duplicate = any(len(slot) > 1 for window in windows.values() for slot in window)
        details[case] = {
            "family": labels[case]["family"], "passed": passed,
            "status": "passed" if passed else "invalid_duplicate_slot" if duplicate else
                      "failed" if finished and verified else "pending",
            "source_explicit_errors_first_three": wrong,
            "source_all_three_protocol_valid": source_valid,
            "tool_correct_first_two": tool_correct,
            "window_attempts": {arm: [[row["attempt_id"] for row in slot] for slot in window]
                                for arm, window in windows.items()},
        }
    families = sorted({row["family"] for row in details.values() if row["passed"]})
    return {"cases": details, "qualifying_cases": [case for case, row in details.items() if row["passed"]],
            "qualifying_families": families, "required_families": required_families,
            "passed": len(families) >= required_families,
            "definition": "For each GPU-verified case, the fixed low32 r1/r2/r3 source slots must all complete under the protocol and contain at least two explicit wrong verdicts; the fixed low32 r1/r2 slots must both be correct for each tool arm. Failed, missing, exhausted, or abstaining slots are never replaced.",
            "interpretation": "Exploratory replication of selected fixed workloads; not a generalization or statistical-significance claim. Default64 is reported separately and cannot qualify the low32 gate."}


def derive(root=ROOT, *, cases=CASES, low_trials=LOW_TRIALS, planned=PLANNED,
           required_families=2, case_range="case_44–case_49"):
    root = Path(root)
    labels = _labels(root, cases)
    records = [record for record in iter_trace_records(benchmark_dir=root.parent)
               if record["dataset"] == DATASET and record["case"] in cases]
    common = build_report(records=records, benchmark_dir=root.parent,
                          labels={DATASET: {case: row["truth"] for case, row in labels.items()
                                            if row["truth"] is not None}})
    metadata_by_path = {str(Path(record["path"]).resolve()): record["metadata"] for record in records}
    rows = []
    for group in common["arms"].values():
        for source in group["per_case"].values():
            row = dict(source)
            path = root.parent / row["path"]
            metadata = metadata_by_path[str(path.resolve())]
            config, captured = _request_config(path, metadata)
            row.update({"attempt_id": "/".join(str(row[key]) for key in
                                               ("case", "arm", "trial", "provider", "model")),
                        "status": metadata.get("status", "unknown"),
                        "created_at": metadata.get("created_at"),
                        "reasoning_effort": config["reasoning_effort"], "max_tokens": config["max_tokens"],
                        "max_rounds": metadata.get("max_rounds"), "request_configs": captured,
                        "kernel_sha256": metadata.get("kernel_sha256"),
                        "problem_sha256": metadata.get("problem_sha256"),
                        "raw_coverage": row.get("usage_coverage", {}).get("status", "unknown"),
                        "truth_status": labels[row["case"]]["truth_status"]})
            row["observed_outcome"] = row["outcome"]
            if row["status"] in {"running", "queued", "reserved", "started"}:
                row["outcome"], row["correct"] = "pending", None
            row["protocol_issues"] = _protocol_issues(row, labels[row["case"]], planned)
            row["trace_link"] = "../" + row["path"] + "/" + (
                "transcript.md" if (path / "transcript.md").exists() else "trace_meta.json")
            row["provider_errors"] = [{"call": failure.parent.name, "details": _read(failure)}
                                      for failure in sorted((path / "llm_calls").glob("*/error.json"))]
            rows.append(row)
    rows.sort(key=lambda row: (row["case"], row["arm"], trial_sort_key(row["trial"]), row["attempt_id"]))
    grouped = defaultdict(list)
    for row in rows:
        key = (experiment_trial(row), row["provider"], row["model"], row["reasoning_effort"], row["max_tokens"], row["arm"], row["max_rounds"],
               (row.get("protocol") or {}).get("total_output_token_budget"))
        grouped[key].append(row)
    for trial, config in planned.items():
        for arm in config["arms"]:
            rounds = 10 if arm == "solo" else 4 if arm == "debate" else None
            grouped.setdefault((trial, "fireworks", MODEL, config["reasoning_effort"], config["max_tokens"], arm, rounds, None), [])
    groups = []
    for key, attempts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        trial, provider, model, reasoning, tokens, arm, rounds, total_output_budget = key
        counts = _counts(attempts)
        groups.append({"trial": trial, "original_trial": trial, "provider": provider, "model": model, "reasoning_effort": reasoning,
                       "total_output_token_budget": total_output_budget,
                       "max_tokens": tokens, "max_rounds": rounds, "arm": arm, "attempts": len(attempts), "outcomes": counts,
                       "missing_cases": [case for case in cases if case not in {row["case"] for row in attempts}],
                       "no_result": sum(counts[name] for name in ("abstention", "token_limit", "no_verdict")),
                       "protocol_invalid_attempts": sum(bool(row["protocol_issues"]) for row in attempts),
                       **_cost(attempts)})
    pairs = defaultdict(dict)
    for row in rows:
        if row["arm"] in {"solo", "debate"}:
            key = (row["case"], experiment_trial(row), row["provider"], row["model"], row["reasoning_effort"], row["max_tokens"],
                   (row.get("protocol") or {}).get("total_output_token_budget"))
            if row["arm"] in pairs[key]:
                raise ValueError(f"Duplicate experiment arm: {key}/{row['arm']}")
            pairs[key][row["arm"]] = row
    comparisons = []
    for key, pair in sorted(pairs.items(), key=lambda item: tuple(map(str, item[0]))):
        if set(pair) != {"solo", "debate"}:
            continue
        solo, debate = pair["solo"], pair["debate"]
        usable = _valid(solo) and _valid(debate) and labels[key[0]]["truth_status"] == "gpu_verified"
        comparisons.append({"case": key[0], "trial": key[1], "config": list(key[2:]),
                            "solo_outcome": solo["outcome"], "debate_outcome": debate["outcome"],
                            "usable": usable, "both_correct": usable and solo["outcome"] == debate["outcome"] == "correct",
                            "debate_corrects_solo_wrong": usable and solo["outcome"] == "wrong_verdict" and debate["outcome"] == "correct",
                            "solo_corrects_debate_wrong": usable and solo["outcome"] == "correct" and debate["outcome"] == "wrong_verdict"})
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": DATASET, "cases": labels,
            "case_range": case_range, "planned_trials": planned, "rows": rows, "groups": groups,
            "replication_gate": replication_gate(rows, labels, cases=cases, low_trials=low_trials,
                                                  required_families=required_families),
            "matched_tool_comparisons": comparisons,
            "cost_note": "Project profile API estimates, not invoices. Unknown or unrecorded charges and Modal GPU charges are excluded. Group accuracy is never pooled across different trial configurations.",
            **_cost(rows)}


def _cell(value):
    return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def _money(value):
    return "unknown" if value is None else f"${value:.6f}"


def render(report):
    case_range = report.get("case_range", "case_44–case_49")
    lines = [f"# Numerical extension {case_range}", "", f"Generated: {report['generated_at']}", "",
             f"This report is derived only from canonical {case_range} traces. CPU-only labels are provisional; GPU verification is required for the replication gate. Missing slots are pending, not successful attempts.", "",
             "| Case | Family | Truth | Evidence | CPU error | GPU max error | Tolerance |",
             "|---|---|---|---|---:|---:|---:|"]
    for case, row in report["cases"].items():
        lines.append("| " + " | ".join(_cell(value) for value in (case, row["family"], row["truth"], row["truth_status"], row["cpu_error"], row["gpu_max_error"], row["budget"])) + " |")
    lines.extend(["", "## Per-trial results", "",
                  "Correct, explicit wrong, and no-result counts are separate. No result includes abstention, token exhaustion, or missing verdict; pending and not-started slots are shown separately. Default means the reasoning_effort field was omitted from the captured API request.", "",
                  "Trace trial names are local rN identifiers; experiment batches below use preserved original_trial metadata.", "",
                  "| Experiment batch | Arm | Provider / model | Reasoning / token cap / rounds | Attempts | Correct | Explicit wrong | No result | Pending | Not started | API estimate | Raw coverage |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for group in report["groups"]:
        counts = group["outcomes"]
        values = (group["trial"], group["arm"], f"{group['provider']} / {group['model']}",
                  f"{group['reasoning_effort']} / {group['max_tokens']} / {_cell(group['max_rounds'])}",
                  group["attempts"], counts["correct"], counts["wrong_verdict"], group["no_result"], counts["pending"],
                  len(group["missing_cases"]), _money(group["recorded_api_estimate_usd"]),
                  json.dumps(group["raw_coverage_counts"], sort_keys=True))
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines.extend(["", f"Recorded API estimate across all attempts: {_money(report['recorded_api_estimate_usd'])}; "
                  f"unpriced attempts: {report['unpriced_attempts']}; partial-cost attempts: {report['partial_cost_attempts']}. "
                  f"Raw calls: {report['captured_calls']}; saved responses: {report['responses_saved']}; responses with usage: {report['responses_with_usage']}.",
                  "", report["cost_note"], "", "## Fixed replication window", "",
                  f"Overall gate: {'PASSED' if report['replication_gate']['passed'] else 'NOT YET MET'}; "
                  f"{len(report['replication_gate']['qualifying_families'])} qualifying families "
                  f"(target: at least {report['replication_gate']['required_families']}).", "",
                  report["replication_gate"]["definition"], "", report["replication_gate"]["interpretation"], "",
                  "| Case | Status | Source explicit errors / 3 | Solo correct / 2 | Debate correct / 2 |",
                  "|---|---|---:|---:|---:|"])
    for case, row in report["replication_gate"]["cases"].items():
        lines.append(f"| {case} | {row['status']} | {row['source_explicit_errors_first_three']} | {row['tool_correct_first_two']['solo']} | {row['tool_correct_first_two']['debate']} |")
    comparisons = report["matched_tool_comparisons"]
    lines.extend(["", f"Matched tool comparisons: {len(comparisons)}; valid both-correct pairs: "
                  f"{sum(row['both_correct'] for row in comparisons)}; debate corrects an explicit solo error: "
                  f"{sum(row['debate_corrects_solo_wrong'] for row in comparisons)}; solo corrects an explicit debate error: "
                  f"{sum(row['solo_corrects_debate_wrong'] for row in comparisons)}.", "",
                  "## Every recorded attempt", "",
                  "| Case | Trial | Arm | Status | Verdict | Outcome | API estimate | Raw coverage | Trace / protocol issues |",
                  "|---|---|---|---|---|---|---:|---|---|"])
    for row in report["rows"]:
        values = (row["case"], row["trial"], row["arm"], row["status"], row["verdict"], row["outcome"],
                  _money(row["usd"]), row["raw_coverage"], f"[trace]({row['trace_link']}) " + "; ".join(row["protocol_issues"]))
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    if not report["rows"]:
        lines.extend(["", f"No canonical {case_range} model attempts exist yet; all scheduled slots remain pending."])
    return "\n".join(lines) + "\n"


def main(*, export_json=False):
    report = derive()
    if export_json:
        output = ROOT / "private_data" / "reports" / "extension_scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "EXTENSION_REPORT.md").write_text(render(report))
    print(f"Recorded attempts: {len(report['rows'])}; qualifying cases: {report['replication_gate']['qualifying_cases']}")
    print(f"API estimate: {_money(report['recorded_api_estimate_usd'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    main(export_json=parser.parse_args().json)
