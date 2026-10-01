"""Derive this experiment's report from every canonical numerical-challenge trace.

Read-only with respect to cases, labels and traces. Running this file writes only
the MAIN RESULTS block in single_call_vs_tools_challenges/README.md by default. With --json, also export private_data/reports/scoreboard.json.
No provider or Modal call is made.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "single_call_vs_tools_challenges"
sys.path.insert(0, str(ROOT.parent / "eval_scripts"))
from summarize_traces import build_report
from generated_results import compact_results, replace_results
from traces import iter_trace_records, experiment_trial, trace_selection_key

DATASET = "numerical_challenges"
MODEL = "accounts/fireworks/models/glm-5p3"
ARMS = ("single_call", "solo", "debate")
OUTCOMES = ("correct", "wrong_verdict", "abstention", "token_limit", "no_verdict", "unscored", "pending")


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def _counts(rows):
    counts = Counter(row["outcome"] for row in rows)
    return {name: counts[name] for name in OUTCOMES}


def _costs(rows):
    priced = [row for row in rows if row.get("usd") is not None]
    complete = [row for row in priced if row["usage_coverage_status"] == "complete"]
    partial = [row for row in priced if row["usage_coverage_status"] == "partial"]
    return {
        "recorded_api_estimate_usd": round(sum(row["usd"] for row in priced), 6) if priced else None,
        "complete_capture_estimate_usd": round(sum(row["usd"] for row in complete), 6),
        "partial_capture_estimate_usd": round(sum(row["usd"] for row in partial), 6),
        "unknown_cost_attempts": [row["attempt_id"] for row in rows if row.get("usd") is None],
        "partial_usage_attempts": [row["attempt_id"] for row in rows
                                   if row["usage_coverage_status"] == "partial"],
        "unknown_coverage_attempts": [row["attempt_id"] for row in rows
                                     if row["usage_coverage_status"] not in {"complete", "partial"}],
        "input_tokens_recorded": sum(row.get("input_tokens", 0) for row in rows),
        "output_tokens_recorded": sum(row.get("output_tokens", 0) for row in rows),
        "pricing_note": "Project profile estimate, not invoice; unknown/unrecorded charges and Modal GPU charges excluded.",
    }


def _finished(row):
    return row["status"] in {"completed", "error"}


def _valid_finished(row):
    return row["status"] == "completed" and not row["protocol_issues"] and not row.get("runner_error")


def replication_gate(rows, labels):
    """Use fixed earliest windows, including unsuccessful attempts in their slots.

    A later success cannot replace an earlier failure, abstention or exhausted
    attempt. Other models/endpoints are displayed but cannot certify this GLM
    protocol. Changed case hashes fail the gate rather than silently resetting it.
    """
    by_case = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["provider"] == "fireworks" and row["model"] == MODEL and row["arm"] in ARMS:
            by_case[row["case"]][row["arm"]].append(row)
    details = {}
    for case, label in sorted(labels.items()):
        arms = by_case[case]
        for attempts in arms.values():
            attempts.sort(key=trace_selection_key)
        source = arms["single_call"][:3]
        tool_windows = {arm: arms[arm][:2] for arm in ("solo", "debate")}
        source_finished = len(source) == 3 and all(_finished(row) for row in source)
        source_valid = len(source) == 3 and all(_valid_finished(row) for row in source)
        source_wrong = sum(_valid_finished(row) and row["outcome"] == "wrong_verdict" for row in source)
        tool_correct = {arm: sum(_valid_finished(row) and row["outcome"] == "correct" for row in window)
                        for arm, window in tool_windows.items()}
        passed = source_valid and source_wrong >= 2 and all(tool_correct[arm] == 2 for arm in tool_windows)
        enough_finished = source_finished and all(len(window) == 2 and all(_finished(row) for row in window)
                                                  for window in tool_windows.values())
        later_tool_counterexamples = [row["attempt_id"] for arm in ("solo", "debate")
                                      for row in arms[arm][2:]
                                      if _finished(row) and (not _valid_finished(row) or row["outcome"] != "correct")]
        details[case] = {
            "family": label["family"], "passed": passed,
            "status": "passed" if passed else "failed" if enough_finished else "pending",
            "source_wrong_in_first_three": source_wrong,
            "source_first_three_finished": source_finished,
            "source_first_three_protocol_valid": source_valid,
            "source_first_three_outcomes": _counts(source),
            "tool_correct_in_first_two": tool_correct,
            "window_attempts": {"single_call": [row["attempt_id"] for row in source],
                                **{arm: [row["attempt_id"] for row in window]
                                   for arm, window in tool_windows.items()}},
            "total_attempts": {arm: len(arms[arm]) for arm in ARMS},
            "all_attempt_outcomes": {arm: _counts(arms[arm]) for arm in ARMS},
            "later_tool_counterexamples": later_tool_counterexamples,
        }
    families = sorted({row["family"] for row in details.values() if row["passed"]})
    return {
        "passed": len(families) >= 2, "qualifying_families": families, "required_families": 2,
        "cases": details,
        "definition": "At least two families each have a frozen case with >=2 explicit source errors in its earliest 3 attempts, and both earliest 2 attempts correct for each tool arm. All gate attempts must finish successfully under the frozen 65536-token GLM protocol.",
        "interpretation": "Exploratory replication gate, not a generalization or statistical-significance claim. Later counterexamples remain disclosed.",
    }


def _construction_summary(root):
    entries = []
    for path in sorted((root / "private_data").glob("search_log_*.json")):
        data = _read(path)
        candidates = data.get("candidates") if isinstance(data, dict) else None
        entries.append({
            "family": data.get("family", path.stem.removeprefix("search_log_")) if isinstance(data, dict)
                      else path.stem.removeprefix("search_log_"),
            "recorded_cpu_candidate_count": len(candidates) if isinstance(candidates, list) else None,
            "recorded_construction_rows": len(data) if isinstance(data, list) else None,
            "path": str(path.relative_to(root)),
            "note": "CPU construction records; not model attempts. A selected-row log does not certify an exhaustive search count.",
        })
    return entries


def derive(root=ROOT, *, cases=None):
    root = Path(root)
    validation = _read(root / "private_data" / "validation_gpu.json")
    if cases is not None:
        validation = {**validation, "cases": {case: row for case, row in validation["cases"].items() if case in cases}}
    labels = validation["cases"]
    benchmark = root.parent
    records = [record for record in iter_trace_records(benchmark_dir=benchmark)
               if record["dataset"] == DATASET and (cases is None or record["case"] in cases)]
    common = build_report(records=records, benchmark_dir=benchmark,
                          labels={DATASET: {case: row["ground_truth"] for case, row in labels.items()}})
    by_path = {str(Path(record["path"]).resolve()): record for record in records}
    rows = []
    for group in common["arms"].values():
        for derived in group["per_case"].values():
            row = dict(derived)
            path = benchmark / row["path"]
            record = by_path[str(path.resolve())]
            metadata = record.get("metadata") or _read(path / "trace_meta.json")
            frozen = labels.get(row["case"], {})
            row.update({
                "attempt_id": "/".join(str(row[key]) for key in ("case", "arm", "trial")),
                "status": metadata.get("status", "unknown"),
                "created_at": metadata.get("created_at"),
                "family": frozen.get("family", "unvalidated"),
                "max_tokens": metadata.get("max_tokens"),
                "max_rounds": metadata.get("max_rounds"),
                "kernel_sha256": metadata.get("kernel_sha256"),
                "problem_sha256": metadata.get("problem_sha256"),
            })
            row["observed_outcome"] = row["outcome"]
            if row["status"] in {"running", "queued", "reserved", "started"}:
                row["outcome"] = "pending"
                row["correct"] = None
            row["protocol_issues"] = []
            if row["model"] != MODEL or row["provider"] != "fireworks":
                row["protocol_issues"].append("different model or provider")
            if row["max_tokens"] != 65536:
                row["protocol_issues"].append("max_tokens differs from 65536")
            if row["arm"] in {"solo", "debate"}:
                expected_rounds = 10 if row["arm"] == "solo" else 4
                if row["max_rounds"] != expected_rounds:
                    row["protocol_issues"].append(f"max_rounds differs from {expected_rounds}")
            for key in ("kernel_sha256", "problem_sha256"):
                if not frozen.get(key) or row[key] != frozen[key]:
                    row["protocol_issues"].append(f"unverified or changed {key}")
            row["usage_coverage_status"] = row.get("usage_coverage", {}).get("status", "unknown")
            row["api_calls_captured"] = row.get("usage_coverage", {}).get("captured_calls", 0)
            row["responses_with_usage"] = row.get("usage_coverage", {}).get("responses_with_usage", 0)
            row["missing_response_calls"] = row.get("usage_coverage", {}).get("missing_response_calls", [])
            verdict = _read(path / "verdict.json")
            row["reason"] = verdict.get("reason") or row.get("reason")
            failure_file = metadata.get("failure_file")
            if failure_file and Path(failure_file).name == failure_file:
                failure = _read(path / failure_file)
                row["runner_error"] = row.get("runner_error") or failure.get("message") or failure.get("failure_kind")
            row["provider_errors"] = [
                {"call": failure.parent.name, "details": _read(failure)}
                for failure in sorted((path / "llm_calls").glob("*/error.json"))]
            row["trace_link"] = "../" + row["path"] + "/" + (
                "transcript.md" if (path / "transcript.md").exists() else "trace_meta.json")
            rows.append(row)
    rows.sort(key=lambda row: (row["case"], ARMS.index(row["arm"]) if row["arm"] in ARMS else 99,
                               trace_selection_key(row)))
    ordinal = Counter()
    for row in rows:
        key = (row["case"], row["arm"], row["model"], row["provider"])
        ordinal[key] += 1
        row["attempt_ordinal"] = ordinal[key]
        count = ordinal[key]
        limit = 3 if row["arm"] == "single_call" else 2
        row["phase"] = "initial" if count == 1 else "confirmation" if count <= limit else "additional"
    arms = {}
    for arm in ARMS:
        attempts = [row for row in rows if row["arm"] == arm]
        arms[arm] = {
            "attempts": len(attempts), "finished_attempts": sum(_finished(row) for row in attempts),
            "pending_attempts": sum(row["outcome"] == "pending" for row in attempts),
            "phase_counts": dict(Counter(row["phase"] for row in attempts)),
            "finished_phase_counts": dict(Counter(row["phase"] for row in attempts if _finished(row))),
            "outcomes": _counts(attempts), "status_counts": dict(Counter(row["status"] for row in attempts)),
            "probes": sum(row.get("probes", 0) for row in attempts), **_costs(attempts),
        }
    # Local rN names are independent per arm; pair the original experiment batch.
    paired = defaultdict(dict)
    for row in rows:
        if row["arm"] in {"solo", "debate"}:
            protocol = row.get("protocol") or {}
            key = (row["case"], experiment_trial(row), row["provider"], row["model"],
                   json.dumps({k: v for k, v in protocol.items() if k != "max_rounds"}, sort_keys=True))
            if row["arm"] in paired[key]:
                raise ValueError(f"Duplicate experiment arm: {key}/{row['arm']}")
            paired[key][row["arm"]] = row
    comparisons = []
    for key, pair in sorted(paired.items()):
        if set(pair) != {"solo", "debate"}:
            continue
        solo, debate = pair["solo"], pair["debate"]
        usable = _valid_finished(solo) and _valid_finished(debate)
        comparisons.append({
            "case": key[0], "trial": key[1], "provider": key[2], "model": key[3],
            "solo_outcome": solo["outcome"], "debate_outcome": debate["outcome"],
            "both_correct": usable and solo["outcome"] == debate["outcome"] == "correct",
            "debate_corrects_solo_wrong_verdict": usable and solo["outcome"] == "wrong_verdict" and debate["outcome"] == "correct",
            "solo_corrects_debate_wrong_verdict": usable and solo["outcome"] == "correct" and debate["outcome"] == "wrong_verdict",
            "operationally_complete_and_protocol_valid": usable,
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "dataset": DATASET,
        "protocol": "README.md", "configured_model": MODEL, "labels": validation,
        "construction": _construction_summary(root), "rows": rows, "arms": arms,
        "candidate_cases_validated": len(labels), "cases_with_recorded_attempts": len({row["case"] for row in rows}),
        "initial_attempts": sum(row["phase"] == "initial" for row in rows),
        "confirmation_attempts": sum(row["phase"] == "confirmation" for row in rows),
        "additional_attempts": sum(row["phase"] == "additional" for row in rows),
        "finished_initial_attempts": sum(row["phase"] == "initial" and _finished(row) for row in rows),
        "finished_confirmation_attempts": sum(row["phase"] == "confirmation" and _finished(row) for row in rows),
        "finished_additional_attempts": sum(row["phase"] == "additional" and _finished(row) for row in rows),
        "replication_gate": replication_gate(rows, labels), "matched_tool_comparisons": comparisons,
        "model_provider_trial_groups": common["arms"], **_costs(rows),
    }


def _cell(value):
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _money(value):
    return "unknown" if value is None else f"${value:.6f}"


def render(report):
    """Compact all-history summary; full rows and gates remain in derive()."""
    gate = report["replication_gate"]
    comparisons = report["matched_tool_comparisons"]
    fixes = sum(row["debate_corrects_solo_wrong_verdict"] for row in comparisons)
    reverse = sum(row["solo_corrects_debate_wrong_verdict"] for row in comparisons)
    lines = [f"生成时间：{report['generated_at']}", "",
        "初始组 case_38–case_43；后续两组采用各自约定的实验协议。"
        "协议与固定复验窗口见本页说明。", "",
        compact_results(report["rows"]),
        f"固定窗口复验门槛： **{'通过' if gate['passed'] else '未达到'}**; "
        f"达到门槛的机制： {', '.join(gate['qualifying_families']) or 'none'} "
        f"({len(gate['qualifying_families'])}/{gate['required_families']}).", "",
        f"配对工具比较： {len(comparisons)}; debate 纠正 solo 明确错误： {fixes}; "
        f"solo 纠正 debate 明确错误： {reverse}. 这些选定案例的比较不能推出普遍优势。"]
    return "\n".join(lines) + "\n"


def main(*, export_json=False):
    report = derive(cases=tuple(f"case_{i:02d}" for i in range(38, 44)))
    replace_results(ROOT, "MAIN RESULTS", render(report))
    if export_json:
        output = ROOT / "private_data" / "reports" / "scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"attempts": len(report["rows"]), "replication_gate_passed": report["replication_gate"]["passed"],
                      "qualifying_families": report["replication_gate"]["qualifying_families"],
                      "recorded_api_estimate_usd": report["recorded_api_estimate_usd"]}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    main(export_json=parser.parse_args().json)
