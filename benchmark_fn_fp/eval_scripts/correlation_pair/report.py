"""Refresh the generated-results block in correlation_pair/README.md.

Read-only with respect to public cases, labels, raw traces and saved prices.
No model or GPU calls are made. --json optionally exports private_data/reports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "correlation_pair"
sys.path.insert(0, str(ROOT.parent / "eval_scripts"))
from audit_traces import audit
from case_registry import case_sort_key
from summarize_traces import build_report
from traces import experiment_trial, iter_trace_records, trial_sort_key

DATASET = "correlation_pair"
ARMS = ("single_call", "solo", "debate")
OUTCOMES = ("correct", "wrong_verdict", "abstention", "token_limit", "no_verdict", "pending")
NEUTRAL_NOTE = "Either branch\nmay individually exceed 0.1 without violating the contract.\n"
BEGIN_RESULTS = "<!-- BEGIN GENERATED RESULTS -->"
END_RESULTS = "<!-- END GENERATED RESULTS -->"


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_checks(path, metadata, usage, run, public, frozen, variant):
    """Only the recorded explanatory-sentence ablation changes the contract hash."""
    code, problem = public["kernel"], public["problem"]
    if variant == "neutral_contract":
        source_hash = usage.get("source_problem_sha256") or metadata.get("source_problem_sha256")
        if NEUTRAL_NOTE not in problem or source_hash != frozen["problem_sha256"]:
            raise ValueError(f"Neutral-contract source mismatch: {path}")
        problem = problem.replace(NEUTRAL_NOTE, "")
    elif variant != "original":
        raise ValueError(f"Unknown prompt variant: {path}: {variant}")
    checked = []
    for name, values in (("usage", usage), ("metadata", metadata)):
        for field, expected in (("kernel_sha256", frozen["kernel_sha256"]),
                                ("problem_sha256", digest(problem))):
            if values.get(field):
                if values[field] != expected:
                    raise ValueError(f"Historical {name}.{field} mismatch: {path}")
                checked.append(f"{name}.{field}")
    artifact = run.get("artifact") or {}
    for field, expected in (("kernel_code", code), ("problem_text", problem)):
        if field in artifact:
            if artifact[field] != expected:
                raise ValueError(f"Historical artifact.{field} mismatch: {path}")
            checked.append(f"artifact.{field}")
    if (path / "user_prompt.txt").exists():
        prompt = (path / "user_prompt.txt").read_text()
        if code not in prompt or problem not in prompt:
            raise ValueError(f"Saved prompt source/contract mismatch: {path}")
        checked.append("user_prompt.txt")
    return checked


def capture(path, arm):
    calls = sorted(p for p in (path / "llm_calls").glob("*") if p.is_dir())
    if calls:
        complete = all((p / "request.json").exists() and (p / "response.json").exists() for p in calls)
        return "complete_api" if complete else "partial_api"
    if arm == "single_call":
        if (path / "request.json").exists() and (path / "raw_response.json").exists():
            return "complete_api"
        if (path / "user_prompt.txt").exists() and (path / "response_text.json").exists():
            return "text_and_usage_only"
    return "history_only" if (path / "run.json").exists() else "missing"


def configuration(path, metadata, usage):
    request = read(path / "request.json")
    if not request:
        requests = sorted((path / "llm_calls").glob("*/request.json"))
        request = read(requests[0]) if requests else {}
    return {
        "max_tokens": metadata.get("max_tokens", usage.get("max_tokens", request.get("max_tokens"))),
        "total_output_token_budget": metadata.get("total_output_token_budget"),
        "reasoning_effort": metadata.get("reasoning_effort", request.get("reasoning_effort")),
        "max_rounds": metadata.get("max_rounds"),
        "prompt_variant": usage.get("prompt_variant", metadata.get("prompt_variant", "original")),
    }


def counts(rows):
    values = Counter(row["outcome"] for row in rows)
    return {outcome: values[outcome] for outcome in OUTCOMES}


def costs(rows):
    priced = [row["usd"] for row in rows if row.get("usd") is not None]
    return {"estimated_api_usd": round(sum(priced), 8) if priced else None,
            "unknown_cost_attempts": sum(row.get("usd") is None for row in rows),
            "partial_cost_attempts": sum(bool(row.get("usd_is_partial")) for row in rows)}


def derive(root=ROOT):
    root = Path(root)
    benchmark = root.parent
    labels = read(root / "private_data" / "validation_gpu.json")
    if not labels.get("cases"):
        raise ValueError("Missing frozen correlation-pair labels")
    public = {}
    for case, frozen in labels["cases"].items():
        public[case] = {}
        for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt")):
            content = (benchmark / "triton_eval_cases" / case / filename).read_text()
            if digest(content) != frozen[f"{kind}_sha256"]:
                raise ValueError(f"Frozen public source changed: {case}/{filename}")
            public[case][kind] = content
    records = [r for r in iter_trace_records(benchmark_dir=benchmark)
               if r["dataset"] == DATASET
               and r["path"].relative_to(benchmark).parts[0] in {"traces_glm", "traces_opus5"}]
    common = build_report(records=records, benchmark_dir=benchmark, labels={
        DATASET: {case: row["ground_truth"] for case, row in labels["cases"].items()}})
    by_path = {str(r["path"].relative_to(benchmark)): r for r in records}
    rows = []
    for group in common["arms"].values():
        for source in group["per_case"].values():
            row = dict(source)
            path = benchmark / row["path"]
            metadata = by_path[row["path"]]["metadata"]
            usage, run = read(path / "usage.json"), read(path / "run.json")
            config = configuration(path, metadata, usage)
            row.update(config)
            row.update(protocol=config, original_trial=experiment_trial(row),
                       status=metadata.get("status", "historical"))
            if row["status"] in {"running", "queued", "reserved", "started"}:
                row["outcome"], row["correct"] = "pending", None
            saved_cost = usage.get("estimated_usd")
            if isinstance(saved_cost, (int, float)) and not isinstance(saved_cost, bool):
                row["usd"], row["cost_source"] = saved_cost, "saved usage.estimated_usd"
            else:
                row["cost_source"] = "saved pricing snapshot" if metadata.get("pricing_snapshot") else "historical project model profile"
            row["cost_note"] = usage.get("pricing")
            row["source_checks"] = source_checks(path, metadata, usage, run,
                public[row["case"]], labels["cases"][row["case"]], row["prompt_variant"])
            row["raw_capture"] = capture(path, row["arm"])
            row["api_calls"] = (row.get("usage_coverage", {}).get("captured_calls")
                or sum(bool(turn.get("usage")) for turn in run.get("history", [])) or (1 if usage else 0))
            row["reason"] = (run.get("verdict") or {}).get("reason") or row.get("reason")
            row["audit_issues"] = audit(path) if run and (path / "tool_events.jsonl").exists() else []
            link_file = next((name for name in ("transcript.md", "user_prompt.txt", "trace_meta.json")
                              if (path / name).exists()), "trace_meta.json")
            row["trace_link"] = "../" + row["path"] + "/" + link_file
            rows.append(row)
    rows.sort(key=lambda r: (r["model"] or "", case_sort_key(r["case"]), ARMS.index(r["arm"]), trial_sort_key(r["trial"])))
    grouped = defaultdict(list)
    for row in rows:
        key = (row["provider"], row["model"], row["original_trial"], row["arm"], json.dumps(row["protocol"], sort_keys=True))
        grouped[key].append(row)
    groups = []
    for key, attempts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        provider, model, batch, arm, protocol = key
        groups.append({"provider": provider, "model": model, "original_trial": batch,
            "arm": arm, "protocol": json.loads(protocol), "attempts": len(attempts),
            "cases": [row["case"] for row in attempts], "outcomes": counts(attempts), **costs(attempts)})
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": DATASET,
            "labels": labels, "rows": rows, "groups": groups, "attempts": len(rows),
            "outcomes": counts(rows), "raw_capture": dict(Counter(row["raw_capture"] for row in rows)),
            "input_tokens": sum(row.get("input_tokens", 0) for row in rows),
            "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
            "gpu_cost": "not measured; excluded", **costs(rows)}


def cell(value):
    return "unknown" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def money(value):
    return "unknown" if value is None else f"${value:.8f}".rstrip("0").rstrip(".")


def render(result):
    lines = ["## 实验结果（自动生成）", "", f"Generated: {result['generated_at']}", "",
        f"All **{result['attempts']} attempts** from `traces_glm` and `traces_opus5` are included once. "
        "Local rN names identify stored attempts; original experiment batches remain separate.", "",
        "See [protocol](#protocol), [evidence audit](#evidence-audit) and "
        "[64K follow-up](#glm-64k-follow-up) for the manually maintained experimental context.", "",
        "### Frozen GPU labels", "", "| Case | Relative L2 error | Budget | Truth |", "|---|---:|---:|---|"]
    for case, row in sorted(result["labels"]["cases"].items(), key=lambda item: case_sort_key(item[0])):
        lines.append(f"| {case} | {max(row['errors']):.10f} | 0.1 | {row['ground_truth']} |")
    lines += ["", "Frozen environment: " + json.dumps(result["labels"].get("environment", {})), "",
        "### Experiment groups", "", "Token caps are per request unless a shared output budget is recorded. "
        "Missing historical settings remain unknown rather than inferred from current defaults.", "",
        "| Model / provider | Original batch | Arm | Token cap / shared cap / variant | Attempts | Correct | Wrong | Abstain | Token limit | No verdict | Pending | API estimate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in result["groups"]:
        config = group["protocol"]
        settings = " / ".join(cell(config[k]) for k in ("max_tokens", "total_output_token_budget", "prompt_variant"))
        lines.append(f"| {cell(group['model'])} / {cell(group['provider'])} | {cell(group['original_trial'])} | "
            f"{group['arm']} | {settings} | {group['attempts']} | "
            + " | ".join(str(group["outcomes"][k]) for k in OUTCOMES) + f" | {money(group['estimated_api_usd'])} |")
    lines += ["", "### Every attempt", "",
        "| Case | Model / provider | Arm | Trial | Original batch | Cap | Variant | Status | Verdict | Outcome | In / out tokens | API estimate / basis | Capture | Trace |",
        "|---|---|---|---|---|---:|---|---|---|---|---:|---|---|---|"]
    for row in result["rows"]:
        values = (row["case"], f"{row['model']} / {row['provider']}", row["arm"], row["trial"],
            row["original_trial"], row["max_tokens"], row["prompt_variant"], row["status"], row.get("verdict"),
            row["outcome"], f"{row.get('input_tokens', 0)} / {row.get('output_tokens', 0)}",
            f"{money(row['usd'])} / {row['cost_source']}", row["raw_capture"], f"[open]({row['trace_link']})")
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines += ["", f"Total recorded API estimate: **{money(result['estimated_api_usd'])}**; "
        f"unknown-cost attempts: {result['unknown_cost_attempts']}; partial-cost attempts: {result['partial_cost_attempts']}.",
        f"Outcomes: {json.dumps(result['outcomes'])}. Input / output tokens: {result['input_tokens']} / {result['output_tokens']}.", "",
        "Single-call costs prefer the original saved estimate. Other costs use saved pricing snapshots or historical "
        "project profiles, not today's list prices. These estimates exclude GPU charges and are not invoices.", "",
        "Capture: " + json.dumps(result["raw_capture"]) + ". `complete_api` requires original request/response payloads; "
        "`text_and_usage_only` has prompt/final text and usage without the API envelope; `history_only` has agent/tool "
        "transcripts and recorded usage without raw API calls.", "",
        "Public hashes are checked against frozen labels. The neutral-contract ablation permits only removal of the "
        "original explanatory branch-error sentence, with source and modified problem hashes checked separately.", "",
        "### Automatic trace-audit findings", ""]
    findings = [row for row in result["rows"] if row["audit_issues"]]
    if not findings:
        lines.append("No issues from the existing automatic checks; see the independent [evidence audit](#evidence-audit).")
    for row in findings:
        lines.append(f"- [{row['model']} / {row['case']} / {row['arm']} / {row['trial']}]({row['trace_link']}): "
                     + "; ".join(cell(issue) for issue in row["audit_issues"]))
    return "\n".join(lines) + "\n"


def main(*, root=ROOT, export_json=False):
    root = Path(root)
    readme = root / "README.md"
    content = readme.read_text() if readme.exists() else ""
    if (content.count(BEGIN_RESULTS) != 1 or content.count(END_RESULTS) != 1
            or content.index(BEGIN_RESULTS) >= content.index(END_RESULTS)):
        raise ValueError("README.md must contain exactly one ordered BEGIN/END GENERATED RESULTS marker pair; refusing to overwrite manual content")
    result = derive(root)
    before, remaining = content.split(BEGIN_RESULTS, 1)
    _, after = remaining.split(END_RESULTS, 1)
    readme.write_text(before + BEGIN_RESULTS + "\n" + render(result).rstrip() + "\n" + END_RESULTS + after)
    if export_json:
        output = root / "private_data" / "reports" / "scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"attempts": result["attempts"], "outcomes": result["outcomes"],
                      "estimated_api_usd": result["estimated_api_usd"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export private_data/reports/scoreboard.json")
    main(export_json=parser.parse_args().json)
