"""Regenerate the readable report from measured labels and saved API responses."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    key = json.loads((ROOT / "answer_key.json").read_text())
    results = json.loads((ROOT / "results_single.json").read_text())
    rows = results["cases"]
    summary = results["summary"]
    text = ["# 24-case numerical pilot: single-call results", "",
            f"Model: `{results['model']}`. One request per case, no tools or debate, "
            "8192 maximum output tokens, no retries or token escalation.", "",
            "## Results", "",
            "| Family | Cases | Correct | Wrong verdict | Needs evidence | No answer |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for family in key["budgets"]:
        rs = [r for r in rows.values() if r["family"] == family]
        text.append(f"| {family} | {len(rs)} | {sum(r['correct'] for r in rs)} | "
                    f"{sum(not r['correct'] and r['verdict'] in ('trust','reject') for r in rs)} | "
                    f"{sum(r['verdict']=='needs_more_evidence' for r in rs)} | "
                    f"{sum(r['verdict']=='no_verdict' for r in rs)} |")
    wrong = [r for r in rows.values() if not r["correct"] and r["verdict"] in ("trust", "reject")]
    capped = sum(r.get("stop_reason") == "max_tokens" for r in rows.values())
    determinate = sum(r["verdict"] in ("trust", "reject") for r in rows.values())
    text += ["", f"Overall: **{summary['correct']}/{summary['n']} correct**, "
             f"{len(wrong)} incorrect verdicts, {summary['inconclusive']} abstentions, "
             f"{summary['no_verdict']} missing answers.", "",
             f"Among {determinate} determinate trust/reject answers, {summary['correct']} are correct. "
             "This conditional rate must be read alongside answer coverage, not substituted for overall accuracy.", "",
             f"Of the missing answers, {capped} hit the output-token cap. These are budget "
             "exhaustions, not incorrect semantic judgments. The original baseline can escalate "
             "its token cap; this cost-bounded pilot deliberately does not, so raw accuracy "
             "is not a controlled head-to-head comparison to the old run.", "",
             f"Recorded API tokens: {summary['input_tokens']} input, {summary['output_tokens']} output. "
             f"Estimated API cost: **${summary['estimated_usd']:.4f}**, using the existing "
             "baseline's $5/$25 per million token assumption. Excludes Modal charges; not an invoice.", "",
             "## Ground-truth checks", "",
             "All 24 selected cases were run three times on NVIDIA T4. Each family has "
             "four passing and four failing workloads. Selected errors are <=0.75x or "
             ">=1.25x their predeclared family budget. CPU FP64 references were cross-checked "
             "against a second computation. No label depends on the model answer.", "",
             "## Case details", "",
             "Error/budget below 1 means compliant. Confidence is recorded, not used in scoring.", "",
             "| Case | Family | Error/budget | Truth | Verdict | Confidence | Correct |",
             "| --- | --- | ---: | --- | --- | ---: | --- |"]
    for name, r in sorted(rows.items()):
        k = key["cases"][name]
        ratio = k["errors"][0] / key["budgets"][r["family"]]
        text.append(f"| {name} | {r['family']} | {ratio:.4f} | {r['ground_truth']} | "
                    f"{r['verdict']} | {r.get('confidence', '—')} | {r['correct']} |")
    text += ["", "## Model explanations", ""]
    for name, r in sorted(rows.items()):
        text += [f"### {name}", "", r.get("reason", r.get("error_type", "No final answer")), ""]
    text += ["## Limits", "",
             "This is an exploratory finite-workload accuracy test, not universal kernel verification. "
             "The 24 cases are correlated variants of three kernels. Budgets are benchmark design choices, "
             "not externally established production standards. Deterministic seed arrays are fully specified "
             "in source, but a no-tool model must mentally reconstruct them, so computation access is a major "
             "confound. Abstention is not a confident error. A correct fixed numerical test can solve these "
             "cases. No result here establishes a multi-agent advantage; that requires comparing against "
             "a single agent with the same tools and matched budget.", "",
             "Construction history: first GPU launch had a module-import error and was stopped. The first "
             "successful candidate sweep did not supply enough failing quantization/recurrence cases. "
             "The budgets were kept fixed; residual-aligned quantization inputs and a lower-noise recurrence "
             "configuration were added before any LLM calls. Both candidate sweep records are retained.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(text))
    print("Wrote", ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
