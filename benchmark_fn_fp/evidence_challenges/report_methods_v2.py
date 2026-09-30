"""Generate the separate case_66–case_71 report without changing the first pilot."""
import json
from pathlib import Path

from report import derive

ROOT = Path(__file__).resolve().parent
CASES = tuple(f"case_{i + 61:02d}" for i in range(5, 11))
TRIALS = ("ea_methods_v2_r1", "ea_methods_v2_r2")


def main(*, version="v2", cases=CASES, trials=TRIALS, export_json=False):
    if version not in {"v2", "v3"}:
        raise ValueError("Unknown report version")
    result = derive(cases=cases, trials=trials)
    if export_json:
        output = ROOT / "private_data" / "reports" / f"methods_{version}_scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# Other-methods pilot {version}: {cases[0]}–{cases[-1]}", "", f"Generated: {result['generated_at']}", "",
             f"See [prospective protocol](METHODS_{version.upper()}_PROTOCOL.md). All three arms share a 32768 total output-token allowance; input tokens, dollars and GPU time differ.", "",
             "Trace trial names are local rN identifiers; experiment batches below use preserved original_trial metadata.", "",
             "| Experiment batch | Arm | Attempts | Outcomes | Recorded API estimate |",
             "|---|---|---:|---|---:|"]
    for row in result["groups"]:
        lines.append(f"| {row['trial']} | {row['arm']} | {row['attempts']} | {json.dumps(row['outcomes'])} | ${row['api_estimate_usd']:.6f} |")
    lines += ["", "Paired repeat gate (a positive result also requires evidence audit):",
              "The gate counts explicit wrong-verdict corrections and reverse explicit errors. Abstention is reported separately; a positive net gate is not necessarily a gain in total accuracy.", "```json",
              json.dumps(result["gate"], indent=2), "```", "",
              "| Case | Trial | Arm | Truth | Verdict | Outcome | Evidence | Issues |",
              "|---|---|---|---|---|---|---|---|"]
    for row in result["rows"]:
        lines.append(f"| {row['case']} | {row['trial']} | {row['arm']} | {row['truth']} | {row.get('verdict')} | {row['outcome']} | [trace]({row['trace_link']}) | {', '.join(row['issues'])} |")
    lines += ["", f"Recorded API estimate: ${result['api_estimate_usd']:.6f}; unknown-cost attempts: {result['unknown_cost_attempts']}; partial-cost attempts: {result['partial_cost_attempts']}.",
              "Excluded: Modal GPU and unreported failed-call charges. Explicit mistakes, abstentions, exhaustion and failures are distinct outcomes."]
    (ROOT / f"METHODS_{version.upper()}_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"groups": result["groups"], "gate": result["gate"],
                      "api_estimate_usd": result["api_estimate_usd"]}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    main(export_json=parser.parse_args().json)
