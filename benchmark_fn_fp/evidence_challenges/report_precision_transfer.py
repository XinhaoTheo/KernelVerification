"""Report the prospective case_76–case_81 accuracy-based confirmation protocol.

Trace validity, costs and outcomes come from the existing audited report.
Only this cohort's numerical gate differs: accuracy gains must survive
abstentions as well as reverse errors, with a repeated explicit correction.
"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from benchmark_fn_fp.evidence_challenges import report as base_report

CASES = tuple(f"case_{i + 61:02d}" for i in range(15, 21))
TRIALS = ("ea_precision_transfer_r1", "ea_precision_transfer_r2")
ARMS = base_report.ARMS
LABEL_OUTCOMES = {"correct", "wrong_verdict", "abstention"}


def transfer_gate(rows, *, cases=CASES, trials=TRIALS):
    """Score only designated slots; preserve failures without calling them errors.

Correct-label counts use the fixed case denominator. Terminal failures and
abstentions are not correct, but only valid explicit wrong verdicts qualify
as corrections. Unverifiable reported labels cannot activate the gate.
    """
    if len(trials) != 2 or len(set(trials)) != 2 or not cases or len(set(cases)) != len(cases):
        raise ValueError("Need two distinct trials and a nonempty unique case list")
    designated = [row for row in rows if row.get("case") in cases
                  and row.get("trial") in trials and row.get("arm") in ARMS]
    slots = {(row["trial"], row["case"], row["arm"]): row for row in designated}
    if len(slots) != len(designated):
        raise ValueError("Duplicate designated slot")
    comparisons, rounds = [], []
    for trial in trials:
        keys = [(trial, case, arm) for case in cases for arm in ARMS]
        missing = [{"case": case, "arm": arm} for t, case, arm in keys if (t, case, arm) not in slots]
        pending = [{"case": case, "arm": arm} for t, case, arm in keys
                   if (row := slots.get((t, case, arm))) is not None
                   and (row.get("status") not in base_report.TERMINAL_STATUSES
                        or row.get("outcome") == "pending")]
        counts = {arm: Counter(slots[(trial, case, arm)].get("outcome", "unclassified")
                               for case in cases if (trial, case, arm) in slots)
                  for arm in ARMS}
        correct = {arm: counts[arm].get("correct", 0) for arm in ARMS}
        # Do not inflate a gain by silently discarding an invalid solo label.
        # Keep reported accuracy visible, but withhold the gate if a label used
        # in that comparison lacks complete source/budget/raw-call auditing.
        unauditable = [{"case": case, "arm": arm} for case in cases for arm in ("solo", "debate")
                      if (row := slots.get((trial, case, arm))) is not None
                      and row.get("outcome") in LABEL_OUTCOMES and not row.get("valid_complete")]
        for case in cases:
            solo, debate = (slots.get((trial, case, arm), {}) for arm in ("solo", "debate"))
            usable = bool(solo.get("valid_complete") and debate.get("valid_complete"))
            comparisons.append({"trial": trial, "case": case,
                "debate_corrects_solo": bool(usable and solo.get("outcome") == "wrong_verdict"
                                             and debate.get("outcome") == "correct"),
                "solo_corrects_debate": bool(usable and debate.get("outcome") == "wrong_verdict"
                                             and solo.get("outcome") == "correct")})
        rounds.append({"trial": trial, "expected_slots": len(keys),
            "observed_slots": sum(key in slots for key in keys),
            "all_slots_terminal": not missing and not pending,
            "missing_slots": missing, "pending_slots": pending,
            "correct_labels": correct, "accuracy_denominator_per_arm": len(cases),
            "outcomes": {arm: dict(counts[arm]) for arm in ARMS},
            "debate_minus_solo_correct": correct["debate"] - correct["solo"],
            "unauditable_label_slots": unauditable,
            "accuracy_labels_auditable": not unauditable})

    first, second = rounds
    r1_corrections = [row["case"] for row in comparisons if row["trial"] == trials[0]
                      and row["debate_corrects_solo"]]
    replicated = [case for case in cases if all(any(row["trial"] == trial and row["case"] == case
        and row["debate_corrects_solo"] for row in comparisons) for trial in trials)]
    activates = bool(first["all_slots_terminal"] and first["accuracy_labels_auditable"]
                     and first["debate_minus_solo_correct"] >= 2 and r1_corrections)
    full = all(row["all_slots_terminal"] for row in rounds)
    both_gain = all(row["debate_minus_solo_correct"] > 0 for row in rounds)
    return comparisons, {"rounds": rounds, "r1_activates_r2": activates,
        "r1_explicit_correction_cases": r1_corrections, "replicated_cases": replicated,
        "all_two_round_slots_terminal": full, "both_rounds_positive_accuracy_gain": both_gain,
        "replicated_advantage_numerical_gate": bool(activates and full and both_gain and replicated
            and second["accuracy_labels_auditable"]),
        "additional_requirement": "Independently audit the decisive runtime evidence before claiming a replicated fresh-seed advantage.",
        "scope": "Fresh seeds within an adaptively selected mechanism; no general superiority claim and no further batch is scheduled."}


def derive():
    result = base_report.derive(cases=CASES, trials=TRIALS)
    result["comparisons"], result["gate"] = transfer_gate(result["rows"])
    result["protocol"] = "PRECISION_TRANSFER_PROTOCOL.md"
    result["designated_cases"], result["designated_trials"] = list(CASES), list(TRIALS)
    return result


def render(result):
    lines = ["# Precision-reference transfer: case_76–case_81", "", f"Generated: {result['generated_at']}", "",
        "Follow the [prospective protocol](PRECISION_TRANSFER_PROTOCOL.md). This is a conditional fresh-seed confirmation within a mechanism selected on development cases.", "",
        "All arms share GLM low and a 32768 cumulative output-token ceiling. Input tokens, actual spend and GPU time are not matched. Every attempted slot remains visible; extra trials cannot replace designated slots.", "",
        "| Trial | Arm | Attempts | Correct labels | Outcomes | Recorded API estimate |",
        "|---|---|---:|---:|---|---:|"]
    for group in result["groups"]:
        denominator = len(CASES) if group["trial"] in TRIALS else group["attempts"]
        correct = group["outcomes"].get("correct", 0)
        lines.append(f"| {group['trial']} | {group['arm']} | {group['attempts']} | {correct}/{denominator} | {json.dumps(group['outcomes'])} | ${group['api_estimate_usd']:.6f} |")
    lines += ["", "The first round activates the full second round only after all 18 slots terminate, with at least two additional correct debate labels and at least one valid explicit solo-error correction. Replication requires positive accuracy gain in both rounds and the same-case explicit correction in both.", "",
        "Abstentions and failures remain in the fixed accuracy denominator and are not explicit reasoning mistakes. Numerical gates additionally require independent evidence review; this report schedules no further batch.", "",
        "```json", json.dumps(result["gate"], indent=2), "```", "",
        "| Case | Trial | Arm | Truth | Verdict | Outcome | Evidence | Audit issues |",
        "|---|---|---|---|---|---|---|---|"]
    for row in result["rows"]:
        lines.append(f"| {row['case']} | {row['trial']} | {row['arm']} | {row.get('truth')} | {row.get('verdict')} | {row['outcome']} | [trace]({row['trace_link']}) | {', '.join(row['issues'])} |")
    lines += ["", f"Recorded API estimate: ${result['api_estimate_usd']:.6f}; unknown-cost attempts: {result['unknown_cost_attempts']}; partial-cost attempts: {result['partial_cost_attempts']}.",
        "Excluded: Modal GPU charges and usage not returned by failed provider calls. Provider errors, budget exhaustion, abstention and explicit mistakes are separate outcomes."]
    return "\n".join(lines) + "\n"


def main(*, export_json=False):
    result = derive()
    if export_json:
        output = ROOT / "private_data" / "reports" / "precision_transfer_scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "PRECISION_TRANSFER_REPORT.md").write_text(render(result))
    print(json.dumps({"groups": result["groups"], "gate": result["gate"],
                      "api_estimate_usd": result["api_estimate_usd"]}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    main(export_json=parser.parse_args().json)
