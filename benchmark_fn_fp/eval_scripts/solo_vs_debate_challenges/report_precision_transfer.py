"""Report the prospective case_76–case_81 accuracy-based confirmation protocol.

Trace validity, costs and outcomes come from the existing audited report.
Only this cohort's numerical gate differs: accuracy gains must survive
abstentions as well as reverse errors, with a repeated explicit correction.
"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "solo_vs_debate_challenges"
sys.path.insert(0, str(ROOT.parents[1]))
from benchmark_fn_fp.eval_scripts.solo_vs_debate_challenges import report as base_report

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
                  and base_report.experiment_trial(row) in trials and row.get("arm") in ARMS]
    slots = {(base_report.experiment_trial(row), row["case"], row["arm"]): row for row in designated}
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
    result["protocol"] = "README.md"
    result["designated_cases"], result["designated_trials"] = list(CASES), list(TRIALS)
    return result


def render(result):
    gate = result["gate"]
    lines = [f"生成时间：{result['generated_at']}；题目范围：case_76–case_81。", "",
        base_report.compact_results(result["rows"]),
        f"新随机种子复验门槛： **{'通过' if gate['replicated_advantage_numerical_gate'] else '未达到'}**; "
        f"复验中重复纠错的题目： {', '.join(gate['replicated_cases']) or 'none'}.", "",
        "| 指定批次 | 无工具正确 | Solo 正确 | Debate 正确 | 每组固定分母 | Debate − solo | 全部结束 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in gate["rounds"]:
        count = row["correct_labels"]
        lines.append(f"| {row['trial']} | {count['single_call']} | {count['solo']} | {count['debate']} | "
                     f"{row['accuracy_denominator_per_arm']} | {row['debate_minus_solo_correct']} | {row['all_slots_terminal']} |")
    lines += ["", "弃答与失败保留在固定分母中。这是自适应选定机制内的新随机种子测试，"
              "不代表广泛泛化；数值门槛通过后仍需独立审核证据。"]
    return "\n".join(lines) + "\n"


def main(*, export_json=False):
    result = derive()
    base_report.replace_results(ROOT, "PRECISION TRANSFER RESULTS", render(result))
    if export_json:
        output = ROOT / "private_data" / "reports" / "precision_transfer_scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"groups": result["groups"], "gate": result["gate"],
                      "api_estimate_usd": result["api_estimate_usd"]}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    main(export_json=parser.parse_args().json)
