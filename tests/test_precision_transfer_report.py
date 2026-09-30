"""The fresh-seed gate uses fixed-cohort accuracy, not just discordant errors."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/evidence_challenges/report_precision_transfer.py"
SPEC = importlib.util.spec_from_file_location("precision_transfer_report_test", SOURCE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def cohort():
    return [{"trial": trial, "case": case, "arm": arm, "status": "completed",
             "outcome": "wrong_verdict" if case in report.CASES[:2] and arm == "solo" else "correct",
             "valid_complete": True}
            for trial in report.TRIALS for case in report.CASES for arm in report.ARMS]


def slot(rows, trial, case, arm):
    return next(row for row in rows if (row["trial"], row["case"], row["arm"]) == (trial, case, arm))


def gate(rows):
    return report.transfer_gate(rows)[1]


@pytest.mark.parametrize("state", ["missing", "running"])
def test_r1_waits_for_every_arm_of_all_six_cases(state):
    rows = cohort()
    pending = slot(rows, report.TRIALS[0], report.CASES[-1], "single_call")
    if state == "missing":
        rows.remove(pending)
    else:
        pending.update(status="running", outcome="pending", valid_complete=False)
    assert not gate(rows)["r1_activates_r2"]
    assert not gate(rows)["replicated_advantage_numerical_gate"]


def test_r1_requires_two_accuracy_gain_and_an_explicit_correction():
    rows = cohort()
    assert gate(rows)["r1_activates_r2"]
    slot(rows, report.TRIALS[0], report.CASES[1], "solo")["outcome"] = "correct"
    assert gate(rows)["rounds"][0]["debate_minus_solo_correct"] == 1
    assert not gate(rows)["r1_activates_r2"]
    slot(rows, report.TRIALS[0], report.CASES[1], "solo")["outcome"] = "abstention"
    assert gate(rows)["r1_activates_r2"]  # One explicit correction plus one abstention gain.
    slot(rows, report.TRIALS[0], report.CASES[0], "solo")["outcome"] = "abstention"
    result = gate(rows)
    assert result["rounds"][0]["debate_minus_solo_correct"] == 2
    assert result["r1_explicit_correction_cases"] == []
    assert not result["r1_activates_r2"]


@pytest.mark.parametrize("outcome", ["transport_error", "budget_exhaustion", "token_limit", "abstention"])
def test_failures_and_abstentions_cannot_substitute_for_explicit_error(outcome):
    rows = cohort()
    for case in report.CASES[:2]:
        slot(rows, report.TRIALS[0], case, "solo").update(outcome=outcome, valid_complete=False,
            status="completed" if outcome in {"token_limit", "abstention"} else "error")
    result = gate(rows)
    assert result["rounds"][0]["all_slots_terminal"]
    assert result["rounds"][0]["outcomes"]["solo"][outcome] == 2
    assert result["r1_explicit_correction_cases"] == []
    assert not result["r1_activates_r2"]


def test_replication_needs_same_case_and_full_second_round():
    rows = cohort()
    result = gate(rows)
    assert result["replicated_advantage_numerical_gate"]
    assert result["replicated_cases"] == list(report.CASES[:2])
    for case in report.CASES[:2]:
        slot(rows, report.TRIALS[1], case, "solo")["outcome"] = "correct"
    for case in report.CASES[2:4]:
        slot(rows, report.TRIALS[1], case, "solo")["outcome"] = "wrong_verdict"
    result = gate(rows)
    assert result["both_rounds_positive_accuracy_gain"]
    assert result["replicated_cases"] == []
    assert not result["replicated_advantage_numerical_gate"]
    pending = slot(rows, report.TRIALS[1], report.CASES[-1], "single_call")
    pending.update(status="running", outcome="pending")
    assert not gate(rows)["all_two_round_slots_terminal"]


def test_repeated_explicit_correction_does_not_override_accuracy_tie():
    rows = cohort()
    for case in report.CASES[2:4]:
        slot(rows, report.TRIALS[1], case, "debate")["outcome"] = "abstention"
    result = gate(rows)
    assert result["replicated_cases"] == list(report.CASES[:2])
    assert result["rounds"][1]["debate_minus_solo_correct"] == 0
    assert not result["both_rounds_positive_accuracy_gain"]
    assert not result["replicated_advantage_numerical_gate"]


def test_invalid_labels_cannot_manufacture_accuracy_gain():
    rows = cohort()
    # Keep a separate valid explicit correction: the invalid second label must
    # still prevent declaring a two-correct-label accuracy gain.
    slot(rows, report.TRIALS[0], report.CASES[1], "solo")["valid_complete"] = False
    result = gate(rows)
    assert result["r1_explicit_correction_cases"] == [report.CASES[0]]
    assert not result["r1_activates_r2"]


def test_retries_cannot_fill_missing_slots_and_duplicates_are_rejected():
    rows = cohort()
    with pytest.raises(ValueError, match="Duplicate designated slot"):
        gate(rows + [rows[0]])
    missing = slot(rows, report.TRIALS[0], report.CASES[-1], "single_call")
    missing["trial"] = "ea_precision_transfer_retry"
    assert not gate(rows)["r1_activates_r2"]


def test_derive_reuses_audited_data_but_replaces_legacy_gate(monkeypatch):
    rows = cohort()
    seen = []
    def audited_derive(*, cases, trials):
        seen.append((cases, trials))
        return {"rows": rows, "gate": {"expansion_numerical_gate": "legacy"},
                "comparisons": [], "api_estimate_usd": 0.42}
    monkeypatch.setattr(report.base_report, "derive", audited_derive)
    result = report.derive()
    assert seen == [(report.CASES, report.TRIALS)]
    assert "expansion_numerical_gate" not in result["gate"]
    assert result["gate"]["replicated_advantage_numerical_gate"]
    assert result["api_estimate_usd"] == 0.42
