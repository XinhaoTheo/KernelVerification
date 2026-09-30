"""The evidence pilot must retain fixed slots and require auditable raw calls."""
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/evidence_challenges/report.py"
SPEC = importlib.util.spec_from_file_location("evidence_report_test", SOURCE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def raw_call(path, number, *, limit, used, finish="stop"):
    directory = path / "llm_calls" / f"{number:03}"
    write(directory / "request.json", {"model": report.MODEL, "reasoning_effort": "low", "max_tokens": limit})
    write(directory / "response.json", {"usage": {"prompt_tokens": 100, "completion_tokens": used},
                                        "choices": [{"finish_reason": finish}]})
    write(directory / "metadata.json", {"status": "completed", "response_saved": True})
    return directory


def row_inputs(path, *, calls=2):
    row = {"case": "case_62", "arm": "solo", "trial": report.TRIALS[0],
           "model": report.MODEL, "provider": "fireworks", "path": str(path),
           "truth": "trust", "verdict": "reject", "outcome": "wrong_verdict",
           "usage_coverage": {"status": "complete", "captured_calls": calls,
                              "responses_saved": calls, "responses_with_usage": calls}}
    metadata = {"status": "completed", "total_output_token_budget": report.CAP,
                "kernel_sha256": "kernel", "problem_sha256": "problem"}
    frozen = {"case_62": {"kernel_sha256": "kernel", "problem_sha256": "problem"}}
    return row, metadata, frozen


def audit(path, row, metadata, frozen):
    return report.audit_row(row, metadata, path, frozen)


def test_raw_requests_must_shrink_by_actual_output_usage(tmp_path):
    first = raw_call(tmp_path, 1, limit=report.CAP, used=1234)
    second = raw_call(tmp_path, 2, limit=report.CAP - 1234, used=5678)
    row, meta, frozen = row_inputs(tmp_path)
    result = audit(tmp_path, row, meta, frozen)
    assert result["valid_complete"]
    assert result["recorded_output_budget_used"] == 6912
    request = report.read(second / "request.json")
    request["max_tokens"] = report.CAP
    write(second / "request.json", request)
    broken = audit(tmp_path, row, meta, frozen)
    assert not broken["valid_complete"]
    assert any("budget mismatch" in issue for issue in broken["issues"])


@pytest.mark.parametrize("missing", ["request.json", "response.json", "metadata.json"])
def test_complete_usage_summary_cannot_hide_missing_raw_files(tmp_path, missing):
    directory = raw_call(tmp_path, 1, limit=report.CAP, used=100)
    (directory / missing).unlink()
    row, meta, frozen = row_inputs(tmp_path, calls=1)
    result = audit(tmp_path, row, meta, frozen)
    assert not result["valid_complete"]
    assert result["issues"]


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 10},
                                   {"prompt_tokens": 10, "completion_tokens": -1}])
def test_missing_or_invalid_usage_is_not_zero_spend(tmp_path, usage):
    directory = raw_call(tmp_path, 1, limit=report.CAP, used=100)
    response = report.read(directory / "response.json")
    response["usage"] = usage
    write(directory / "response.json", response)
    raw_call(tmp_path, 2, limit=report.CAP, used=100)
    row, meta, frozen = row_inputs(tmp_path)
    result = audit(tmp_path, row, meta, frozen)
    assert not result["valid_complete"]
    assert any("invalid usage" in issue for issue in result["issues"])
    assert any("budget mismatch" in issue for issue in result["issues"])


@pytest.mark.parametrize("error,outcome", [
    ("OutputTokenBudgetExhausted: used all", "budget_exhaustion"),
    ("OutputTokenBudgetAccountingError: missing usage", "budget_accounting_error"),
    ("openai.InternalServerError: 504", "transport_error"),
    ("RuntimeError: agent protocol", "runner_error"),
])
def test_runner_errors_override_stale_explicit_verdicts(tmp_path, error, outcome):
    raw_call(tmp_path, 1, limit=report.CAP, used=100)
    row, meta, frozen = row_inputs(tmp_path, calls=1)
    meta["status"] = "error"
    row["runner_error"] = error
    result = audit(tmp_path, row, meta, frozen)
    assert result["outcome"] == outcome
    assert not result["valid_complete"]
    assert result["correct"] is False


def test_truncated_final_json_is_token_limit_not_wrong_verdict(tmp_path):
    raw_call(tmp_path, 1, limit=report.CAP, used=report.CAP, finish="length")
    row, meta, frozen = row_inputs(tmp_path, calls=1)
    result = audit(tmp_path, row, meta, frozen)
    assert result["outcome"] == "token_limit"


def pilot_rows():
    return [{"trial": trial, "case": case, "arm": arm, "status": "completed",
             "outcome": "wrong_verdict" if case == report.CASES[0] and arm == "solo" else "correct",
             "valid_complete": True}
            for trial in report.TRIALS for case in report.CASES for arm in report.ARMS]


def slot(rows, trial, case, arm):
    return next(row for row in rows if (row["trial"], row["case"], row["arm"]) == (trial, case, arm))


def gate(rows):
    return report.pilot_gate(rows)[1]


def test_same_case_must_replicate_in_designated_rounds():
    rows = pilot_rows()
    assert gate(rows)["expansion_numerical_gate"]
    assert gate(rows)["replicated_cases"] == [report.CASES[0]]
    slot(rows, report.TRIALS[1], report.CASES[0], "solo")["outcome"] = "correct"
    slot(rows, report.TRIALS[1], report.CASES[1], "solo")["outcome"] = "wrong_verdict"
    result = gate(rows)
    assert result["net_paired_improvements"] == 2
    assert result["replicated_cases"] == []
    assert not result["expansion_numerical_gate"]


@pytest.mark.parametrize("outcome", ["abstention", "token_limit", "budget_exhaustion", "transport_error", "no_verdict"])
def test_unanswered_attempts_do_not_activate_r2(outcome):
    rows = pilot_rows()
    slot(rows, report.TRIALS[0], report.CASES[0], "solo")["outcome"] = outcome
    assert not gate(rows)["r1_activates_r2"]
    assert not gate(rows)["expansion_numerical_gate"]


def test_partial_capture_cannot_establish_correction():
    rows = pilot_rows()
    slot(rows, report.TRIALS[0], report.CASES[0], "debate")["valid_complete"] = False
    assert not gate(rows)["r1_activates_r2"]
    assert not gate(rows)["expansion_numerical_gate"]


def test_reverse_errors_are_counted_against_improvements():
    rows = pilot_rows()
    for trial in report.TRIALS:
        slot(rows, trial, report.CASES[1], "debate")["outcome"] = "wrong_verdict"
    result = gate(rows)
    assert result["replicated_cases"] == [report.CASES[0]]
    assert result["net_paired_improvements"] == 0
    assert not result["expansion_numerical_gate"]


def test_failed_slot_cannot_be_replaced_with_retry():
    rows = pilot_rows()
    failed = slot(rows, report.TRIALS[1], report.CASES[0], "solo")
    rows.append({**failed, "trial": "ea_pilot_retry"})
    failed.update(status="error", outcome="transport_error", valid_complete=False)
    assert not gate(rows)["expansion_numerical_gate"]
    assert gate(rows)["all_two_round_slots_terminal"]


def test_r1_waits_for_all_12_slots_and_expansion_waits_for_all_24():
    rows = pilot_rows()
    pending = slot(rows, report.TRIALS[0], report.CASES[-1], "single_call")
    pending.update(status="running", outcome="pending", valid_complete=False)
    assert not gate(rows)["r1_activates_r2"]
    assert not gate(rows)["expansion_numerical_gate"]
    rows.remove(pending)
    assert not gate(rows)["all_two_round_slots_terminal"]


def test_duplicate_designated_slots_fail_instead_of_overwriting():
    rows = pilot_rows()
    with pytest.raises(ValueError, match="Duplicate designated slot"):
        gate(rows + [rows[0]])


def test_new_cohort_cannot_borrow_old_case_or_repeat_slots():
    cases = ("case_66", "case_67")
    trials = ("ea_methods_v2_r1", "ea_methods_v2_r2")
    rows = [{"trial": trial, "case": case, "arm": arm, "status": "completed",
             "outcome": "wrong_verdict" if case == cases[0] and arm == "solo" else "correct",
             "valid_complete": True}
            for trial in trials for case in cases for arm in report.ARMS]
    result = report.pilot_gate(pilot_rows() + rows, cases=cases, trials=trials)[1]
    assert result["replicated_cases"] == ["case_66"]
    assert result["net_paired_improvements"] == 2
    assert result["expansion_numerical_gate"]
    slot(rows, trials[1], cases[0], "solo")["trial"] = report.TRIALS[1]
    result = report.pilot_gate(pilot_rows() + rows, cases=cases, trials=trials)[1]
    assert result["r1_activates_r2"]
    assert not result["expansion_numerical_gate"]


def test_derive_filters_dataset_and_keeps_failure_coverage(tmp_path, monkeypatch):
    root = tmp_path / "evidence_challenges"
    case = root / "eval_cases" / "case_62"
    case.mkdir(parents=True)
    frozen = {"case_62": {"ground_truth": "trust"}}
    for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt")):
        (case / filename).write_text("frozen material")
        frozen["case_62"][f"{kind}_sha256"] = report.hashlib.sha256(b"frozen material").hexdigest()
    write(root / "private_data" / "validation_gpu.json", {"cases": frozen})
    path = tmp_path / "traces_glm" / "case_62" / "solo" / report.TRIALS[0]
    metadata = {"case": "case_62", "arm": "solo", "trial": report.TRIALS[0],
                "dataset": "evidence_challenges", "provider": "fireworks", "model": report.MODEL,
                "status": "error", "total_output_token_budget": report.CAP, **frozen["case_62"]}
    write(path / "trace_meta.json", metadata)
    write(path / "run.json", {"verdict": {"verdict": "reject"}})
    raw_call(path, 1, limit=report.CAP, used=100)
    second = raw_call(path, 2, limit=report.CAP - 100, used=10)
    (second / "response.json").unlink()
    write(second / "metadata.json", {"status": "error", "response_saved": False})
    (path / "runner_error.txt").write_text("openai.InternalServerError: offline 504")
    unrelated = tmp_path / "traces_glm" / "case_other" / "solo" / "unrelated"
    write(unrelated / "trace_meta.json", {**metadata, "case": "case_other", "dataset": "numerical_challenges"})
    monkeypatch.setattr(report, "ROOT", root)
    result = report.derive()
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["outcome"] == "transport_error"
    assert row["recorded_output_budget_used"] == 100
    assert row["usage_coverage"]["status"] == "partial"
    assert not row["valid_complete"]
    assert result["partial_cost_attempts"] == 1
    assert result["unknown_cost_attempts"] == 0
    assert not result["gate"]["r1_activates_r2"]
