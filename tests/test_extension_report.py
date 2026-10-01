"""The extension's fixed replication window must not promote missing evidence."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/eval_scripts/single_call_vs_tools_challenges/report_extension.py"
SPEC = importlib.util.spec_from_file_location("numerical_extension_report_test", SOURCE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _eligible_case():
    labels = {case: {"family": "test_family", "truth_status": "gpu_verified"}
              for case in report.CASES}
    rows = []
    for arm in report.ARMS:
        trials = report.LOW_TRIALS if arm == "single_call" else report.LOW_TRIALS[:2]
        for index, trial in enumerate(trials):
            rows.append({
                "case": "case_44", "arm": arm, "trial": trial, "status": "completed",
                "protocol_issues": [], "attempt_id": f"case_44/{arm}/{trial}",
                "outcome": "wrong_verdict" if arm == "single_call" and index < 2 else "correct",
                "request_configs": [{}], "usage_coverage": {
                    "status": "complete", "captured_calls": 1,
                    "responses_saved": 1, "responses_with_usage": 1,
                },
            })
    return rows, labels


def _passed(rows, labels):
    return report.replication_gate(rows, labels)["cases"]["case_44"]["passed"]


def test_numeric_trial_names_preserve_designated_extension_slots():
    rows, labels = _eligible_case()
    expected = report.replication_gate(rows, labels)
    renamed = [{**row, "original_trial": row["trial"], "trial": f"r{index + 2}"}
               for index, row in enumerate(rows)]
    assert report.replication_gate(renamed, labels) == expected
    # An unrelated batch called r1 cannot stand in for an original planned slot.
    renamed[0]["original_trial"] = "completion_20260930_r1"
    renamed[0]["trial"] = "r1"
    assert not _passed(renamed, labels)


@pytest.mark.parametrize("outcome", ["token_limit", "abstention", "no_verdict"])
def test_two_errors_with_unanswered_third_slot_cannot_qualify(outcome):
    rows, labels = _eligible_case()
    assert _passed(rows, labels)
    rows[2]["outcome"] = outcome
    assert not _passed(rows, labels)


@pytest.mark.parametrize("missing", ["responses_saved", "responses_with_usage", "request_configs"])
def test_incomplete_raw_evidence_cannot_qualify(missing):
    rows, labels = _eligible_case()
    if missing == "request_configs":
        rows[0][missing] = []
    else:
        rows[0]["usage_coverage"][missing] = 0
    assert not _passed(rows, labels)


def test_later_retry_cannot_replace_a_failed_fixed_slot():
    rows, labels = _eligible_case()
    retry = {**rows[0], "trial": "extension_low32_r4", "attempt_id": "retry"}
    rows[0]["status"], rows[0]["outcome"] = "error", "no_verdict"
    rows.append(retry)
    assert not _passed(rows, labels)


def test_cpu_only_evidence_cannot_qualify():
    rows, labels = _eligible_case()
    labels["case_44"]["truth_status"] = "cpu_only"
    assert not _passed(rows, labels)


def test_length_stop_with_a_json_verdict_remains_a_token_limit():
    from summarize_traces import classify_outcome
    assert classify_outcome({"verdict": "trust", "stop_reason": "length"}, "trust") == "token_limit"


def test_new_case_range_has_independent_fixed_trial_windows():
    old_rows, _ = _eligible_case()
    trials = ("oz_low32_r1", "oz_low32_r2", "oz_low32_r3")
    labels = {"case_50": {"family": "new_family", "truth_status": "gpu_verified"}}
    new_rows = [{**row, "case": "case_50", "trial": row["trial"].replace("extension_", "oz_")}
                for row in old_rows]
    # Earlier I--N outcomes do not fill an O--Z slot.
    missing = report.replication_gate(old_rows, labels, cases=("case_50",),
                                      low_trials=trials, required_families=1)
    assert not missing["passed"]
    result = report.replication_gate(old_rows + new_rows, labels, cases=("case_50",),
                                     low_trials=trials, required_families=1)
    assert result["passed"]
    assert result["qualifying_cases"] == ["case_50"]
    assert not report.replication_gate(new_rows, labels, cases=("case_50",),
                                       low_trials=trials, required_families=2)["passed"]


def test_generated_results_preserve_other_blocks_and_manual_text(tmp_path):
    from generated_results import replace_results

    before = "手工说明\n<!-- BEGIN GENERATED MAIN RESULTS -->\nmain\n<!-- END GENERATED MAIN RESULTS -->\n"
    after = "\n<!-- BEGIN GENERATED OZ RESULTS -->\noz\n<!-- END GENERATED OZ RESULTS -->\n尾部说明\n"
    path = tmp_path / "README.md"
    path.write_text(before + "<!-- BEGIN GENERATED EXTENSION RESULTS -->\nold\n<!-- END GENERATED EXTENSION RESULTS -->" + after)
    replace_results(tmp_path, "EXTENSION RESULTS", "new summary")
    assert path.read_text() == (before + "<!-- BEGIN GENERATED EXTENSION RESULTS -->\n\nnew summary\n\n<!-- END GENERATED EXTENSION RESULTS -->" + after)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("content", [
    "manual only",
    "<!-- BEGIN GENERATED EXTENSION RESULTS --><!-- BEGIN GENERATED EXTENSION RESULTS --><!-- END GENERATED EXTENSION RESULTS -->",
    "<!-- END GENERATED EXTENSION RESULTS --><!-- BEGIN GENERATED EXTENSION RESULTS -->",
])
def test_generated_results_refuse_missing_duplicate_or_reversed_markers(tmp_path, content):
    from generated_results import replace_results

    path = tmp_path / "README.md"
    path.write_text(content)
    with pytest.raises(ValueError, match="markers|marker pair"):
        replace_results(tmp_path, "EXTENSION RESULTS", "replacement")
    assert path.read_text() == content


def test_compact_summary_keeps_per_call_and_total_budgets_separate():
    from generated_results import compact_results

    row = {"trial": "r1", "original_trial": "initial", "provider": "fireworks",
           "model": "glm", "arm": "solo", "outcome": "correct", "usd": 0.1,
           "protocol": {"max_tokens": 32768, "total_output_token_budget": None}}
    capped = {**row, "protocol": {"max_tokens": 32768, "total_output_token_budget": 32768}}
    text = compact_results([row, capped])
    assert text.count("| initial |") == 2
    assert "每次 32768 / 总 unknown" in text
    assert "每次 32768 / 总 32768" in text
