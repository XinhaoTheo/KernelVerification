"""Coverage counts completed attempts without cherry-picking correct results."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def index(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/eval_scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("trace_coverage_index_test", scripts / "index_traces.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(case="case_01", arm="solo", trial="r1", verdict="trust", outcome="correct", **extra):
    return {"case": case, "arm": arm, "trial": trial, "verdict": verdict, "outcome": outcome,
            "dataset": "benchmark_fn_fp", "path": f"traces_glm/{case}/{arm}/{trial}",
            "model": "accounts/fireworks/models/glm-5p3", "provider": "fireworks", **extra}


def test_counts_the_whole_registry_including_cases_without_traces(index):
    details = {f"case_{i:02}": {"dataset": "benchmark_fn_fp"}
               for i in range(1, 106) if i != 3}
    wrong = row(outcome="wrong_verdict")
    coverage = index.build_coverage([wrong], details, {})
    assert len(coverage["cases"]) == 104
    assert coverage["totals"]["solo"] == {
        "completed": 1, "missing": 103, "failed_or_unfinished": 0, "running": 0}
    assert coverage["totals"]["single_call"]["missing"] == 104
    assert coverage["totals"]["debate"]["missing"] == 104
    assert coverage["cases"][-1]["case"] == "case_105"


def test_selects_first_valid_attempt_even_if_wrong_and_later_one_correct(index):
    wrong = row(trial="earlier", outcome="wrong_verdict")
    correct = row(trial="later")
    failed = row(trial="first_failed", verdict=None, outcome="no_verdict")
    metadata = {
        wrong["path"]: {"status": "completed", "created_at": "2026-09-20T00:00:00Z"},
        correct["path"]: {"status": "completed", "created_at": "2026-09-21T00:00:00Z"},
        failed["path"]: {"status": "error", "created_at": "2026-09-19T00:00:00Z"},
    }
    coverage = index.build_coverage([correct, failed, wrong], {"case_01": {"dataset": "benchmark_fn_fp"}}, metadata)
    slot = coverage["cases"][0]["arms"]["solo"]
    assert slot["selected"] == wrong
    assert slot["attempts"] == 3


def test_legacy_absent_metadata_and_abstention_are_completed(index):
    legacy = row(trial="legacy", verdict="needs_more_evidence", outcome="abstention",
                 model="z-ai/glm-5.3-flash", provider="openrouter")
    newer = row(trial="new")
    coverage = index.build_coverage([newer, legacy], {"case_01": {"dataset": "benchmark_fn_fp"}},
        {newer["path"]: {"created_at": "2026-09-30T00:00:00Z", "status": "completed"}})
    assert coverage["cases"][0]["arms"]["solo"]["selected"] == legacy


def test_numeric_rename_preserves_original_coverage_tiebreak(index):
    earlier = row(trial="r10", outcome="wrong_verdict")
    later = row(trial="r2")
    metadata = {
        earlier["path"]: {"selection_sort_key": ["", "traces_glm/case_01/solo/aaa"]},
        later["path"]: {"selection_sort_key": ["", "traces_glm/case_01/solo/zzz"]},
    }
    coverage = index.build_coverage([later, earlier], {"case_01": {"dataset": "benchmark_fn_fp"}}, metadata)
    assert coverage["cases"][0]["arms"]["solo"]["selected"] == earlier
    assert sorted(["r10", "r2", "r1"], key=index.trial_sort_key) == ["r1", "r2", "r10"]


@pytest.mark.parametrize("overrides, metadata", [
    ({"outcome": "token_limit"}, {}),
    ({"outcome": "no_verdict"}, {}),
    ({"runner_error": "APIConnectionError"}, {}),
    ({"outcome": "runner_error"}, {}),
    ({}, {"status": "error"}),
    ({}, {"status": "running"}),
    ({}, {"status": "cancelled"}),
    ({}, {"stop_reason": "length"}),
])
def test_interrupted_attempts_do_not_count_as_completed(index, overrides, metadata):
    assert not index.completed(row(**overrides), metadata)


def test_running_and_failed_are_separate_from_missing_and_history_is_retained(index, tmp_path, monkeypatch):
    failed = row("case_01", "single_call", "failed", verdict=None, outcome="no_verdict")
    running = row("case_01", "solo", "in_progress")
    abstention = row("case_01", "debate", "first", verdict="needs_more_evidence", outcome="abstention")
    later = row("case_01", "debate", "second")
    rows = [failed, running, abstention, later]
    details = {"case_01": {"dataset": "benchmark_fn_fp"}, "case_02": {"dataset": "benchmark_fn_fp"}}
    metadata = {
        failed["path"]: {"status": "error"},
        running["path"]: {"status": "running"},
        abstention["path"]: {"status": "completed", "created_at": "2026-09-20"},
        later["path"]: {"status": "completed", "created_at": "2026-09-21"},
    }
    coverage = index.build_coverage(rows, details, metadata)
    assert coverage["totals"]["single_call"] == {
        "completed": 0, "missing": 1, "failed_or_unfinished": 1, "running": 0}
    assert coverage["totals"]["solo"] == {
        "completed": 0, "missing": 1, "failed_or_unfinished": 0, "running": 1}
    records = [{"path": tmp_path / r["path"], "metadata": metadata[r["path"]]} for r in rows]
    monkeypatch.setattr(index, "BENCHMARK", tmp_path)
    monkeypatch.setattr(index, "iter_trace_records", lambda **kwargs: iter(records))
    monkeypatch.setattr(index, "load_registry", lambda root: {"case_details": details})
    monkeypatch.setattr(index, "build_report", lambda **kwargs: {
        "arms": {"fixture": {"per_case": {str(i): r for i, r in enumerate(rows)}}}})
    (tmp_path / "traces_glm").mkdir()
    index.main()
    text = (tmp_path / "traces_glm/INDEX.md").read_text()
    coverage_text, history = text.split("## 全部历史 trials")
    assert "case_02" in coverage_text and "| single_call | 0 | 1 | 1 | 0 | 2 |" in coverage_text
    assert "[first]" in coverage_text and "[second]" not in coverage_text
    assert "z-ai/glm-5.3-flash" in coverage_text and "accounts/fireworks/models/glm-5p3" in coverage_text
    assert "16 条已完成旧记录" in coverage_text
    assert history.count("[open]") == 4
    assert "4 recorded GLM trials" in history


def test_enabled_capture_does_not_hide_missing_response_or_usage(index, tmp_path):
    record = row(usage_coverage={"status": "partial"})
    metadata = {record["path"]: {"status": "completed", "raw_api_capture": True}}
    call = tmp_path / record["path"] / "llm_calls/001"
    call.mkdir(parents=True)
    (call / "request.json").write_text("{}")
    coverage = index.build_coverage([record], {"case_01": {"dataset": "benchmark_fn_fp"}}, metadata)
    assert coverage["totals"]["solo"]["completed"] == 1
    lines = "\n".join(index._coverage_lines(coverage, metadata, tmp_path))
    assert "选用记录中 1 条没有可确认的完整原始 API capture" in lines
    (call / "response.json").write_text("{}")
    assert not index._raw_capture_complete(record, tmp_path)  # usage remains partial


def test_complete_usage_still_requires_the_saved_request(index, tmp_path):
    record = row(usage_coverage={"status": "complete"})
    call = tmp_path / record["path"] / "llm_calls/001"
    call.mkdir(parents=True)
    (call / "response.json").write_text("{}")
    assert not index._raw_capture_complete(record, tmp_path)
    (call / "request.json").write_text("{}")
    assert index._raw_capture_complete(record, tmp_path)


def test_legacy_single_with_original_request_response_is_still_captured(index, tmp_path):
    record = row(arm="single_call", trial="legacy")
    directory = tmp_path / record["path"]
    directory.mkdir(parents=True)
    (directory / "request.json").write_text("{}")
    (directory / "raw_response.json").write_text("{}")
    assert index._raw_capture_complete(record, tmp_path)
