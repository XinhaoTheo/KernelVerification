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


def test_archived_cases_leave_active_coverage_and_collection_groups_pair_with_numerical(index, tmp_path, monkeypatch):
    collection = "single_call_vs_tools_challenges"
    details = {
        "case_36": {"dataset": "correlation_pair", "collection": collection},
        "case_38": {"dataset": "numerical_challenges", "collection": collection},
        "case_82": {"dataset": "numerical_pilot", "status": "archived"},
    }
    rows = [row(case=case, dataset=detail["dataset"]) for case, detail in details.items()]
    coverage = index.build_coverage(rows, details, {})
    assert [c["case"] for c in coverage["cases"]] == ["case_36", "case_38"]
    assert {c["collection"] for c in coverage["cases"]} == {collection}
    assert coverage["totals"]["solo"]["completed"] == 2
    assert coverage["totals"]["debate"]["missing"] == 2
    assert coverage["cases"][0]["arms"]["solo"]["selected"] == rows[0]
    (tmp_path / "traces_glm").mkdir()
    archived = tmp_path / "archive/numerical_pilot/traces_glm/case_82/solo/r1"
    archived.mkdir(parents=True)
    (archived / "trace_meta.json").write_text("{}")
    records = [{"path": tmp_path / r["path"], "metadata": {}} for r in rows]
    monkeypatch.setattr(index, "BENCHMARK", tmp_path)
    monkeypatch.setattr(index, "iter_trace_records", lambda **kwargs: iter(records))
    monkeypatch.setattr(index, "load_registry", lambda root: {"case_details": details})
    monkeypatch.setattr(index, "build_report", lambda **kwargs: {
        "arms": {"fixture": {"per_case": {str(i): r for i, r in enumerate(rows)}}}})
    index.main()
    text = (tmp_path / "traces_glm/INDEX.md").read_text()
    assert "2 个活跃案例" in text and "6 个 case-arm 槽位" in text
    assert "1 题、1 条 GLM trials" in text
    assert "../archive/numerical_pilot/README.md" in text
    assert "case_82" not in text
    assert text.count("### " + index.DATASET_LABELS.get(collection, collection)) == 1
    assert "| correlation_pair |" in text and "| numerical_challenges |" in text
    assert "2 recorded GLM trials" in text


def test_each_archive_has_its_own_counts_and_case_links(index, tmp_path, monkeypatch):
    import json
    import index_cases

    benchmark = tmp_path / "benchmark_fn_fp"
    details = {
        "case_01": {"dataset": "benchmark_fn_fp", "public_dir": "triton_eval_cases/case_01"},
        "case_34": {"dataset": "benchmark_fn_fp", "status": "archived",
                    "archive_root": "archive/input_specific",
                    "public_dir": "archive/input_specific/triton_eval_cases/case_34"},
        "case_82": {"dataset": "numerical_pilot", "status": "archived",
                    "archive_root": "archive/numerical_pilot",
                    "public_dir": "archive/numerical_pilot/eval_cases/case_82"},
    }
    for case, detail in details.items():
        public = benchmark / detail["public_dir"]
        public.mkdir(parents=True)
        (public / "kernel.py").write_text("# frozen")
        prefix = detail.get("archive_root", "")
        trace = benchmark / prefix / "traces_glm" / case / "solo/r1"
        trace.mkdir(parents=True)
        (trace / "trace_meta.json").write_text("{}")
    (benchmark / "case_map.json").write_text(json.dumps({"case_details": details}))
    monkeypatch.setattr(index_cases, "ROOT", benchmark)
    index_cases.main()
    case_text = (benchmark / "CASE_INDEX.md").read_text()
    assert "当前 1 题；另有 2 题已归档" in case_text
    assert "(archive/input_specific/traces_glm/case_34/)" in case_text
    assert "(archive/numerical_pilot/traces_glm/case_82/)" in case_text
    assert "(archive/numerical_pilot/traces_glm/case_34/)" not in case_text

    current = row()
    monkeypatch.setattr(index, "BENCHMARK", benchmark)
    monkeypatch.setattr(index, "iter_trace_records", lambda **kwargs: iter([
        {"path": benchmark / current["path"], "metadata": {}}]))
    monkeypatch.setattr(index, "build_report", lambda **kwargs: {
        "arms": {"fixture": {"per_case": {"case_01": current}}}})
    index.main()
    trace_text = (benchmark / "traces_glm/INDEX.md").read_text()
    assert trace_text.count("1 题、1 条 GLM trials") == 2
    assert "../archive/input_specific/README.md" in trace_text
    assert "../archive/numerical_pilot/README.md" in trace_text
    assert "1 recorded GLM trials" in trace_text


def test_case_index_ranges_exclude_archive_gap_when_new_cases_are_added(index, tmp_path, monkeypatch):
    import json
    import index_cases

    benchmark = tmp_path / "benchmark_fn_fp"
    details = {}
    for number in range(1, 169):
        case = f"case_{number:02}"
        archived = 91 <= number <= 162
        prefix = "archive/input_specific/" if archived else ""
        detail = {
            "dataset": "benchmark_fn_fp" if number <= 32 else "real_kernel_challenges",
            "public_dir": f"{prefix}triton_eval_cases/{case}",
        }
        if archived:
            detail.update(status="archived", archive_root="archive/input_specific")
        details[case] = detail
        public = benchmark / detail["public_dir"]
        public.mkdir(parents=True)
        (public / "kernel.py").write_text("# frozen")
    registry = benchmark / "case_map.json"
    registry.write_text(json.dumps({"case_details": details, "next_case_number": 169}))
    before = registry.read_bytes()
    monkeypatch.setattr(index_cases, "ROOT", benchmark)

    index_cases.main()

    text = (benchmark / "CASE_INDEX.md").read_text()
    assert "当前 96 题；另有 72 题已归档" in text
    assert "当前编号：`case_01`–`case_90`、`case_163`–`case_168`。" in text
    assert "| `case_01`–`case_32` | 32 |" in text
    assert "| `case_33`–`case_90`、`case_163`–`case_168` | 64 |" in text
    assert "`case_33`–`case_168`" not in text
    assert "当前题连续编号" not in text
    assert registry.read_bytes() == before


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
