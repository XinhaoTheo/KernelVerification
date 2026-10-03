"""Offline checks for reading repeated runs without mixing endpoints or labels."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _reader():
    path = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/eval_scripts/summarize_traces.py"
    spec = importlib.util.spec_from_file_location("trace_reader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _single(root, *, provider, model, trial, dataset="benchmark_fn_fp",
            verdict="trust", stop="stop", case="case_a"):
    dest = root / provider / dataset / case / "single_call" / trial
    dest.mkdir(parents=True)
    (dest / "usage.json").write_text(json.dumps({
        "usage": {"input_tokens": 1000, "output_tokens": 2000},
        "stop_reason": stop, "response": {"verdict": verdict, "confidence": 0.8},
    }))
    return {"path": dest, "case": case, "arm": "single_call", "trial": trial,
            "dataset": dataset, "model": model, "provider": provider, "metadata": {}}


def test_local_trial_numbers_do_not_merge_batches_or_budgets(tmp_path):
    reader = _reader()
    records = []
    configs = [
        ("case_01", "r1", "batch_a", 32768, 32768),
        ("case_02", "r1", "batch_b", 32768, 32768),
        ("case_04", "r2", "batch_a", 32768, 32768),
        ("case_05", "r1", "batch_a", 65536, 32768),
        ("case_06", "r1", "batch_a", 32768, 65536),
    ]
    for case, trial, batch, cap, total in configs:
        record = _single(tmp_path, provider="fireworks", model="accounts/fireworks/models/glm-5p3",
                         case=case, trial=trial)
        record["metadata"] = {"original_trial": batch, "max_tokens": cap,
                              "total_output_token_budget": total}
        records.append(record)
    report = reader.build_report(records=records, benchmark_dir=tmp_path, labels={})
    groups = list(report["arms"].values())
    assert len(groups) == 4
    paired = next(group for group in groups if group["attempts"] == 2)
    assert paired["original_trial"] == "batch_a"
    assert {row["trial"] for row in paired["per_case"].values()} == {"r1", "r2"}
    assert set(paired["per_case"]) == {"case_01", "case_04"}


def test_reader_retains_batch_and_chronology_after_numeric_rename(tmp_path):
    from benchmark_fn_fp.eval_scripts.traces import iter_trace_records, trace_selection_key
    leaf = tmp_path / "traces_glm/case_01/solo/r2"
    leaf.mkdir(parents=True)
    original = "traces_glm/case_01/solo/completion_20260930_r1"
    (leaf / "trace_meta.json").write_text(json.dumps({
        "case": "case_01", "arm": "solo", "trial": "r2", "original_trial": "completion_20260930_r1",
        "model": "accounts/fireworks/models/glm-5p3", "provider": "fireworks",
        "original_trace_path": original, "selection_sort_key": ["", original],
        "created_at": "2026-09-30T00:00:00Z"}))
    row, = iter_trace_records(tmp_path)
    assert row["trial"] == "r2"
    assert row["original_trial"] == "completion_20260930_r1"
    assert trace_selection_key(row) == ("", original)


def test_archived_pilot_is_opt_in_and_preserves_identity_truth_and_batches(tmp_path):
    from benchmark_fn_fp.eval_scripts.traces import iter_trace_records
    reader = _reader()
    details = {
        "case_36": {"dataset": "correlation_pair", "collection": "single_call_vs_tools_challenges"},
        "case_38": {"dataset": "numerical_challenges", "collection": "single_call_vs_tools_challenges"},
        "case_82": {"dataset": "numerical_pilot", "previous_id": "case_01", "status": "archived"},
    }
    (tmp_path / "case_map.json").write_text(json.dumps({"cases": {}, "case_details": details}))
    for case, detail in details.items():
        archived = detail.get("status") == "archived"
        base = tmp_path / "archive/numerical_pilot" if archived else tmp_path
        dest = base / "traces_glm" / case / "single_call/r1"
        dest.mkdir(parents=True)
        (dest / "trace_meta.json").write_text(json.dumps({
            "case": "case_01" if archived else case, "dataset": detail["dataset"],
            "arm": "single_call", "trial": "r1", "original_trial": "completion_batch",
            "model": "accounts/fireworks/models/glm-5p3", "provider": "fireworks",
            "status": "completed", "max_tokens": 32768,
            "pricing_snapshot": {"input_per_million": 1.4, "output_per_million": 4.4},
        }))
        (dest / "usage.json").write_text(json.dumps({
            "usage": {"input_tokens": 1000, "output_tokens": 2000},
            "stop_reason": "stop", "response": {"verdict": "trust"}}))
        truth_path = (base / "answer_key.json" if archived else
                      tmp_path / detail["dataset"] / "validation_gpu.json")
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(json.dumps({"cases": {case: {"ground_truth": "reject" if archived else "trust"}}}))
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert {r["case"] for r in iter_trace_records(tmp_path)} == {"case_36", "case_38"}
    all_records = list(iter_trace_records(tmp_path, include_archived=True))
    pilot, = [r for r in all_records if r["dataset"] == "numerical_pilot"]
    assert pilot["case"] == "case_82" and pilot["original_case"] == "case_01"
    assert pilot["metadata"]["case"] == "case_01"
    assert pilot["trial"] == "r1" and pilot["original_trial"] == "completion_batch"
    active = reader.build_report(benchmark_dir=tmp_path)
    assert {g["dataset"] for g in active["arms"].values()} == {"correlation_pair", "numerical_challenges"}
    assert "numerical_pilot" not in active["ground_truth_by_dataset"]
    # Explicitly supplying archived records still needs the opt-in.
    assert reader.build_report(records=all_records, benchmark_dir=tmp_path) == active
    historical = reader.build_report(benchmark_dir=tmp_path, include_archived=True)
    assert len(historical["arms"]) == 3  # One collection must not merge source cohorts.
    group, = [g for g in historical["arms"].values() if g["dataset"] == "numerical_pilot"]
    row = group["per_case"]["case_82"]
    assert row["truth"] == "reject" and row["outcome"] == "wrong_verdict"
    assert row["path"] == "archive/numerical_pilot/traces_glm/case_82/single_call/r1"
    assert row["usd"] == pytest.approx(0.0102)
    assert group["original_trial"] == "completion_batch"
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_archived_json_cli_does_not_overwrite_active_scoreboard(tmp_path, monkeypatch, capsys):
    reader = _reader()
    output = tmp_path / "scoreboard.json"
    output.write_bytes(b"active scoreboard sentinel\n")
    calls = []
    monkeypatch.setattr(reader, "OUT", output)
    monkeypatch.setattr(reader, "build_report", lambda **kwargs: calls.append(kwargs) or {"arms": {}})
    assert reader.main(["--include-archived", "--json"]) == 0
    assert calls == [{"include_archived": True}]
    assert json.loads(capsys.readouterr().out) == {"arms": {}}
    assert output.read_bytes() == b"active scoreboard sentinel\n"


def test_report_separates_provider_trial_dataset_and_uses_exact_model_price(tmp_path):
    reader = _reader()
    fw = "accounts/fireworks/models/glm-5p3"
    router = "z-ai/glm-5.3-flash"
    records = [
        _single(tmp_path, provider="fireworks", model=fw, trial="r1"),
        _single(tmp_path, provider="fireworks", model=fw, trial="r2", verdict="reject"),
        _single(tmp_path, provider="openrouter", model=router, trial="r1"),
        _single(tmp_path, provider="fireworks", model=fw, trial="r1", dataset="correlation_pair"),
    ]
    report = reader.build_report(records=records, benchmark_dir=tmp_path, labels={
        "benchmark_fn_fp": {"case_a": "trust"}, "correlation_pair": {"case_a": "reject"},
    })
    groups = list(report["arms"].values())
    assert len(groups) == 4
    assert sum(group["attempts"] for group in groups) == 4
    assert sorted(group["correct"] for group in groups) == [0, 0, 1, 1]
    for group in groups:
        assert group["usd"] == pytest.approx(0.00248 if group["provider"] == "fireworks" else 0.000575)
    assert report["ground_truth"] == {"case_a": "trust"}


def test_numerical_labels_are_loaded_and_grouped_separately(tmp_path):
    reader = _reader()
    (tmp_path / "case_map.json").write_text('{"cases": {}}')
    records = []
    for dataset, truth in (("correlation_pair", "trust"), ("numerical_challenges", "reject")):
        directory = tmp_path / dataset
        directory.mkdir()
        (directory / "validation_gpu.json").write_text(json.dumps({"cases": {
            "case_a": {"ground_truth": truth}}}))
        records.append(_single(tmp_path, provider="fireworks", model="accounts/fireworks/models/glm-5p3",
                               trial="r1", dataset=dataset, verdict="trust"))
    report = reader.build_report(records=records, benchmark_dir=tmp_path)
    groups = {row["dataset"]: row for row in report["arms"].values()}
    assert groups["correlation_pair"]["correct"] == 1
    assert groups["numerical_challenges"]["correct"] == 0
    assert groups["numerical_challenges"]["wrong"] == ["case_a"]
    assert report["ground_truth_by_dataset"]["numerical_challenges"] == {"case_a": "reject"}


def test_exhaustion_abstention_and_unknown_price_are_not_wrong_or_free(tmp_path):
    reader = _reader()
    records = [
        _single(tmp_path, provider="custom", model="unknown/model", trial="limited",
                verdict="reject", stop="length"),
        _single(tmp_path, provider="custom", model="unknown/model", trial="abstain",
                verdict="needs_more_evidence"),
    ]
    report = reader.build_report(records=records, benchmark_dir=tmp_path,
                                 labels={"benchmark_fn_fp": {"case_a": "trust"}})
    outcomes = {group["trial"]: group for group in report["arms"].values()}
    assert outcomes["limited"]["outcomes"]["token_limit"] == 1
    assert outcomes["abstain"]["outcomes"]["abstention"] == 1
    for group in outcomes.values():
        assert group["wrong"] == []
        assert group["usd"] is None
        assert group["unpriced"] == 1


def test_dated_price_snapshot_does_not_reprice_historical_trials(tmp_path):
    reader = _reader()
    model = "accounts/fireworks/models/glm-5p3"
    old = _single(tmp_path, provider="fireworks", model=model, trial="legacy")
    new = _single(tmp_path, provider="fireworks", model=model, trial="completion")
    new["metadata"]["pricing_snapshot"] = {
        "input_per_million": 1.4, "output_per_million": 4.4,
        "checked_at": "2026-09-30"}
    report = reader.build_report(records=[old, new], benchmark_dir=tmp_path,
                                 labels={"benchmark_fn_fp": {"case_a": "trust"}})
    costs = {group["trial"]: group["usd"] for group in report["arms"].values()}
    assert costs["legacy"] == pytest.approx(0.00248)
    assert costs["completion"] == pytest.approx(0.0102)


def test_reads_legacy_flat_usage_and_raw_provider_fallback(tmp_path):
    reader = _reader()
    profile = reader.profile_for("accounts/fireworks/models/glm-5p3")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "usage.json").write_text(json.dumps({"input_tokens": 1000, "output_tokens": 2000,
                                                   "stop_reason": "stop"}))
    (legacy / "response_text.json").write_text('{"verdict":"trust","confidence":0.9}')
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "raw_response.json").write_text(json.dumps({
        "usage": {"prompt_tokens": 1000, "completion_tokens": 2000},
        "choices": [{"finish_reason": "stop", "message": {"content":
                     '{"verdict":"trust","confidence":0.9}'}}],
    }))
    assert reader.read_single_call(legacy, profile) == reader.read_single_call(raw, profile)


def test_duplicate_attempt_identity_cannot_silently_overwrite(tmp_path):
    reader = _reader()
    record = _single(tmp_path, provider="fireworks", model="accounts/fireworks/models/glm-5p3", trial="r1")
    with pytest.raises(ValueError, match="Duplicate"):
        reader.build_report(records=[record, record], benchmark_dir=tmp_path,
                            labels={"benchmark_fn_fp": {"case_a": "trust"}})


def _raw_tool_call(run_dir, call, *, usage=None, finish=None, pending=False):
    folder = run_dir / "llm_calls" / call
    folder.mkdir(parents=True)
    (folder / "request.json").write_text('{"model":"accounts/fireworks/models/glm-5p3"}')
    (folder / "metadata.json").write_text(json.dumps({"status": "started" if pending else "completed"}))
    if not pending:
        (folder / "response.json").write_text(json.dumps({
            "usage": usage, "choices": [{"finish_reason": finish, "message": {"content": ""}}],
        }))


def test_raw_usage_includes_unparsed_calls_and_keeps_recovered_final_verdict(tmp_path):
    reader = _reader()
    profile = reader.profile_for("accounts/fireworks/models/glm-5p3")
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "history": [{"usage": {"input_tokens": 1000, "output_tokens": 2000}}],
        "verdict": {"verdict": "trust", "confidence": 0.9},
    }))
    _raw_tool_call(tmp_path, "001", usage={"prompt_tokens": 1000, "completion_tokens": 2000}, finish="length")
    _raw_tool_call(tmp_path, "002", usage={"prompt_tokens": 3000, "completion_tokens": 4000}, finish="tool_calls")
    row = reader.read_run(path, profile)
    assert row["usage_source"] == "raw_chat_responses"
    assert row["input_tokens"] == 4000
    assert row["output_tokens"] == 6000
    assert row["usd"] == pytest.approx(0.00772)
    assert row["usage_coverage"]["status"] == "complete"
    assert row["usage_coverage"]["history_calls_with_usage"] == 1
    assert row["usd_is_partial"] is False
    assert [call["finish_reasons"] for call in row["call_finish_reasons"]] == [["length"], ["tool_calls"]]
    assert reader.classify_outcome(row, "trust") == "correct"
    row["verdict"] = "needs_more_evidence"
    assert reader.classify_outcome(row, "trust") == "abstention"


def test_failed_tool_without_run_json_keeps_raw_cost_and_reports_partial_coverage(tmp_path):
    reader = _reader()
    _raw_tool_call(tmp_path, "001", usage={"prompt_tokens": 1000, "completion_tokens": 2000}, finish="length")
    _raw_tool_call(tmp_path, "002", pending=True)
    record = {"path": tmp_path, "case": "case_a", "arm": "solo", "trial": "r1",
              "dataset": "benchmark_fn_fp", "model": "accounts/fireworks/models/glm-5p3",
              "provider": "fireworks", "metadata": {}}
    report = reader.build_report(records=[record], benchmark_dir=tmp_path,
                                 labels={"benchmark_fn_fp": {"case_a": "trust"}})
    group = next(iter(report["arms"].values()))
    row = group["per_case"]["case_a"]
    assert row["outcome"] == "token_limit"
    assert row["usd"] == pytest.approx(0.00248)
    assert row["usage_coverage"]["status"] == "partial"
    assert row["usage_coverage"]["missing_response_calls"] == ["002"]
    assert row["usage_coverage"]["responses_saved"] == 1
    assert row["usage_coverage"]["history_calls_with_usage"] == 0
    assert group["partial_cost_attempts"] == 1
    assert group["wrong"] == []


def test_legacy_history_fallback_and_response_without_usage_are_explicit(tmp_path):
    reader = _reader()
    profile = reader.profile_for("accounts/fireworks/models/glm-5p3")
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"history": [{"usage": {"input_tokens": 1000, "output_tokens": 2000}}]}))
    legacy = reader.read_run(path, profile)
    assert legacy["usage_source"] == "history"
    assert legacy["usd"] == pytest.approx(0.00248)
    assert legacy["usage_coverage"]["status"] == "legacy_history"
    _raw_tool_call(tmp_path, "001", pending=True)
    pending = reader.read_run(path, profile)
    assert pending["usage_source"] == "history"
    assert pending["usd"] == pytest.approx(0.00248)
    assert pending["usd_is_partial"] is True
    _raw_tool_call(tmp_path, "002", usage=None, finish="stop")
    missing = reader.read_run(path, profile)
    assert missing["usage_source"] == "raw_chat_responses"
    assert missing["usd"] is None
    assert missing["usage_coverage"]["missing_usage_calls"] == ["002"]
    assert missing["usd_is_partial"] is True


def test_single_failure_retains_raw_cost_or_marks_unknown(tmp_path):
    reader = _reader()
    profile = reader.profile_for("accounts/fireworks/models/glm-5p3")
    pending_dir = tmp_path / "pending"
    _raw_tool_call(pending_dir, "001", pending=True)
    pending = reader.read_single_call(pending_dir, profile)
    assert pending["usd"] is None
    assert pending["usd_is_partial"] is True
    received_dir = tmp_path / "received"
    _raw_tool_call(received_dir, "001", usage={"prompt_tokens": 1000, "completion_tokens": 2000}, finish="length")
    received = reader.read_single_call(received_dir, profile)
    assert received["usd"] == pytest.approx(0.00248)
    assert received["input_tokens"] == 1000
    assert received["stop_reason"] == "length"
    assert reader.classify_outcome(received, "trust") == "token_limit"
