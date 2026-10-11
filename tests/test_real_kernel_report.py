"""Raw response capture survives downstream parsing failures without scoring success."""
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/eval_scripts/real_kernel_challenges/report.py"
SPEC = importlib.util.spec_from_file_location("real_kernel_report_test", SOURCE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _failed_attempt(tmp_path, *, finish="length", stale_verdict=None):
    case = "case_106"
    path = tmp_path / "traces_glm" / case / "debate" / "r1"
    call = path / "llm_calls" / "001"
    public = {case: {"kernel": "def run(): pass\n", "problem": "Verify run."}}
    labels = {case: {"ground_truth": "trust", "kernel_sha256": "kernel-hash",
                     "problem_sha256": "problem-hash"}}
    metadata = {"status": "error", "kernel_sha256": "kernel-hash",
                "problem_sha256": "problem-hash", "reasoning_effort": "low",
                "total_output_token_budget": 32768, "raw_api_capture": True,
                "verifier_sha256": "verifier-hash"}
    record = {"path": str(path), "metadata": metadata, "case": case,
              "arm": "debate", "trial": "r1", "model": report.MODEL,
              "provider": "fireworks", "dataset": "real_kernel_challenges"}
    artifact = {"kernel_code": "1: def run(): pass", "problem_text": "Verify run."}
    _write(call / "request.json", {
        "model": report.MODEL, "reasoning_effort": "low", "max_tokens": 8192,
        "messages": [{"role": "user", "content": "=== Current Run State ===\n"
                      + json.dumps({"artifact": artifact})}],
    })
    _write(call / "response.json", {
        "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        "choices": [{"finish_reason": finish, "message": {
            "tool_calls": [{"function": {"name": "record_verdict", "arguments": "{"}}],
        }}],
    })
    _write(call / "metadata.json", {"status": "error", "response_saved": True})
    _write(path / "runtime.json", {"verifier_sha256": "verifier-hash"})
    _write(path / "run.json", {"verdict": {"verdict": stale_verdict}})
    (path / "runner_error.txt").write_text("JSONDecodeError: invalid tool arguments")
    return record, labels, public, call


@pytest.mark.parametrize("finish,stale_verdict,outcome", [
    ("length", None, "token_limit"),
    ("stop", None, "error"),
    ("tool_calls", "trust", "error"),
])
def test_saved_response_remains_complete_after_local_parse_error(
        tmp_path, finish, stale_verdict, outcome):
    record, labels, public, call = _failed_attempt(
        tmp_path, finish=finish, stale_verdict=stale_verdict)
    row = report._attempt(record, labels, public, tmp_path)

    assert row["captures"] == [{"call": "001", "status": "error", "complete": True}]
    assert "raw API capture incomplete" not in row["audit_issues"]
    assert row["status"] == "error"
    assert row["outcome"] == outcome
    assert row["input_tokens"] == 100 and row["output_tokens"] == 40
    assert row["usd"] is not None and row["usd"] > 0
    assert report._counts([row])["audited_correct"] == 0
    assert report.read(call / "metadata.json")["status"] == "error"


@pytest.mark.parametrize("missing", ["request.json", "response.json", "completion_tokens"])
def test_parse_error_does_not_hide_missing_capture_or_usage(tmp_path, missing):
    record, labels, public, call = _failed_attempt(tmp_path)
    if missing.endswith(".json"):
        (call / missing).unlink()
    else:
        response = report.read(call / "response.json")
        del response["usage"][missing]
        _write(call / "response.json", response)
    row = report._attempt(record, labels, public, tmp_path)

    assert row["captures"][0]["complete"] is False
    assert "raw API capture incomplete" in row["audit_issues"]
    assert row["outcome"] in {"token_limit", "error"}
    assert report._counts([row])["audited_correct"] == 0
