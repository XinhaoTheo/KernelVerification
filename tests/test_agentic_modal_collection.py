"""Offline collection tests: every submitted job retains its identity and trace."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def pip_install(self, *args, **kwargs):
            return self

        def add_local_dir(self, *args, **kwargs):
            return self

    class App:
        def __init__(self, *args):
            pass

        def function(self, **kwargs):
            return lambda function: function

        def local_entrypoint(self):
            return lambda function: function

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(
        App=App, Image=Image, Secret=SimpleNamespace(from_dotenv=lambda *args: None)))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    source = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/eval_scripts/run_agentic_modal.py"
    spec = importlib.util.spec_from_file_location("offline_agentic_modal_runner", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(source.parent))
    storage = importlib.import_module("traces")
    benchmark = tmp_path / "benchmark_fn_fp"
    cases = benchmark / "triton_eval_cases"
    for name in ("case_a", "case_b", "case_c"):
        case = cases / name
        case.mkdir(parents=True)
        (case / "kernel.py").write_text("def kernel(x): return x + 1\n")
        (case / "problem.txt").write_text("Add one.\n")
        (case / "meta.json").write_text("{}")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "CASES_DIR", cases)
    monkeypatch.setattr(module, "_REMOTE_TRACE_ROOT", str(tmp_path / "remote_traces"))
    monkeypatch.setattr(storage, "BENCHMARK_DIR", benchmark)
    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-collection-secret")
    return module, storage, benchmark


def make_result(entry, arm):
    buffer = io.BytesIO()
    payloads = {
        "verdict.json": b'{"verdict":"trust","confidence":0.9}',
        "run.json": b'{"history":[]}',
        "llm_calls/call_1/response.json": b'{"choices":[{"finish_reason":"stop"}]}',
    }
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return {"entry": entry, "arm": arm, "ok": True, "error": None,
            "verdict": {"verdict": "trust", "confidence": 0.9},
            "stdout": "completed\n", "tar": buffer.getvalue()}


def trace_dir(benchmark, case, trial="collection"):
    return benchmark / "traces_glm" / case / "solo" / trial


def read(path):
    return json.loads(path.read_text())


def test_remote_failure_does_not_stop_other_submitted_jobs(runner, monkeypatch):
    module, _, benchmark = runner
    barrier = threading.Barrier(3)
    calls = []

    def remote(*job):
        calls.append(job[:2])
        barrier.wait(timeout=5)
        if job[0] == "case_b":
            raise RuntimeError("Remote failed with offline-collection-secret")
        return make_result(*job[:2])

    monkeypatch.setattr(module, "run_one", SimpleNamespace(remote=remote))
    with pytest.raises(SystemExit) as error:
        module.main(arm="solo", cases="case_a,case_b,case_c", provider="fireworks", trial="collection")
    assert error.value.code == 1
    assert sorted(calls) == [("case_a", "solo"), ("case_b", "solo"), ("case_c", "solo")]
    for case in ("case_a", "case_c"):
        dest = trace_dir(benchmark, case)
        assert read(dest / "trace_meta.json")["status"] == "completed"
        assert (dest / "llm_calls/call_1/response.json").exists()
    failed = trace_dir(benchmark, "case_b")
    metadata = read(failed / "trace_meta.json")
    assert metadata["status"] == "error"
    assert metadata["failure_kind"] == "remote_call"
    assert metadata["case"] == "case_b" and metadata["arm"] == "solo"
    detail = read(failed / metadata["failure_file"])
    assert "[REDACTED]" in detail["message"]
    assert "offline-collection-secret" not in detail["message"]


def test_write_failure_keeps_archive_and_continues_collecting(runner, monkeypatch):
    module, storage, benchmark = runner
    original_write = storage.write_trace
    results = {}
    calls = []

    def remote(*job):
        calls.append(job[:2])
        result = make_result(*job[:2])
        results[job[0]] = result
        return result

    def write_trace(case, *args, **kwargs):
        if case == "case_b":
            raise ValueError("Synthetic archive extraction failure")
        return original_write(case, *args, **kwargs)

    monkeypatch.setattr(module, "run_one", SimpleNamespace(remote=remote))
    monkeypatch.setattr(storage, "write_trace", write_trace)
    with pytest.raises(SystemExit) as error:
        module.main(arm="solo", cases="case_a,case_b,case_c", provider="fireworks", trial="collection")
    assert error.value.code == 1
    assert sorted(calls) == [("case_a", "solo"), ("case_b", "solo"), ("case_c", "solo")]
    for case in ("case_a", "case_c"):
        assert read(trace_dir(benchmark, case) / "verdict.json")["verdict"] == "trust"
    dest = trace_dir(benchmark, "case_b")
    metadata = read(dest / "trace_meta.json")
    assert metadata["status"] == "error"
    assert metadata["failure_kind"] == "trace_write"
    assert (dest / metadata["recovery_archive"]).read_bytes() == results["case_b"]["tar"]


def test_completed_result_is_saved_while_another_job_is_running(runner, monkeypatch):
    module, storage, benchmark = runner
    saved = threading.Event()
    original_write = storage.write_trace

    def remote(*job):
        if job[0] == "case_b":
            assert saved.wait(timeout=5), "First result was buffered until all jobs completed"
        return make_result(*job[:2])

    def write_trace(case, *args, **kwargs):
        dest = original_write(case, *args, **kwargs)
        if case == "case_a":
            saved.set()
        return dest

    monkeypatch.setattr(module, "run_one", SimpleNamespace(remote=remote))
    monkeypatch.setattr(storage, "write_trace", write_trace)
    module.main(arm="solo", cases="case_a,case_b", provider="fireworks", trial="collection")
    assert saved.is_set()
    assert read(trace_dir(benchmark, "case_b") / "trace_meta.json")["status"] == "completed"


@pytest.mark.parametrize("failure", ["malformed_verdict", "error_with_verdict"])
def test_remote_error_preserves_existing_archive(runner, monkeypatch, failure):
    module, _, _ = runner
    import os
    from verifier import agentic_run

    monkeypatch.setattr(os, "chdir", lambda path: None)
    for key in ("AGENTIC_MODEL", "AGENTIC_PROVIDER", "AGENTIC_PROBE_SANDBOX",
                "AGENTIC_LLM_TIMEOUT_SECONDS", "AGENTIC_LLM_TRACE_DIR", "AGENTIC_LLM_TRACE_PROGRESS"):
        # The remote worker sets these for its process; restore them after the
        # offline simulation so later SDK tests cannot inherit a stale path.
        monkeypatch.setenv(key, os.environ.get(key, ""))

    def fake_main(argv):
        dest = Path(argv[argv.index("--run-dir") + 1])
        dest.mkdir(parents=True)
        (dest / "run.json").write_text('{"history":[]}')
        calls = dest / "llm_calls/call_1"
        calls.mkdir(parents=True)
        (calls / "response.json").write_text('{"usage":{"completion_tokens":10}}')
        (dest / "verdict.json").write_text("not-json" if failure == "malformed_verdict"
                                          else '{"verdict":"trust"}')
        if failure == "error_with_verdict":
            raise RuntimeError("Failure after verdict; offline-collection-secret")
        return 0

    monkeypatch.setattr(agentic_run, "main", fake_main)
    result = module.run_one("case_a", "solo", 10, "accounts/fireworks/models/glm-5p3",
                            32768, "fireworks", 1800, trial="remote_test")
    assert result["ok"] is False
    assert result["error"]
    assert "offline-collection-secret" not in result["error"]
    with tarfile.open(fileobj=io.BytesIO(result["tar"]), mode="r:gz") as archive:
        assert archive.extractfile("./llm_calls/call_1/response.json").read() == b'{"usage":{"completion_tokens":10}}'
        assert archive.extractfile("./verdict.json") is not None


def test_numerical_batch_checks_all_frozen_cases_before_submitting(runner, monkeypatch):
    import hashlib
    import shutil
    module, _, benchmark = runner
    root = benchmark / "numerical_challenges"
    shutil.copytree(benchmark / "triton_eval_cases", root / "eval_cases")
    validated = {}
    for name in ("case_a", "case_b", "case_c"):
        validated[name] = {"ground_truth": "trust", **{
            f"{kind}_sha256": hashlib.sha256((root / "eval_cases" / name / filename).read_bytes()).hexdigest()
            for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt"))}}
    (root / "validation_gpu.json").write_text(json.dumps({"cases": validated}))
    edited = root / "eval_cases/case_b/problem.txt"
    original = edited.read_text()
    edited.write_text("Changed contract")
    calls = []

    def remote(*job):
        calls.append(job)
        return make_result(*job[:2])

    monkeypatch.setattr(module, "run_one", SimpleNamespace(remote=remote))
    with pytest.raises(ValueError, match="Frozen case changed"):
        module.main(arm="both", cases="case_a,case_b", provider="fireworks", trial="challenge",
                    dataset="numerical_challenges")
    assert calls == []
    assert not (benchmark / "traces_glm").exists()
    edited.write_text(original)
    module.main(arm="both", cases="case_a,case_b", provider="fireworks", trial="challenge",
                dataset="numerical_challenges")
    assert len(calls) == 4
    for job in calls:
        assert job[7] == "numerical_challenges"
        assert job[9] == {key: value for key, value in validated[job[0]].items() if key.endswith("sha256")}
        meta = read(benchmark / "traces_glm" / job[0] / job[1] / "challenge/trace_meta.json")
        assert meta["dataset"] == "numerical_challenges"
