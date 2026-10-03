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
from types import ModuleType

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    class Image:
        def __init__(self):
            self.packages = []
            self.mounts = []

        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def pip_install(self, *args, **kwargs):
            self.packages.extend(args)
            return self

        def add_local_dir(self, *args, **kwargs):
            self.mounts.append(args)
            return self

    class App:
        def __init__(self, *args):
            pass

        def function(self, **kwargs):
            def decorate(function):
                function._modal_options = kwargs
                return function
            return decorate

        def local_entrypoint(self):
            return lambda function: function

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(
        App=App, Image=Image,
        Secret=SimpleNamespace(from_dotenv=lambda *args: SimpleNamespace(kind="secret"),
                               from_dict=lambda *args: SimpleNamespace(kind="secret")),
        is_local=lambda: True))
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
    monkeypatch.setattr(module, "_REMOTE_CASE_ROOT", str(tmp_path / "remote_cases"))
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


def test_remote_import_never_reads_host_sources_or_dotenv(runner, monkeypatch):
    modal_stub = sys.modules["modal"]
    monkeypatch.setattr(modal_stub, "is_local", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("remote import accessed host-only mounts or dotenv")

    monkeypatch.setattr(modal_stub.Image, "add_local_dir", forbidden)
    monkeypatch.setattr(modal_stub.Secret, "from_dotenv", forbidden)
    source = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/eval_scripts/run_agentic_modal.py"
    remote = {"__name__": "offline_remote_agentic", "__file__": "/root/run_agentic_modal.py"}
    exec(compile(source.read_text(), remote["__file__"], "exec"), remote)
    assert remote["REPO_ROOT"] == Path("/root")
    assert remote["CASES_DIR"] == Path("/root/cases")
    assert remote["image"].mounts == []
    local = runner[0]
    for name in ("run_one", "preflight_remote"):
        assert len(remote[name]._modal_options["secrets"]) == len(getattr(local, name)._modal_options["secrets"]) == 1
        assert remote[name]._modal_options["secrets"][0].kind == "secret"
    assert remote["run_one"]._modal_options["gpu"] == "T4"


def test_preflight_only_never_reserves_trials_or_calls_models(runner, monkeypatch):
    module, _, benchmark = runner
    calls = []
    monkeypatch.setattr(module, "preflight_remote", SimpleNamespace(remote=lambda *args: calls.append(args) or {"model_calls": 0}))
    monkeypatch.setattr(module, "run_one", SimpleNamespace(remote=lambda *args: pytest.fail("model trial submitted")))
    module.main(cases="case_a", provider="fireworks", preflight_only=True)
    assert calls[0][0] == ["case_a"]
    assert set(calls[0][1]["case_a"]) >= {"kernel_sha256", "problem_sha256"}
    assert set(calls[0][2]) == {"case_a"}
    assert set(calls[0][2]["case_a"]) == {"kernel.py", "problem.txt", "meta.json"}
    assert not (benchmark / "traces_glm").exists()


def test_worker_image_does_not_mount_the_public_corpus(runner):
    module, _, _ = runner
    assert [remote for _, remote in module.image.mounts] == ["/root/verifier"]
    for worker in (module.run_one, module.preflight_remote):
        assert worker._modal_options["single_use_containers"] is True


def test_only_allowlisted_public_files_are_sent_per_job(runner, monkeypatch):
    module, _, benchmark = runner
    case = module.CASES_DIR / "case_a"
    (case / "LICENSE").write_text("Public attribution")
    (case / "truth.json").write_text('{"ground_truth":"reject"}')
    (case / "test.py").write_text("expected_truth = 'reject'")
    (case / "private_data").mkdir()
    (case / "private_data/answer.json").write_text('{"ground_truth":"reject"}')
    calls = []
    monkeypatch.setattr(module, "run_one", SimpleNamespace(
        remote=lambda *job: calls.append(job) or make_result(*job[:2])))
    module.main(arm="both", cases="case_a,case_b", provider="fireworks", trial="public_bundle")
    for job in calls:
        files = job[-1]
        assert set(files) == ({"kernel.py", "problem.txt", "meta.json", "LICENSE"}
                              if job[0] == "case_a" else {"kernel.py", "problem.txt", "meta.json"})
        assert all(isinstance(data, bytes) for data in files.values())
        meta = read(benchmark / "traces_glm" / job[0] / job[1] / "public_bundle/trace_meta.json")
        assert meta["case_visibility"] == "current_case_only"
        assert meta["public_input_files"] == sorted(files)


def test_materialization_removes_previous_case_and_previous_arm_traces(runner):
    import hashlib
    module, _, _ = runner
    first = module._public_files_for_case(module.CASES_DIR / "case_a")
    second = module._public_files_for_case(module.CASES_DIR / "case_b")
    module._materialize_public_case("case_a", first)
    old_trace = Path(module._REMOTE_TRACE_ROOT) / "r1/case_a/debate/verdict.json"
    old_trace.parent.mkdir(parents=True)
    old_trace.write_text('{"verdict":"reject"}')
    old_source = Path(module._REMOTE_CASE_ROOT) / "case_a/kernel.py"
    (old_source.parent / "private_answer.json").write_text('{"truth":"reject"}')
    hashes = {f"{kind}_sha256": hashlib.sha256(second[name]).hexdigest()
              for kind, name in (("kernel", "kernel.py"), ("problem", "problem.txt"))}
    report = module._materialize_public_case("case_b", second, hashes)
    assert report["visible_cases"] == ["case_b"]
    assert report["previous_trace_root_removed"] is True
    assert not old_source.exists() and not old_trace.exists()
    assert sorted(p.name for p in Path(module._REMOTE_CASE_ROOT).iterdir()) == ["case_b"]
    for name, content in second.items():
        assert (Path(module._REMOTE_CASE_ROOT) / "case_b" / name).read_bytes() == content
        assert report["public_files_sha256"][name] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("bad", ["extra_file", "nonneutral_meta", "path_escape", "hash_mismatch", "missing_payload"])
def test_invalid_public_payload_fails_before_worker_execution(runner, bad):
    module, _, _ = runner
    files = module._public_files_for_case(module.CASES_DIR / "case_a")
    name, hashes = "case_a", None
    if bad == "extra_file":
        files["private_truth.json"] = b'{"ground_truth":"reject"}'
    elif bad == "nonneutral_meta":
        files["meta.json"] = b'{"passed":true}'
    elif bad == "path_escape":
        name = "../case_b"
    elif bad == "hash_mismatch":
        hashes = {"kernel_sha256": "0" * 64, "problem_sha256": "0" * 64}
    else:
        files = None
    with pytest.raises(ValueError):
        module._materialize_public_case(name, files, hashes)
    assert not Path(module._REMOTE_CASE_ROOT).exists()


@pytest.mark.parametrize("status", ["under_test", "unverified", "unknown"])
@pytest.mark.parametrize("family", [None, "sequence_audit", "joint_audit", "mutation_audit", "box_relu_audit"])
def test_legacy_neutral_metadata_preserves_exact_bytes(runner, status, family):
    module, _, _ = runner
    case = module.CASES_DIR / "case_a"
    meta = {"name": "case_a", "status": status, "passed": None}
    if family is not None:
        meta["family"] = family
    original = (json.dumps(meta, indent=4) + "\n\n").encode()
    (case / "meta.json").write_bytes(original)
    files = module._public_files_for_case(case)
    module._materialize_public_case("case_a", files)
    assert files["meta.json"] == original
    assert (Path(module._REMOTE_CASE_ROOT) / "case_a/meta.json").read_bytes() == original
    assert (case / "meta.json").read_bytes() == original


@pytest.mark.parametrize("meta", [
    {"name": "case_a", "status": "under_test", "passed": True},
    {"name": "case_a", "status": "under_test", "passed": False},
    {"name": "case_a", "status": "under_test", "passed": 0},
    {"name": "case_a", "status": "completed", "passed": None},
    {"name": "case_b", "status": "unverified", "passed": None},
    {"name": "case_a", "status": "unverified"},
    {"name": "case_a", "status": "unverified", "passed": None, "ground_truth": "reject"},
    {"name": "case_a", "status": "unverified", "passed": None, "notes": "Anything else"},
    {"name": "case_a", "status": "unverified", "passed": None, "family": "reject_cases"},
    {"name": "case_a", "status": "unverified", "passed": None,
     "family": "sequence_audit", "truth": "trust"},
    [], "trust", None,
])
def test_nonneutral_metadata_fails_before_trial_reservation_or_api(runner, monkeypatch, meta):
    module, _, benchmark = runner
    (module.CASES_DIR / "case_a/meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(module, "run_one", SimpleNamespace(
        remote=lambda *args: pytest.fail("nonneutral input reached model worker")))
    monkeypatch.setattr(module, "preflight_remote", SimpleNamespace(
        remote=lambda *args: pytest.fail("nonneutral input reached GPU worker")))
    with pytest.raises(ValueError, match="strict neutral legacy schema"):
        module.main(arm="solo", cases="case_a", provider="fireworks", trial="invalid_meta")
    assert not (benchmark / "traces_glm").exists()


def test_all_active_repository_public_bundles_collect_without_changing_metadata(runner):
    from case_registry import is_active_case

    module, _, _ = runner
    benchmark = Path(__file__).resolve().parents[1] / "benchmark_fn_fp"
    details = read(benchmark / "case_map.json")["case_details"]
    active = {name: row for name, row in details.items() if is_active_case(row)}
    # Include the original empty schema, original explicit unknown schema and
    # all four historical family values that the isolation change must retain.
    assert {"case_01", "case_64", "case_66", "case_70", "case_72", "case_115"} <= set(active)
    for name, row in active.items():
        directory = benchmark / row["public_dir"]
        original = (directory / "meta.json").read_bytes()
        files = module._public_files_for_case(directory)
        assert files["meta.json"] == original, name
        assert (directory / "meta.json").read_bytes() == original, name


def test_duplicate_metadata_fields_cannot_hide_a_label_in_preserved_bytes(runner, monkeypatch):
    module, _, benchmark = runner
    (module.CASES_DIR / "case_a/meta.json").write_bytes(
        b'{"name":"case_a","status":"under_test","passed":true,"passed":null}')
    monkeypatch.setattr(module, "run_one", SimpleNamespace(
        remote=lambda *args: pytest.fail("duplicate metadata reached model worker")))
    with pytest.raises(ValueError, match="duplicate fields"):
        module.main(arm="solo", cases="case_a", provider="fireworks", trial="invalid_meta")
    assert not (benchmark / "traces_glm").exists()


def test_actual_modal_sdk_local_remote_dependency_order_matches(monkeypatch):
    sdk = pytest.importorskip("modal")
    source = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/eval_scripts/run_agentic_modal.py"
    local = ModuleType("offline_sdk_agentic_local")
    local.__file__ = str(source)
    remote = ModuleType("offline_sdk_agentic_remote")
    remote.__file__ = "/root/run_agentic_modal.py"
    for module in (local, remote):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    with monkeypatch.context() as context:
        context.setattr(sdk, "is_local", lambda: True)
        exec(compile(source.read_text(), local.__file__, "exec"), local.__dict__)
    with monkeypatch.context() as context:
        context.setattr(sdk, "is_local", lambda: False)
        context.setattr(sdk.Secret, "from_dotenv", lambda *a, **kw: pytest.fail("remote read dotenv"))
        exec(compile(source.read_text(), remote.__file__, "exec"), remote.__dict__)
    for name in ("run_one", "preflight_remote"):
        local_deps = getattr(local, name).deps(only_explicit_mounts=True)
        remote_deps = getattr(remote, name).deps(only_explicit_mounts=True)
        assert [type(dep).__name__.lstrip("_") for dep in local_deps] == ["Secret", "Image"]
        assert [type(dep).__name__.lstrip("_") for dep in remote_deps] == ["Secret", "Image"]
        # Never resolve objects or send any request to Modal from this test.
        assert not any(dep.is_hydrated for dep in [*local_deps, *remote_deps])


@pytest.mark.parametrize("dataset,expected", [
    ("benchmark_fn_fp", ["case_a"]),
    ("correlation_pair", ["case_b"]),
    ("numerical_challenges", ["case_c"]),
    ("single_call_vs_tools_challenges", ["case_b", "case_c"]),
])
def test_all_keeps_historical_sources_when_running_combined_collection(runner, monkeypatch, dataset, expected):
    import hashlib
    module, _, benchmark = runner
    origins = {"case_a": "benchmark_fn_fp", "case_b": "correlation_pair", "case_c": "numerical_challenges"}
    (benchmark / "case_map.json").write_text(json.dumps({"case_details": {
        name: {"dataset": origin} for name, origin in origins.items()}}))
    for case, suffix in (("case_b", "private_data/correlation_pair"), ("case_c", "private_data")):
        public = benchmark / "triton_eval_cases" / case
        private = benchmark / "single_call_vs_tools_challenges" / suffix
        private.mkdir(parents=True, exist_ok=True)
        (private / "validation_gpu.json").write_text(json.dumps({"cases": {case: {
            "ground_truth": "trust", **{
                f"{kind}_sha256": hashlib.sha256((public / filename).read_bytes()).hexdigest()
                for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt"))}}}}))
    calls = []
    monkeypatch.setattr(module, "run_one", SimpleNamespace(
        remote=lambda *job: calls.append(job) or make_result(*job[:2])))
    module.main(arm="solo", all=True, dataset=dataset, provider="fireworks", trial="shared_directory")
    assert sorted(job[0] for job in calls) == expected
    for job in calls:
        assert job[7] == origins[job[0]]
        assert read(trace_dir(benchmark, job[0], "shared_directory") / "trace_meta.json")["dataset"] == origins[job[0]]
    assert all("correlation_pair/eval_cases" not in local and remote != "/root/correlation_cases"
               for local, remote in module.image.mounts)


@pytest.mark.parametrize("dataset", ["correlation_pair", "numerical_challenges", "evidence_challenges", "real_kernel_challenges"])
def test_frozen_remote_worker_uses_only_current_public_case(runner, monkeypatch, dataset):
    import os
    from verifier import agentic_run
    module, _, _ = runner
    monkeypatch.setattr(os, "chdir", lambda path: None)
    for key in ("AGENTIC_MODEL", "AGENTIC_PROVIDER", "AGENTIC_PROBE_SANDBOX",
                "AGENTIC_LLM_TIMEOUT_SECONDS", "AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET",
                "AGENTIC_OPENAI_REASONING_EFFORT", "AGENTIC_LLM_TRACE_DIR", "AGENTIC_LLM_TRACE_PROGRESS"):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    def fake_main(argv):
        assert argv[argv.index("--dataset-dir") + 1] == module._REMOTE_CASE_ROOT
        assert [p.name for p in Path(module._REMOTE_CASE_ROOT).iterdir()] == ["case_b"]
        dest = Path(argv[argv.index("--run-dir") + 1])
        assert dest.parts[-3:] == (dataset, "case_b", "solo")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "verdict.json").write_text('{"verdict":"trust"}')
        return 0

    monkeypatch.setattr(agentic_run, "main", fake_main)
    result = module.run_one("case_b", "solo", 10, "accounts/fireworks/models/glm-5p3",
                            32768, "fireworks", 1800, dataset=dataset, trial="shared",
                            public_files=module._public_files_for_case(module.CASES_DIR / "case_b"))
    assert result["ok"] is True
    with tarfile.open(fileobj=io.BytesIO(result["tar"])) as archive:
        runtime = json.load(archive.extractfile("./runtime.json"))
        visibility = json.load(archive.extractfile("./artifact_visibility.json"))
    assert runtime["python"]
    assert len(runtime["verifier_sha256"]) == 64
    assert visibility["visible_cases"] == ["case_b"]
    assert visibility["previous_trace_root_removed"] is True


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
        dest.mkdir(parents=True, exist_ok=True)
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
                            32768, "fireworks", 1800, trial="remote_test",
                            public_files=module._public_files_for_case(module.CASES_DIR / "case_a"))
    assert result["ok"] is False
    assert result["error"]
    assert "offline-collection-secret" not in result["error"]
    with tarfile.open(fileobj=io.BytesIO(result["tar"]), mode="r:gz") as archive:
        assert archive.extractfile("./llm_calls/call_1/response.json").read() == b'{"usage":{"completion_tokens":10}}'
        assert archive.extractfile("./verdict.json") is not None


def test_numerical_batch_checks_all_frozen_cases_before_submitting(runner, monkeypatch):
    import hashlib
    module, _, benchmark = runner
    root = benchmark / "single_call_vs_tools_challenges"
    root.mkdir()
    validated = {}
    for name in ("case_a", "case_b", "case_c"):
        validated[name] = {"ground_truth": "trust", **{
            f"{kind}_sha256": hashlib.sha256((benchmark / "triton_eval_cases" / name / filename).read_bytes()).hexdigest()
            for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt"))}}
    (root / "validation_gpu.json").write_text(json.dumps({"cases": validated}))
    edited = benchmark / "triton_eval_cases/case_b/problem.txt"
    original = edited.read_text()
    edited.write_text("Changed contract")
    calls = []

    def remote(*job):
        calls.append(job)
        return make_result(*job[:2])

    monkeypatch.setattr(module, "run_one", SimpleNamespace(remote=remote))
    with pytest.raises(ValueError, match="Frozen case changed"):
        module.main(arm="both", cases="case_a,case_b", provider="fireworks", trial="challenge",
                    dataset="single_call_vs_tools_challenges")
    assert calls == []
    assert not (benchmark / "traces_glm").exists()
    edited.write_text(original)
    module.main(arm="both", cases="case_a,case_b", provider="fireworks", trial="challenge",
                dataset="single_call_vs_tools_challenges")
    assert len(calls) == 4
    for job in calls:
        assert job[7] == "numerical_challenges"
        assert job[9] == {key: value for key, value in validated[job[0]].items() if key.endswith("sha256")}
        meta = read(benchmark / "traces_glm" / job[0] / job[1] / "challenge/trace_meta.json")
        assert meta["dataset"] == "numerical_challenges"


def test_archived_pilot_is_not_mounted_or_submitted(runner, monkeypatch):
    module, _, benchmark = runner
    assert all("pilot" not in local and remote != "/root/pilot_cases" for local, remote in module.image.mounts)
    assert not hasattr(module, "pilot_image")
    worker = module.run_one
    with pytest.raises(ValueError, match="archived"):
        worker("case_82", "solo", 10, "model", 32768, "fireworks", 1800,
               dataset="numerical_pilot")
    calls = []
    monkeypatch.setattr(module, "run_one", SimpleNamespace(remote=lambda *job: calls.append(job)))
    with pytest.raises(ValueError, match="archived"):
        module.main(arm="both", all=True, dataset="numerical_pilot", provider="fireworks",
                    trial="pilot", max_tokens=32768, total_output_tokens=32768)
    assert not calls
    assert not (benchmark / "traces_glm").exists()


def test_only_missing_reuses_any_valid_judgment_across_trials_in_target_tree(runner, monkeypatch):
    module, storage, benchmark = runner

    def previous(case, arm, verdict, *, tree="traces_glm", status="completed", trial="old",
                 error="", model="z-ai/glm-5.3-flash"):
        storage.reserve_trace(case, arm, traces_dir=tree, trial=trial,
                              metadata={"status": status, "model": model, "provider": "openrouter"})
        storage.write_trace(case, arm, traces_dir=tree, trial=trial,
                            files={"run.json": json.dumps({"verdict": {"verdict": verdict}}),
                                   "runner_error.txt": error})

    # Correctness and provider are irrelevant to coverage; abstention is a valid answer.
    previous("case_a", "solo", "reject", status="historical")
    previous("case_b", "solo", "needs_more_evidence")
    # Failed finalization and another model tree must not hide a missing GLM judgment.
    previous("case_c", "solo", "trust", status="error")
    previous("case_c", "solo", "trust", tree="traces_opus5", model="claude-opus-5")
    previous("case_a", "debate", "trust", error="transport failure after verdict")
    calls = []
    monkeypatch.setattr(module, "run_one", SimpleNamespace(
        remote=lambda *job: calls.append(job[:2]) or make_result(*job[:2])))
    module.main(arm="both", all=True, provider="fireworks", trial="fill", only_missing=True)
    assert sorted(calls) == [("case_a", "debate"), ("case_b", "debate"),
                             ("case_c", "debate"), ("case_c", "solo")]
    assert not (benchmark / "traces_glm/case_a/solo/fill").exists()
    assert not (benchmark / "traces_glm/case_b/solo/fill").exists()


@pytest.mark.parametrize("existing_batches", [[], ["original_batch", "original_batch"], ["batch_a", "batch_b"]])
def test_numeric_skip_existing_preserves_one_batch_or_refuses_ambiguous_resume(runner, monkeypatch, existing_batches):
    module, storage, benchmark = runner
    for (case, arm), batch in zip((("case_a", "solo"), ("case_b", "debate")), existing_batches):
        dest = storage.reserve_trace(case, arm, traces_dir="traces_glm", trial="r9",
            metadata={"original_trial": batch, "status": "completed"})
        (dest / "verdict.json").write_text('{"verdict":"trust"}')
    before = {p.relative_to(benchmark): p.read_bytes() for p in benchmark.rglob("*") if p.is_file()}
    calls = []
    monkeypatch.setattr(module, "run_one", SimpleNamespace(
        remote=lambda *job: calls.append(job[:2]) or make_result(*job[:2])))
    if len(set(existing_batches)) > 1:
        with pytest.raises(ValueError, match="multiple experiment batches"):
            module.main(arm="both", all=True, provider="fireworks", trial="r9", skip_existing=True)
        assert calls == []
        assert {p.relative_to(benchmark): p.read_bytes() for p in benchmark.rglob("*") if p.is_file()} == before
        return
    module.main(arm="both", all=True, provider="fireworks", trial="r9", skip_existing=True)
    assert len(calls) == 6 - len(existing_batches)
    saved = [read(p) for p in (benchmark / "traces_glm").glob("case_*/*/r9/trace_meta.json")]
    batches = {meta["original_trial"] for meta in saved}
    assert len(saved) == 6 and len(batches) == 1
    if existing_batches:
        assert batches == {"original_batch"}
    else:
        assert next(iter(batches)).startswith("run_")
