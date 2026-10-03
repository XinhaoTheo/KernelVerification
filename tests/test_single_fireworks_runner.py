"""Source-only Fireworks runner trace completeness without network calls."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from types import SimpleNamespace

import openai
import pytest

from benchmark_fn_fp.eval_scripts import run_single_fireworks as runner


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmark_fn_fp"
    case = benchmark / "triton_eval_cases" / "case_tiny"
    case.mkdir(parents=True)
    code = "def kernel(x):\n    return x + 1\n"
    problem = "For the supplied input, add one.\n"
    (case / "kernel.py").write_text(code)
    (case / "problem.txt").write_text(problem)
    monkeypatch.setattr(runner, "REPO", tmp_path)
    # The CLI also supports direct execution and imports its shared storage
    # module by its flat name. Patch the instance its imported functions use.
    trace_module = importlib.import_module(runner.reserve_trace.__module__)
    monkeypatch.setattr(trace_module, "BENCHMARK_DIR", benchmark)
    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-fireworks-test-key")
    return benchmark, code, problem


def read(path):
    return json.loads(path.read_text())


def fake_openai(monkeypatch, raw, *, before_response=None):
    requests = []
    constructors = []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        if before_response:
            before_response(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=raw["usage"]["prompt_tokens"],
                                  completion_tokens=raw["usage"]["completion_tokens"]),
            model_dump=lambda **_: deepcopy(raw),
        )

    def client(**kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(openai, "OpenAI", client)
    return requests, constructors


def response(content, finish_reason):
    return {
        "id": "offline-single-call", "model": runner.MODEL, "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
            "role": "assistant", "content": content,
            "reasoning_content": "Raw provider reasoning, preserved verbatim.\n第二行。",
        }}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 321, "total_tokens": 441,
                  "completion_tokens_details": {"reasoning_tokens": 290}},
    }


@pytest.mark.parametrize("dataset,expected", [
    ("benchmark_fn_fp", ["case_tiny"]), ("correlation_pair", ["case_pair"]),
    ("numerical_challenges", ["case_numeric"]),
    ("single_call_vs_tools_challenges", ["case_numeric", "case_pair"]),
])
def test_all_runs_combined_collection_with_each_cases_original_identity(isolated_runner, monkeypatch, dataset, expected):
    benchmark, code, problem = isolated_runner
    origins = {"case_tiny": "benchmark_fn_fp", "case_pair": "correlation_pair", "case_numeric": "numerical_challenges"}
    for case, suffix in (("case_pair", "private_data/correlation_pair"), ("case_numeric", "private_data")):
        public = benchmark / "triton_eval_cases" / case
        public.mkdir()
        (public / "kernel.py").write_text(code)
        (public / "problem.txt").write_text(problem)
        private = benchmark / "single_call_vs_tools_challenges" / suffix
        private.mkdir(parents=True, exist_ok=True)
        (private / "validation_gpu.json").write_text(json.dumps({"cases": {case: {
            "ground_truth": "trust", "kernel_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "problem_sha256": hashlib.sha256(problem.encode()).hexdigest()}}}))
    (benchmark / "case_map.json").write_text(json.dumps({"case_details": {
        name: {"dataset": origin} for name, origin in origins.items()}}))
    requests, _ = fake_openai(monkeypatch, response('{"verdict":"trust"}', "stop"))
    runner.main(["--all", "--dataset", dataset, "--trial", "shared_directory"])
    assert len(requests) == len(expected)
    records = [read(path) for path in (benchmark / "traces_glm").glob("*/*/*/trace_meta.json")]
    assert sorted(row["case"] for row in records) == expected
    assert all(row["dataset"] == origins[row["case"]] for row in records)


def test_archived_pilot_stops_before_client_or_trace_creation(isolated_runner, monkeypatch):
    benchmark, _, _ = isolated_runner
    requests, constructors = fake_openai(monkeypatch, response('{"verdict":"trust"}', "stop"))
    with pytest.raises(ValueError, match="archived"):
        runner.main(["--all", "--dataset", "numerical_pilot"])
    with pytest.raises(ValueError, match="archived"):
        runner.run_one("case_82", dataset="numerical_pilot", trial="r1", max_tokens=1000)
    assert requests == constructors == []
    assert not (benchmark / "traces_glm").exists()


def test_original_baseline_loader_excludes_pair_in_shared_directory(isolated_runner, monkeypatch):
    from benchmark_fn_fp.eval_scripts import common
    benchmark, code, problem = isolated_runner
    pair = benchmark / "triton_eval_cases/case_pair"
    pair.mkdir()
    (pair / "kernel.py").write_text(code)
    (pair / "problem.txt").write_text(problem)
    for name in ("case_tiny", "case_pair"):
        (benchmark / "triton_eval_cases" / name / "meta.json").write_text("{}")
    key = benchmark / "triton/case_tiny"
    key.mkdir(parents=True)
    (key / "meta.json").write_text(json.dumps({
        "group": "FP", "seed_class": "FP1", "kernel_family": "add",
        "reference": "fixture", "expected": {"correct_verdict": "CORRECT"}}))
    case_map = benchmark / "case_map.json"
    case_map.write_text(json.dumps({"cases": {"case_tiny": "case_tiny"}, "case_details": {
        "case_tiny": {"dataset": "benchmark_fn_fp"}, "case_pair": {"dataset": "correlation_pair"}}}))
    monkeypatch.setattr(common, "CASE_MAP_PATH", case_map)
    monkeypatch.setattr(common, "CASES_DIR", benchmark / "triton_eval_cases")
    monkeypatch.setattr(common, "ANSWER_KEY_DIR", benchmark / "triton")
    assert [case.name for case in common.load_cases()] == ["case_tiny"]


def test_run_one_saves_complete_trace_and_refuses_duplicate_before_api(isolated_runner, monkeypatch):
    benchmark, code, problem = isolated_runner
    trial = "test_completed"
    dest = benchmark / "traces_glm" / "case_tiny" / "single_call" / trial
    answer = {"verdict": "trust", "confidence": 0.8, "reason": "The supplied input satisfies the contract."}
    raw = response(json.dumps(answer), "stop")

    def before_response(request):
        assert read(dest / "request.json") == request
        assert read(dest / "trace_meta.json")["status"] == "running"
        call, = (dest / "llm_calls").iterdir()
        assert read(call / "request.json") == request
        assert not (call / "response.json").exists()

    requests, constructors = fake_openai(monkeypatch, raw, before_response=before_response)
    result = runner.run_one("case_tiny", dataset="benchmark_fn_fp", trial=trial, max_tokens=1024)
    assert len(requests) == len(constructors) == 1
    assert constructors[0]["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert constructors[0]["max_retries"] == 0
    request = requests[0]
    assert request["model"] == runner.MODEL
    assert request["max_tokens"] == 1024
    assert request["reasoning_effort"] == "low"
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][1]["content"] == runner.USER_TEMPLATE.format(problem=problem, kernel=code)
    assert read(dest / "request.json") == request
    assert (dest / "system_prompt.txt").read_text() == request["messages"][0]["content"]
    assert (dest / "user_prompt.txt").read_text() == request["messages"][1]["content"]
    assert read(dest / "raw_response.json") == raw
    call, = (dest / "llm_calls").iterdir()
    assert read(call / "response.json") == raw
    assert read(call / "metadata.json")["status"] == "completed"
    assert read(dest / "response_text.json") == answer
    assert read(dest / "verdict.json") == answer
    assert (dest / "response_thinking.txt").read_text() == raw["choices"][0]["message"]["reasoning_content"]
    transcript = (dest / "transcript.md").read_text()
    assert problem in transcript and code in transcript
    assert raw["choices"][0]["message"]["reasoning_content"] in transcript
    assert raw["choices"][0]["message"]["content"] in transcript
    assert '"stop_reason": "stop"' in transcript
    usage = read(dest / "usage.json")
    assert usage == result
    assert usage["response"] == answer
    assert usage["usage"] == {"input_tokens": 120, "output_tokens": 321}
    assert usage["stop_reason"] == "stop"
    assert usage["estimated_usd"] > 0
    meta = read(dest / "trace_meta.json")
    assert meta["status"] == "completed"
    assert meta["stop_reason"] == "stop"
    assert meta["provider"] == "fireworks"
    assert meta["dataset"] == "benchmark_fn_fp"
    assert meta["kernel_sha256"] == hashlib.sha256(code.encode()).hexdigest()
    assert meta["problem_sha256"] == hashlib.sha256(problem.encode()).hexdigest()
    assert meta["raw_api_capture"] is True
    assert meta["reasoning_effort"] == "low"
    assert meta["timeout_s"] == 1800
    assert meta["public_input_files"] == ["kernel.py", "problem.txt"]
    assert len(meta["runner_sha256"]) == 64
    runtime = read(dest / "runtime.json")
    assert runtime["verifier_sha256"] == meta["verifier_sha256"]
    assert runtime["python"] and "openai" in runtime["packages"]
    before = {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        runner.run_one("case_tiny", dataset="benchmark_fn_fp", trial=trial, max_tokens=1024)
    assert len(requests) == len(constructors) == 1
    assert {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()} == before
    assert all(b"offline-fireworks-test-key" not in content for content in before.values())


def test_source_only_and_tool_roles_receive_identical_complete_public_task(isolated_runner, monkeypatch):
    from verifier.agentic.agents.base import LLMAgent
    from verifier.agentic.state import RunState
    benchmark, _, _ = isolated_runner
    case = benchmark / "triton_eval_cases/case_tiny"
    # Put the reference and final contractual clause past the former 12k cap.
    code = "# Public implementation\n" * 700 + "def reference(x):\n    return x + 1\n"
    problem = "The input domain is general.\n" * 500 + "All legal inputs must satisfy the contract.\n"
    (case / "kernel.py").write_text(code)
    (case / "problem.txt").write_text(problem)
    requests, _ = fake_openai(monkeypatch, response('{"verdict":"trust"}', "stop"))
    runner.run_one("case_tiny", dataset="benchmark_fn_fp", trial="full_task", max_tokens=1024)
    assert requests[0]["messages"][1]["content"] == runner.USER_TEMPLATE.format(problem=problem, kernel=code)
    state = RunState()
    state.artifact = {"kernel_code": code, "problem_text": problem, "test_code": ""}
    for role in ("solo", "describer", "skeptic", "experimenter", "judge"):
        prompt = LLMAgent(role=role, instructions="Verify.", llm_client=None)._build_user_prompt(state=state)
        payload = prompt.split("=== Current Run State ===\n", 1)[1]
        artifact = json.JSONDecoder().raw_decode(payload)[0]["artifact"]
        recovered = "\n".join(line.split(": ", 1)[1] for line in artifact["kernel_code"].splitlines())
        assert recovered == code.rstrip("\n")
        assert artifact["problem_text"] == problem


def test_length_with_empty_content_remains_no_final_verdict(isolated_runner, monkeypatch):
    benchmark, _, _ = isolated_runner
    raw = response("", "length")
    requests, _ = fake_openai(monkeypatch, raw)
    result = runner.run_one("case_tiny", dataset="benchmark_fn_fp", trial="test_length", max_tokens=321)
    dest = benchmark / "traces_glm" / "case_tiny" / "single_call" / "test_length"
    assert len(requests) == 1
    assert result["response"]["verdict"] == "no_verdict"
    assert result["stop_reason"] == "length"
    assert result["usage"]["output_tokens"] == 321
    verdict = read(dest / "verdict.json")
    assert verdict == {"verdict": None, "status": "no_final_verdict"}
    assert (dest / "response_text.json").read_text() == ""
    assert read(dest / "raw_response.json") == raw
    call, = (dest / "llm_calls").iterdir()
    saved = read(call / "response.json")
    assert saved["choices"][0]["finish_reason"] == "length"
    assert saved["choices"][0]["message"]["reasoning_content"] == raw["choices"][0]["message"]["reasoning_content"]
    assert read(dest / "trace_meta.json")["stop_reason"] == "length"
    transcript = (dest / "transcript.md").read_text()
    assert "(no final text)" in transcript
    assert '"verdict": null' in transcript
    assert '"stop_reason": "length"' in transcript


def test_provider_default_omits_reasoning_setting_and_records_configuration(isolated_runner, monkeypatch):
    benchmark, _, _ = isolated_runner
    requests, _ = fake_openai(monkeypatch, response('{"verdict":"trust"}', "stop"))
    result = runner.run_one("case_tiny", dataset="benchmark_fn_fp", trial="default_control",
                            max_tokens=65536, reasoning_effort="default", timeout_s=900)
    assert "reasoning_effort" not in requests[0]
    assert result["reasoning_effort"] == "default"
    meta = read(benchmark / "traces_glm/case_tiny/single_call/default_control/trace_meta.json")
    assert meta["reasoning_effort"] == "default"
    assert meta["timeout_s"] == 900


def test_numerical_single_checks_freeze_before_client_creation(isolated_runner, monkeypatch):
    benchmark, _, _ = isolated_runner
    root = benchmark / "single_call_vs_tools_challenges"
    root.mkdir()
    case = benchmark / "triton_eval_cases/case_tiny"
    hashes = {f"{kind}_sha256": hashlib.sha256((case / filename).read_bytes()).hexdigest()
              for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt"))}
    (root / "validation_gpu.json").write_text(json.dumps({"cases": {
        "case_tiny": {"ground_truth": "trust", **hashes}}}))
    raw = response('{"verdict":"trust"}', "stop")
    requests, constructors = fake_openai(monkeypatch, raw)
    code = (case / "kernel.py").read_text()
    (case / "kernel.py").write_text(code + "# changed\n")
    with pytest.raises(ValueError, match="Frozen case changed"):
        runner.run_one("case_tiny", dataset="single_call_vs_tools_challenges", trial="challenge", max_tokens=1000)
    assert requests == constructors == []
    assert not (benchmark / "traces_glm").exists()
    (case / "kernel.py").write_text(code)
    result = runner.run_one("case_tiny", dataset="single_call_vs_tools_challenges", trial="challenge", max_tokens=1000)
    assert result["response"]["verdict"] == "trust"
    assert len(requests) == len(constructors) == 1
    meta = read(benchmark / "traces_glm/case_tiny/single_call/challenge/trace_meta.json")
    assert meta["dataset"] == "numerical_challenges"
