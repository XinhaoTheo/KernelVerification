"""The unified report preserves historical cohorts, prices and source evidence."""
import json
from pathlib import Path

import pytest

from benchmark_fn_fp.eval_scripts.correlation_pair import report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "correlation_pair"
    cases = {}
    for case, truth in (("case_36", "reject"), ("case_37", "trust")):
        public = root.parent / "triton_eval_cases" / case
        public.mkdir(parents=True)
        code = "def run(x):\n    return x\n"
        problem = "Check the final error.\n" + report.NEUTRAL_NOTE
        (public / "kernel.py").write_text(code)
        (public / "problem.txt").write_text(problem)
        cases[case] = {"ground_truth": truth, "errors": [0.2 if truth == "reject" else 0.05],
                       "kernel_sha256": report.digest(code), "problem_sha256": report.digest(problem)}
    write(root / "private_data/validation_gpu.json", {"cases": cases, "environment": {"gpu": "T4"}})
    (root / "README.md").write_text("Manual protocol.\n" + report.BEGIN_RESULTS
                                   + "\nOld generated text.\n" + report.END_RESULTS + "\nManual audit.\n")
    return root


def single(root, case="case_36", trial="r1", batch="r1", cap=32768,
           variant="original", provider="fireworks", stop="stop", verdict="reject", saved=0.123):
    model = "accounts/fireworks/models/glm-5p3" if provider == "fireworks" else "claude-opus-5"
    tree = "traces_glm" if provider == "fireworks" else "traces_opus5"
    path = root.parent / tree / case / "single_call" / trial
    label = report.read(root / "private_data/validation_gpu.json")["cases"][case]
    public = root.parent / "triton_eval_cases" / case
    code, problem = (public / "kernel.py").read_text(), (public / "problem.txt").read_text()
    if variant == "neutral_contract":
        problem = problem.replace(report.NEUTRAL_NOTE, "")
    usage = {"usage": {"input_tokens": 100, "output_tokens": 200}, "estimated_usd": saved,
             "stop_reason": stop, "max_tokens": cap, "prompt_variant": variant,
             "response": {"verdict": verdict, "confidence": 0.8},
             "kernel_sha256": label["kernel_sha256"], "problem_sha256": report.digest(problem)}
    if variant == "neutral_contract":
        usage["source_problem_sha256"] = label["problem_sha256"]
    write(path / "trace_meta.json", {"case": case, "arm": "single_call", "trial": trial,
        "original_trial": batch, "dataset": "correlation_pair", "provider": provider,
        "model": model, "max_tokens": cap, "status": "historical"})
    write(path / "usage.json", usage)
    (path / "user_prompt.txt").write_text(problem + "\n" + code)
    (path / "response_text.json").write_text(json.dumps(usage["response"]))
    if provider == "fireworks":
        write(path / "request.json", {"model": model, "max_tokens": cap})
        write(path / "raw_response.json", {"usage": {"prompt_tokens": 100, "completion_tokens": 200}})
    return path


def test_shared_trees_keep_batches_budgets_and_saved_prices_separate(root):
    single(root)
    single(root, trial="r2", batch="r2_64k", cap=65536, saved=0.234)
    single(root, case="case_37", stop="length", verdict="no_verdict", saved=0.345)
    single(root, trial="r3", batch="neutral1", provider="anthropic", variant="neutral_contract", saved=0.456)
    result = report.derive(root)
    assert result["attempts"] == 4
    assert result["outcomes"]["correct"] == 3 and result["outcomes"]["token_limit"] == 1
    assert result["estimated_api_usd"] == pytest.approx(1.158)
    assert {group["original_trial"] for group in result["groups"]} == {"r1", "r2_64k", "neutral1"}
    assert {row["max_tokens"] for row in result["rows"]} == {32768, 65536}
    assert all(row["cost_source"] == "saved usage.estimated_usd" for row in result["rows"])
    assert result["raw_capture"] == {"complete_api": 3, "text_and_usage_only": 1}
    for row in result["rows"]:
        assert (root / row["trace_link"]).resolve().exists()
    text = report.render(result)
    assert "AUDIT.md" not in text and "r2_64k" in text and "neutral1" in text


def test_neutral_ablation_does_not_exempt_unrelated_hash_changes(root):
    path = single(root, provider="anthropic", variant="neutral_contract")
    assert report.derive(root)["attempts"] == 1
    usage = report.read(path / "usage.json")
    usage["problem_sha256"] = "unrelated contract"
    write(path / "usage.json", usage)
    with pytest.raises(ValueError, match="problem_sha256 mismatch"):
        report.derive(root)


def test_public_source_and_historical_tool_artifact_hashes_still_checked(root):
    single(root)
    (root.parent / "triton_eval_cases/case_36/kernel.py").write_text("changed")
    with pytest.raises(ValueError, match="Frozen public source changed"):
        report.derive(root)


def test_tool_history_uses_historical_profile_without_claiming_raw_capture(root):
    path = root.parent / "traces_opus5/case_37/solo/r1"
    public = root.parent / "triton_eval_cases/case_37"
    write(path / "trace_meta.json", {"case": "case_37", "arm": "solo", "trial": "r1",
        "original_trial": "r1", "dataset": "correlation_pair", "provider": "anthropic",
        "model": "claude-opus-5", "status": "historical"})
    run = {"artifact": {"kernel_code": (public / "kernel.py").read_text(),
                        "problem_text": (public / "problem.txt").read_text()},
           "history": [{"usage": {"input_tokens": 100, "output_tokens": 200}}],
           "verdict": {"verdict": "trust", "confidence": 0.9}}
    write(path / "run.json", run)
    row, = report.derive(root)["rows"]
    assert row["usd"] == pytest.approx(0.0055)
    assert row["cost_source"] == "historical project model profile"
    assert row["raw_capture"] == "history_only" and row["max_tokens"] is None
    run["artifact"]["kernel_code"] = "different kernel"
    write(path / "run.json", run)
    with pytest.raises(ValueError, match="artifact.kernel_code mismatch"):
        report.derive(root)


def test_default_replaces_only_readme_block_json_export_is_explicit(root):
    single(root)
    report.main(root=root)
    content = (root / "README.md").read_text()
    assert content.startswith("Manual protocol.\n" + report.BEGIN_RESULTS)
    assert content.endswith(report.END_RESULTS + "\nManual audit.\n")
    assert "Old generated text." not in content and "Every attempt" in content
    assert not (root / "REPORT.md").exists()
    assert not (root / "scoreboard.json").exists()
    output = root / "private_data/reports/scoreboard.json"
    assert not output.exists()
    report.main(root=root, export_json=True)
    assert report.read(output)["attempts"] == 1


@pytest.mark.parametrize("content", ["Manual notes only", report.BEGIN_RESULTS,
    report.END_RESULTS + report.BEGIN_RESULTS, report.BEGIN_RESULTS * 2 + report.END_RESULTS])
def test_missing_or_invalid_readme_markers_refuse_all_outputs(root, content):
    readme = root / "README.md"
    readme.write_text(content)
    with pytest.raises(ValueError, match="refusing to overwrite manual content"):
        report.main(root=root, export_json=True)
    assert readme.read_text() == content
    assert not (root / "private_data/reports/scoreboard.json").exists()
