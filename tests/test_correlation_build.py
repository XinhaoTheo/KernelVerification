"""CPU reconstruction shares the public tree without overwriting other cases."""
import json

import pytest

from benchmark_fn_fp.eval_scripts.single_call_vs_tools_challenges.correlation_pair import build, validate_cpu


def test_fresh_pair_rebuild_and_cpu_validation_preserve_other_cases(tmp_path, monkeypatch):
    dataset = tmp_path / "benchmark_fn_fp/single_call_vs_tools_challenges"
    public = dataset.parent / "triton_eval_cases"
    unrelated = public / "case_01/kernel.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("frozen unrelated kernel\n")
    other_answers = dataset / "private_data/validation_gpu.json"
    other_answers.parent.mkdir(parents=True)
    other_answers.write_text('{"other_collection_cases": true}')
    monkeypatch.setattr(build, "ROOT", dataset)
    monkeypatch.setattr(validate_cpu, "ROOT", dataset)
    build.main()
    assert unrelated.read_text() == "frozen unrelated kernel\n"
    assert other_answers.read_text() == '{"other_collection_cases": true}'
    assert not (dataset / "eval_cases").exists()
    before = {str(path.relative_to(public)): path.read_bytes()
              for path in public.rglob("*") if path.is_file()}
    assert (public / "case_36/kernel.py").exists()
    assert (public / "case_37/problem.txt").exists()
    validate_cpu.main()
    validation = json.loads((dataset / "private_data/correlation_pair/validation_cpu.json").read_text())
    assert validation["checks_passed"]
    assert validation["cases"]["case_36"]["label"] == "reject"
    assert validation["cases"]["case_37"]["label"] == "trust"
    assert {str(path.relative_to(public)): path.read_bytes()
            for path in public.rglob("*") if path.is_file()} == before
    with pytest.raises(RuntimeError, match="overwriting frozen cases"):
        build.main()
    assert other_answers.read_text() == '{"other_collection_cases": true}'
    assert {str(path.relative_to(public)): path.read_bytes()
            for path in public.rglob("*") if path.is_file()} == before


def test_existing_private_answers_block_reconstruction_before_public_writes(tmp_path, monkeypatch):
    dataset = tmp_path / "benchmark_fn_fp/single_call_vs_tools_challenges"
    private = dataset / "private_data/correlation_pair"
    private.mkdir(parents=True)
    sentinel = private / "answer_key_cpu.json"
    sentinel.write_text('{"frozen": true}')
    monkeypatch.setattr(build, "ROOT", dataset)
    with pytest.raises(RuntimeError, match="overwriting frozen cases"):
        build.main()
    assert sentinel.read_text() == '{"frozen": true}'
    assert not (dataset.parent / "triton_eval_cases").exists()
