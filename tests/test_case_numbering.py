"""Renaming cases must preserve historical evidence and correct label matching."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from benchmark_fn_fp.eval_scripts.case_registry import case_sort_key, resolve_trace_case
from benchmark_fn_fp.eval_scripts.datasets import checked_case_hashes
from benchmark_fn_fp.eval_scripts.traces import iter_trace_records


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def registry():
    return {"cases": {}, "case_details": {
        "case_36": {"dataset": "correlation_pair", "previous_id": "case_a", "source_name": "case_a"},
        "case_37": {"dataset": "correlation_pair", "previous_id": "case_b", "source_name": "case_b"},
    }, "retired_ids": ["case_03"], "next_case_number": 106}


def test_moved_trace_keeps_raw_old_metadata_and_resolves_new_case(tmp_path):
    write(tmp_path / "case_map.json", registry())
    trace = tmp_path / "traces_glm/case_36/solo/r1"
    write(trace / "trace_meta.json", {"case": "case_a", "dataset": "correlation_pair",
        "arm": "solo", "trial": "r1", "model": "accounts/fireworks/models/glm-5p3"})
    write(trace / "run.json", {"artifact": {"entry": "/root/cases/case_a"}, "history": []})
    before = {path.name: path.read_bytes() for path in trace.iterdir()}
    rows = list(iter_trace_records(tmp_path))
    assert len(rows) == 1
    assert rows[0]["case"] == "case_36"
    assert rows[0]["original_case"] == "case_a"
    assert rows[0]["metadata"]["case"] == "case_a"
    assert rows[0]["dataset"] == "correlation_pair"
    assert {path.name: path.read_bytes() for path in trace.iterdir()} == before


@pytest.mark.parametrize("metadata, match", [
    ({"case": "case_b", "dataset": "correlation_pair"}, "case mismatch"),
    ({"case": "case_a", "dataset": "evidence_challenges"}, "dataset mismatch"),
])
def test_moved_trace_rejects_another_case_or_dataset(metadata, match):
    with pytest.raises(ValueError, match=match):
        resolve_trace_case("case_36", metadata, registry())


@pytest.mark.parametrize("dataset_name", ["correlation_pair", "numerical_challenges", "evidence_challenges"])
def test_private_validation_takes_precedence_over_stale_legacy_copy(tmp_path, dataset_name):
    dataset = tmp_path / "benchmark_fn_fp" / dataset_name
    directory = tmp_path / "benchmark_fn_fp/triton_eval_cases"
    case = directory / "case_38"
    case.mkdir(parents=True)
    hashes = {}
    for kind, filename, text in (("kernel", "kernel.py", "def run(): return 1\n"),
                                 ("problem", "problem.txt", "Return one.\n")):
        (case / filename).write_text(text)
        hashes[f"{kind}_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    write(dataset / "private_data/validation_gpu.json", {"cases": {
        "case_38": {"ground_truth": "trust", **hashes}}})
    write(dataset / "validation_gpu.json", {"cases": {
        "case_38": {"ground_truth": "trust", "kernel_sha256": "stale"}}})
    assert checked_case_hashes(tmp_path, dataset_name, "case_38") == hashes


@pytest.mark.parametrize("existing_case", ["case_36", "case_37"])
def test_pair_builder_cannot_overwrite_cases_after_trace_cleanup(tmp_path, monkeypatch, existing_case):
    from benchmark_fn_fp.eval_scripts.single_call_vs_tools_challenges.correlation_pair import build
    dataset = tmp_path / "benchmark_fn_fp/single_call_vs_tools_challenges"
    monkeypatch.setattr(build, "ROOT", dataset)
    case = dataset.parent / "triton_eval_cases" / existing_case
    case.mkdir(parents=True)
    source = case / "kernel.py"
    source.write_text("frozen kernel")
    assert not (dataset / "traces").exists()
    with pytest.raises(RuntimeError, match="overwriting frozen cases"):
        build.main()
    assert source.read_text() == "frozen kernel"
    assert not (dataset / "private_data/correlation_pair").exists()


def test_numeric_sort_handles_three_digit_cases():
    assert sorted(["case_105", "case_82", "case_02"], key=case_sort_key) == [
        "case_02", "case_82", "case_105"]


def test_original_builder_allocates_globally_and_preserves_source_names(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/generation/generators/build_eval_cases.py"
    spec = importlib.util.spec_from_file_location("numbered_case_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    root = tmp_path / "benchmark_fn_fp"
    for attribute, value in {"REPO": tmp_path, "SOURCE_DIR": root / "triton",
                             "EVAL_DIR": root / "triton_eval_cases", "MAP_PATH": root / "case_map.json"}.items():
        monkeypatch.setattr(builder, attribute, value)
    for directory, original, value in (("case_01", "fn1_existing_source", 1),
                                       ("fn2_new_source", "fn2_new_source", 2)):
        folder = root / "triton" / directory
        write(folder / "meta.json", {"name": original, "kernel_family": "addition"})
        (folder / "kernel.py").write_text(f'"""Triton kernel under test: {original}."""\n\ndef run(): return {value}\n')
        (folder / "problem.txt").write_text(f"Return {value}.\n")
    initial = {"cases": {"case_01": "case_01"}, "retired_ids": ["case_03"],
        "next_case_number": 106, "custom_provenance": {"keep": True}, "case_details": {
            "case_01": {"dataset": "benchmark_fn_fp", "previous_id": "case_01", "source_name": "fn1_existing_source"},
            "case_36": {"dataset": "correlation_pair", "public_dir": "triton_eval_cases/case_36"},
            "case_105": {"dataset": "numerical_pilot", "previous_id": "case_24"}}}
    pair = root / "triton_eval_cases/case_36"
    pair.mkdir(parents=True)
    (pair / "kernel.py").write_text("# Another frozen dataset: buggy is ordinary text here.\n")
    (pair / "problem.txt").write_text("Frozen correlation contract.\n")
    write(pair / "meta.json", {"name": "case_36"})
    frozen_pair = {p.name: p.read_bytes() for p in pair.iterdir()}
    write(root / "case_map.json", initial)
    assert builder.main() == 0
    result = json.loads((root / "case_map.json").read_text())
    assert result["cases"] == {"case_01": "case_01", "case_106": "case_106"}
    assert result["next_case_number"] == 107
    assert result["retired_ids"] == ["case_03"]
    assert result["custom_provenance"] == {"keep": True}
    assert result["case_details"]["case_105"] == initial["case_details"]["case_105"]
    assert result["case_details"]["case_36"] == initial["case_details"]["case_36"]
    assert {p.name: p.read_bytes() for p in pair.iterdir()} == frozen_pair
    assert result["case_details"]["case_106"]["source_name"] == "fn2_new_source"
    assert not (root / "triton/fn2_new_source").exists()
    for case in ("case_01", "case_106"):
        assert "Triton kernel under test" not in (root / "triton_eval_cases" / case / "kernel.py").read_text()
