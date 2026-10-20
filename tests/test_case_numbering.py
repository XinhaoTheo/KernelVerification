"""Renaming cases must preserve historical evidence and correct label matching."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from benchmark_fn_fp.eval_scripts.case_registry import (case_sort_key, resolve_trace_case,
                                                       validation_cases)
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


@pytest.mark.parametrize("case_index", [0, 1])
def test_pair_builder_cannot_overwrite_cases_after_trace_cleanup(tmp_path, monkeypatch, case_index):
    from benchmark_fn_fp.eval_scripts.single_call_vs_tools_challenges.correlation_pair import build
    existing_case = build.CASE_IDS[case_index]
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


def compacted_registry():
    return {"cases": {}, "case_details": {
        "case_36": {"dataset": "real_kernel_challenges", "previous_id": "case_109",
                    "aliases": ["case_109"], "validation_case_id": "case_109",
                    "public_dir": "triton_eval_cases/case_36"},
        "case_93": {"dataset": "correlation_pair", "previous_id": "case_36",
                    "aliases": ["case_36", "case_a"], "validation_case_id": "case_36",
                    "status": "archived", "archive_root": "archive/input_specific",
                    "public_dir": "archive/input_specific/triton_eval_cases/case_93"},
    }}


def test_second_renumber_keeps_all_aliases_and_separates_archived_trace_identity(tmp_path):
    write(tmp_path / "case_map.json", compacted_registry())
    paths = []
    for prefix, case, old, dataset in (
        ("", "case_36", "case_109", "real_kernel_challenges"),
        ("archive/input_specific", "case_93", "case_a", "correlation_pair"),
    ):
        trace = tmp_path / prefix / "traces_glm" / case / "solo/r1"
        write(trace / "trace_meta.json", {"case": old, "dataset": dataset,
              "model": "accounts/fireworks/models/glm-5p3", "arm": "solo", "trial": "r1"})
        write(trace / "run.json", {"artifact": {"entry": f"/root/cases/{old}"}})
        paths.extend(trace.iterdir())
    before = {path: path.read_bytes() for path in paths}
    assert [row["case"] for row in iter_trace_records(tmp_path)] == ["case_36"]
    rows = {row["case"]: row for row in iter_trace_records(tmp_path, include_archived=True)}
    assert set(rows) == {"case_36", "case_93"}
    assert rows["case_93"]["original_case"] == "case_a"
    assert rows["case_93"]["case_aliases"] == ["case_36", "case_93", "case_a"]
    assert {path: path.read_bytes() for path in paths} == before
    with pytest.raises(ValueError, match="dataset mismatch"):
        resolve_trace_case("case_36", {"case": "case_36", "dataset": "correlation_pair"},
                           compacted_registry())


def test_compacted_source_checks_use_frozen_id_and_never_numeric_coincidence(tmp_path):
    benchmark = tmp_path / "benchmark_fn_fp"
    write(benchmark / "case_map.json", compacted_registry())
    public = benchmark / "triton_eval_cases/case_36"
    public.mkdir(parents=True)
    hashes = {}
    for kind, filename, content in (("kernel", "kernel.py", "def run(x): return x\n"),
                                    ("problem", "problem.txt", "Return x unchanged.\n")):
        (public / filename).write_text(content)
        hashes[f"{kind}_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    frozen = benchmark / "real_kernel_challenges/private_data/validation_gpu.json"
    write(frozen, {"cases": {"case_109": {"ground_truth": "trust", **hashes},
                              "case_36": {"ground_truth": "reject", "kernel_sha256": "wrong"}}})
    before = frozen.read_bytes()
    assert checked_case_hashes(tmp_path, "real_kernel_challenges", "case_36") == hashes
    assert validation_cases(benchmark, "real_kernel_challenges") == {
        "case_36": {"ground_truth": "trust", **hashes}}
    assert frozen.read_bytes() == before
    write(frozen, {"cases": {"case_36": {"ground_truth": "trust", **hashes}}})
    with pytest.raises(ValueError, match="No GPU validation"):
        checked_case_hashes(tmp_path, "real_kernel_challenges", "case_36")


def test_archived_author_keys_resolve_current_paths_without_rewriting_evidence(tmp_path):
    from benchmark_fn_fp.eval_scripts.single_call_vs_tools_challenges.archive_paths import (
        canonical_case_rows, frozen_case_id, frozen_case_rows,
    )

    benchmark = tmp_path / "benchmark_fn_fp"
    collection = benchmark / "archive/input_specific/single_call_vs_tools_challenges"
    write(benchmark / "case_map.json", {"case_details": {
        "case_99": {"dataset": "correlation_pair", "validation_case_id": "case_36"},
        "case_100": {"dataset": "correlation_pair", "validation_case_id": "case_37"},
        "case_36": {"dataset": "real_kernel_challenges", "validation_case_id": "case_109"},
    }})
    key = collection / "private_data/correlation_pair/answer_key_cpu.json"
    frozen = {"case_36": {"cpu_ground_truth": "reject"},
              "case_37": {"cpu_ground_truth": "trust"}}
    write(key, {"cases": frozen})
    before = key.read_bytes()

    current = canonical_case_rows(collection, "correlation_pair",
                                  json.loads(before)["cases"])
    assert current == {"case_99": frozen["case_36"], "case_100": frozen["case_37"]}
    assert frozen_case_id(collection, "correlation_pair", "case_99") == "case_36"
    assert frozen_case_rows(collection, "correlation_pair", current) == frozen
    assert key.read_bytes() == before
    with pytest.raises(ValueError, match="Unresolved or ambiguous"):
        canonical_case_rows(collection, "correlation_pair", {"case_missing": {}})
    with pytest.raises(ValueError, match="dataset mismatch"):
        frozen_case_id(collection, "correlation_pair", "case_36")


def test_compacted_scoreboard_uses_dataset_scoped_frozen_labels(tmp_path):
    from benchmark_fn_fp.eval_scripts.summarize_traces import build_report

    write(tmp_path / "case_map.json", compacted_registry())
    for prefix, case, old, dataset, verdict in (
        ("", "case_36", "case_109", "real_kernel_challenges", "reject"),
        ("archive/input_specific", "case_93", "case_36", "correlation_pair", "trust"),
    ):
        trace = tmp_path / prefix / "traces_glm" / case / "single_call/r1"
        write(trace / "trace_meta.json", {"case": old, "dataset": dataset,
              "model": "accounts/fireworks/models/glm-5p3", "arm": "single_call", "trial": "r1"})
        write(trace / "usage.json", {"usage": {"input_tokens": 1, "output_tokens": 1},
                                    "response": {"verdict": verdict}, "stop_reason": "stop"})
    write(tmp_path / "real_kernel_challenges/private_data/validation_gpu.json",
          {"cases": {"case_109": {"ground_truth": "reject"}}})
    write(tmp_path / "archive/input_specific/single_call_vs_tools_challenges/private_data/correlation_pair/validation_gpu.json",
          {"cases": {"case_36": {"ground_truth": "trust"}}})
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    report = build_report(benchmark_dir=tmp_path, include_archived=True)
    rows = [row for group in report["arms"].values() for row in group["per_case"].values()]
    assert {row["case"]: row["outcome"] for row in rows} == {
        "case_36": "correct", "case_93": "correct"}
    assert {path: path.read_bytes() for path in before} == before


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
