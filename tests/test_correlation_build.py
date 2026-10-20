"""CPU reconstruction shares the public tree without overwriting other cases."""
import json

import pytest

from benchmark_fn_fp.eval_scripts.single_call_vs_tools_challenges.correlation_pair import build, validate_cpu
from benchmark_fn_fp.eval_scripts.single_call_vs_tools_challenges.archive_paths import (
    ARCHIVE, BENCHMARK, COLLECTIONS, archived_collection,
)


def test_legacy_project_destinations_route_to_archive_but_fixtures_stay_local(tmp_path):
    for collection in COLLECTIONS:
        assert archived_collection(BENCHMARK / collection) == ARCHIVE / collection
        fixture = tmp_path / "benchmark_fn_fp" / collection
        assert archived_collection(fixture) == fixture
    assert build.ROOT == validate_cpu.ROOT == ARCHIVE / "single_call_vs_tools_challenges"


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
    first, second = build.CASE_IDS
    assert (public / first / "kernel.py").exists()
    assert (public / second / "problem.txt").exists()
    validate_cpu.main()
    validation = json.loads((dataset / "private_data/correlation_pair/validation_cpu.json").read_text())
    assert validation["checks_passed"]
    assert validation["cases"][first]["label"] == "reject"
    assert validation["cases"][second]["label"] == "trust"
    assert {str(path.relative_to(public)): path.read_bytes()
            for path in public.rglob("*") if path.is_file()} == before
    with pytest.raises(RuntimeError, match="overwriting frozen cases"):
        build.main()
    assert other_answers.read_text() == '{"other_collection_cases": true}'
    assert {str(path.relative_to(public)): path.read_bytes()
            for path in public.rglob("*") if path.is_file()} == before


def test_cpu_validation_uses_current_public_paths_and_frozen_npz_names(tmp_path, monkeypatch):
    dataset = tmp_path / "benchmark_fn_fp/single_call_vs_tools_challenges"
    monkeypatch.setattr(build, "ROOT", dataset)
    monkeypatch.setattr(validate_cpu, "ROOT", dataset)
    build.main()
    public = dataset.parent / "triton_eval_cases"
    private = dataset / "private_data/correlation_pair"
    frozen_files = {p: p.read_bytes() for p in private.iterdir()}
    mapping = dict(zip(build.CASE_IDS, ("case_201", "case_202")))
    for old, current in mapping.items():
        (public / old).rename(public / current)
    registry = {"case_details": {
        current: {"dataset": "correlation_pair", "validation_case_id": old,
                  "public_dir": f"triton_eval_cases/{current}"}
        for old, current in mapping.items()
    }}
    (dataset.parent / "case_map.json").write_text(json.dumps(registry))

    validate_cpu.main()

    result = json.loads((private / "validation_cpu.json").read_text())
    assert result["checks_passed"]
    assert set(result["cases"]) == set(build.CASE_IDS)
    assert {p: p.read_bytes() for p in frozen_files} == frozen_files
    assert all((public / current / "kernel.py").is_file() for current in mapping.values())


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
