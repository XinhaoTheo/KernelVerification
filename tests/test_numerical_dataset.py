"""New datasets stay answer-free and immutable before any paid submission."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmark_fn_fp.eval_scripts.datasets import cases_dir, checked_case_hashes


def frozen_case(tmp_path, dataset="numerical_challenges"):
    root = tmp_path / "benchmark_fn_fp" / dataset
    case = root / "eval_cases" / "case_38"
    case.mkdir(parents=True)
    (case / "kernel.py").write_text("def run(x): return x\n")
    (case / "problem.txt").write_text("Return x unchanged.\n")
    row = {"ground_truth": "trust", **{
        f"{kind}_sha256": hashlib.sha256((case / filename).read_bytes()).hexdigest()
        for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt"))}}
    filename = "answer_key.json" if dataset == "numerical_pilot" else "validation_gpu.json"
    (root / filename).write_text(json.dumps({"cases": {"case_38": row}}))
    return case, row


@pytest.mark.parametrize("dataset", ["correlation_pair", "numerical_challenges", "evidence_challenges",
                                    "numerical_pilot"])
def test_frozen_sources_checked_without_releasing_labels(tmp_path, dataset):
    case, row = frozen_case(tmp_path, dataset)
    hashes = checked_case_hashes(tmp_path, dataset, "case_38")
    assert hashes == {key: value for key, value in row.items() if key.endswith("sha256")}
    assert cases_dir(tmp_path, dataset) == case.parent
    (case / "problem.txt").write_text("Changed numerical contract.\n")
    with pytest.raises(ValueError, match="Frozen case changed"):
        checked_case_hashes(tmp_path, dataset, "case_38")


def test_missing_or_invalid_validation_is_not_silently_accepted(tmp_path):
    case, _ = frozen_case(tmp_path)
    validation = case.parent.parent / "validation_gpu.json"
    validation.unlink()
    with pytest.raises(FileNotFoundError):
        checked_case_hashes(tmp_path, "numerical_challenges", "case_38")
    validation.write_text('{"cases": {}}')
    with pytest.raises(ValueError, match="No GPU validation"):
        checked_case_hashes(tmp_path, "numerical_challenges", "case_38")
    with pytest.raises(ValueError, match="Unknown dataset"):
        cases_dir(tmp_path, "misspelled")
    with pytest.raises(ValueError, match="Invalid case name"):
        checked_case_hashes(tmp_path, "numerical_challenges", "../case_38")


def test_pilot_requires_its_existing_answer_key_and_usable_label(tmp_path):
    case, row = frozen_case(tmp_path, "numerical_pilot")
    root = case.parent.parent
    (root / "answer_key.json").unlink()
    # A similarly shaped unrelated freeze must not silently replace the pilot's key.
    (root / "validation_gpu.json").write_text(json.dumps({"cases": {"case_38": row}}))
    with pytest.raises(FileNotFoundError):
        checked_case_hashes(tmp_path, "numerical_pilot", "case_38")
    (root / "answer_key.json").write_text(json.dumps({"cases": {
        "case_38": {**row, "ground_truth": "unknown"}}}))
    with pytest.raises(ValueError, match="no usable label"):
        checked_case_hashes(tmp_path, "numerical_pilot", "case_38")


def test_validator_refuses_cpu_gpu_disagreements(monkeypatch):
    class Image:
        @classmethod
        def debian_slim(cls, **kwargs): return cls()
        def pip_install(self, *args, **kwargs): return self
        def add_local_dir(self, *args, **kwargs): return self

    class App:
        def __init__(self, *args): pass
        def function(self, **kwargs): return lambda function: function
        def local_entrypoint(self): return lambda function: function

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(App=App, Image=Image))
    path = Path(__file__).resolve().parents[1] / "benchmark_fn_fp/numerical_challenges/validate_modal.py"
    spec = importlib.util.spec_from_file_location("offline_numerical_validator", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    row = {"input_sha256": ["inputs"], "kernel_sha256": "kernel", "problem_sha256": "problem",
           "ground_truth": "trust"}
    cpu = {"case_38": {**row, "cpu_ground_truth": "trust"}}
    validator.check_cpu_agreement({"cases": {"case_38": row}}, cpu)
    for key in ("input_sha256", "kernel_sha256", "problem_sha256", "ground_truth"):
        with pytest.raises(ValueError, match="CPU/GPU"):
            validator.check_cpu_agreement({"cases": {"case_38": {**row, key: "wrong"}}}, cpu)
