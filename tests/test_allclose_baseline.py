"""Protect comparison semantics and the frozen public-input baseline protocol."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "benchmark_fn_fp" / "eval_scripts"
sys.path.insert(0, str(SCRIPTS))
from baseline_allclose_protocol import comparison_profile, compare_outputs, execute_payload, generator_defaults


def test_worker_import_from_root_needs_no_repository_or_local_mount():
    # Modal relocates the module to /root, where HERE.parents[1] does not exist.
    # Run a fresh interpreter to avoid an already imported local Modal app.
    program = '''import pathlib, sys, types
sys.path.insert(0, sys.argv[1])
class Image:
    @classmethod
    def debian_slim(cls, **kwargs): return cls()
    def pip_install(self, *args): return self
    def add_local_file(self, *args):
        raise AssertionError('remote import must not register local source mounts')
class App:
    def __init__(self, name): pass
    def function(self, **kwargs): return lambda function: function
    def local_entrypoint(self): return lambda function: function
sys.modules['modal'] = types.SimpleNamespace(is_local=lambda: False, Image=Image, App=App)
namespace = {'__file__': '/root/baseline1_allclose_modal.py', '__name__': 'remote_baseline'}
source = pathlib.Path(sys.argv[1], 'baseline1_allclose_modal.py').read_text()
exec(compile(source, namespace['__file__'], 'exec'), namespace)
assert namespace['REPO_ROOT'] == pathlib.Path('/root')
import baseline_allclose_protocol
baseline_allclose_protocol.execute_payload = lambda payload: {'case': payload['case']}
assert namespace['run_case']({'case': 'case_01'}) == {'case': 'case_01'}
'''
    result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_active_modern_case_has_explicit_public_comparison():
    registry = json.loads((ROOT / "benchmark_fn_fp" / "case_map.json").read_text())
    modern = [detail for detail in registry["case_details"].values()
              if detail.get("status", "active") == "active" and detail["dataset"] == "real_kernel_challenges"]
    assert modern
    for detail in modern:
        profile = comparison_profile(detail)
        public = ROOT / "benchmark_fn_fp" / detail["public_dir"] / "kernel.py"
        assert profile["kind"] in {"allclose", "bits", "exact", "rmsnorm"}
        assert generator_defaults(public.read_text())["seed"] == 0
    with pytest.raises(ValueError, match="No reviewed public comparison"):
        comparison_profile({"cohort": "new_unreviewed_operation"})


def test_integer_comparison_does_not_round_through_float64():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([2**60], dtype=torch.int64)
    actual = torch.tensor([2**60 + 1], dtype=torch.int64)
    check = compare_outputs(actual, reference, {"kind": "exact"}, ())[0]
    assert check["passed"] is False
    assert check["mismatched_elements"] == 1


def test_exact_bits_preserve_signed_zero_requirement():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([0.0], dtype=torch.float32)
    actual = torch.tensor([-0.0], dtype=torch.float32)
    assert compare_outputs(actual, reference, {"kind": "bits"}, ())[0]["passed"] is False
    assert compare_outputs(actual, reference, {"kind": "allclose", "atol": 1e-6, "rtol": 0}, ())[0]["passed"] is True


def test_nonfinite_values_fail_even_if_both_outputs_are_nan():
    torch = pytest.importorskip("torch")
    value = torch.tensor([float("nan")])
    row = compare_outputs(value, value, {"kind": "allclose", "atol": 1e-6, "rtol": 0}, ())[0]
    assert row["passed"] is False
    assert row["nonfinite_actual"] == 1
    json.dumps(row, allow_nan=False)


def test_rmsnorm_dw_uses_sum_of_absolute_terms_not_reference_magnitude():
    torch = pytest.importorskip("torch")
    x = torch.ones((2, 1), dtype=torch.float16)
    dy = torch.tensor([[1.], [-1.]], dtype=torch.float16)
    inputs = (x, torch.ones(1, dtype=torch.float16), dy, torch.ones(2))
    reference = (torch.zeros((2, 1)), torch.zeros(1))
    actual = (reference[0], torch.tensor([2e-5]))
    rows = compare_outputs(actual, reference, {"kind": "rmsnorm"}, inputs)
    assert all(row["passed"] for row in rows)  # 2e-5 <= 1e-5 + 1e-5 * 2


@pytest.fixture
def cpu_execution(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test stub")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (0, 0))


def test_reference_is_frozen_before_mutating_kernel(cpu_execution):
    source = '''import torch

def make_inputs(device='cuda', seed=0):
    return (torch.tensor([1.0]),)

def reference(x):
    return x

def run(x):
    x.add_(1)
    return x
'''
    payload = {"case": "case_test", "mode": "public_default", "sources": {"kernel.py": source},
               "comparison": {"kind": "allclose", "atol": 1e-6, "rtol": 0}}
    result = execute_payload(payload)
    assert result["error"] is None
    assert result["verdict"] == "reject"
    assert result["inputs"][0]["shape"] == [1]


def test_legacy_error_after_marker_is_not_silently_accepted(cpu_execution):
    source = '''def test_kernel():
    print('NAIVE_ALLCLOSE_VERDICT: True')
    raise RuntimeError('test failed after verdict')
'''
    result = execute_payload({"case": "case_test", "mode": "legacy_naive", "sources": {"test.py": source}})
    assert result["verdict"] == "no_verdict"
    assert result["passed"] is None
    assert "test failed after verdict" in result["error"]
    assert "NAIVE_ALLCLOSE_VERDICT: True" in result["stdout"]
