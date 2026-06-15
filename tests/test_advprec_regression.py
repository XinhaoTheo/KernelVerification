"""Assert-based regression guard for the precision-judgment line.

Unlike the advprec_*_demo.py scripts (which print numbers for a human to read),
this FAILS loudly if any piece regresses. It locks in:
  - classify routes each operator to the right class/judge,
  - the red-team kernels really are benign-pass / adversarial-fail,
  - compare.judge dispatches correctly (J2 catches topk, low_bit abstains),
  - verdict.combine applies the priority rule.

CPU-only and deterministic (red-team kernels are pure torch; classify forces CPU).
Run:  CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_advprec_regression.py
  or: CUDA_VISIBLE_DEVICES="" uv run python tests/test_advprec_regression.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from verifier import compare
from verifier.classify import classify_entry
from verifier.verdict import combine

DS = Path(__file__).resolve().parent.parent / "dataset"
N = 256


def _kernel(entry):
    path = DS / entry / "kernel.py"
    spec = importlib.util.spec_from_file_location(f"_t_{entry}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel_function


# --- classify routes every operator to the right class/judge --------------- #

def test_classify_routes():
    cases = {
        "_advprec_topk_boundary": ("select", "normal"),
        "_advprec_softmax_tail": ("compress", "normal"),
        "_advprec_softmax_fp8_tail": ("compress", "low_bit"),
        "kb_level1_1_square_matrix_multiplication": ("preserve", "normal"),
        "kb_level1_19_relu": ("compress", "normal"),
    }
    for entry, (op, prec) in cases.items():
        r = classify_entry(entry)
        assert r["op_class"] == op, f"{entry}: op_class {r['op_class']} != {op}"
        assert r["precision"] == prec, f"{entry}: precision {r['precision']} != {prec}"


# --- the red-team kernels: benign passes, adversarial is caught ------------ #

def test_topk_redteam_distribution_dependent():
    kf = _kernel("_advprec_topk_boundary")

    benign = torch.randn(1, N, generator=torch.Generator().manual_seed(0))
    bj = compare.judge(kf(benign, 8), torch.topk(benign, 8, dim=-1).indices,
                       op_class="select", scores=benign)
    assert bj["verdict"] == "match", f"benign should pass, got {bj}"

    adv = torch.randn(1, N, generator=torch.Generator().manual_seed(0))
    adv[0, 0:8] = torch.linspace(6.0, 5.3, 8)  # winners concentrated in one block
    aj = compare.judge(kf(adv, 8), torch.topk(adv, 8, dim=-1).indices,
                       op_class="select", scores=adv)
    assert aj["verdict"] == "mismatch", f"adversarial should be caught, got {aj}"
    assert aj["judge"] == "J2"
    assert aj["value_gap"] > 1.0, "dropped a high-value key -> large value-gap"


def test_softmax_redteam_distribution_dependent():
    kf = _kernel("_advprec_softmax_tail")

    benign = torch.randn(1, N, generator=torch.Generator().manual_seed(0))
    bj = compare.judge(kf(benign), torch.softmax(benign, dim=-1), op_class="compress")
    assert bj["verdict"] == "match", f"benign should pass, got {bj}"

    adv = torch.full((1, N), -30.0)
    adv[0, 0] = 8.5
    adv[0, 1:200] = 0.0  # heavy tail just past the truncation cutoff
    aj = compare.judge(kf(adv), torch.softmax(adv, dim=-1), op_class="compress")
    assert aj["verdict"] == "mismatch", f"adversarial should be caught, got {aj}"
    assert aj["max_diff"] > 0.02


def test_fp8_redteam_abstains():
    # low-bit output -> the judge must abstain regardless of the tensors.
    out = torch.zeros(4)
    ref = torch.zeros(4)
    r = compare.judge(out, ref, op_class="compress", scores=None)  # op alone -> J1
    assert r["verdict"] == "match"
    r2 = compare.judge(out, ref, op_class="low_bit")
    assert r2["verdict"] == "abstain" and r2["judge"] == "J3/abstain"


# --- J2 separates harmful from harmless misses ----------------------------- #

def test_value_gap_harmless_vs_harmful():
    scores = torch.tensor([[5.0, 5.0, 5.0, 3.0, 2.0]])
    ref = torch.topk(scores, 2, dim=-1).indices
    # harmless tie-break swap: picked two of the three equal 5.0s -> values match
    harmless = torch.tensor([[1, 2]])
    hj = compare.judge(harmless, ref, op_class="select", scores=scores)
    assert hj["verdict"] == "match", f"tie swap is harmless, got {hj}"
    # harmful: dropped a 5.0, kept a 2.0
    harmful = torch.tensor([[0, 4]])
    fj = compare.judge(harmful, ref, op_class="select", scores=scores)
    assert fj["verdict"] == "mismatch" and fj["value_gap"] >= 2.0


# --- verdict.combine priority rule ----------------------------------------- #

def test_combine_thin_floor():
    # design A: judge is the final arbiter; combine defers to its verdict ...
    assert combine(recheck_status="passed", debate_verdict="reject")["final"] == "reject"
    assert combine(recheck_status="passed", debate_verdict="trust")["final"] == "trust"
    assert combine(recheck_status="passed", debate_verdict="needs_downstream")["final"] == "needs_downstream"
    assert combine(recheck_status="passed", debate_verdict="needs_more_evidence")["final"] == "needs_more_evidence"
    # ... except the ONE fact-based floor: standard fail + judge "trust" -> reject
    assert combine(recheck_status="failed", debate_verdict="trust")["final"] == "reject"
    # the floor only touches "trust"; a judge reject/downstream stands as-is
    assert combine(recheck_status="failed", debate_verdict="reject")["final"] == "reject"
    # robustness gaps annotate, never veto
    r = combine(recheck_status="passed", debate_verdict="trust", robustness={"odd_size": "fail"})
    assert r["final"] == "trust" and r["robustness_gaps"] == ["odd_size"]


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
