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

from verifier import compare, precision_recheck
from verifier.classify import classify_entry, classify_problem
from verifier.verdict import combine

DS = Path(__file__).resolve().parent.parent / "dataset"
N = 256


def _kernel(entry):
    path = DS / entry / "kernel.py"
    spec = importlib.util.spec_from_file_location(f"_t_{entry}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel_function


def _problem(forward: str, *, init="", get_init="return []",
             inputs="return [torch.randn(8, 64)]") -> str:
    """Build a minimal problem.txt string (Model + get_inputs + get_init_inputs)
    so we can test classify on operators not in the dataset."""
    init_block = f"    def __init__(self):\n        super().__init__()\n{init}\n" if init else ""
    return (
        "import torch\nimport torch.nn as nn\n"
        "class Model(nn.Module):\n"
        f"{init_block}"
        f"    def forward(self, x):\n        {forward}\n"
        f"def get_inputs():\n    {inputs}\n"
        f"def get_init_inputs():\n    {get_init}\n"
    )


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


# --- classify on operators NOT in the dataset (string problems) ------------ #

def test_classify_more_operators():
    cases = {
        # preserve: magnitude-preserving, numerical error is faithful
        "x * 2.0": "preserve",
        "x + x": "preserve",
        "x.sum(dim=-1)": "preserve",
        "x.mean(dim=-1)": "preserve",
        "torch.matmul(x, x.transpose(-1, -2))": "preserve",
        # compress: squashes / saturates -> blind regions
        "torch.tanh(x)": "compress",
        "torch.sigmoid(x)": "compress",
        "torch.softmax(x, dim=-1)": "compress",
        "torch.relu(x)": "compress",
        # select: discrete index / permutation
        "torch.argmax(x, dim=-1)": "select",
        "torch.sort(x, dim=-1).values": "select",
        "torch.sort(x, dim=-1).indices": "select",
    }
    for forward, expect in cases.items():
        got = classify_problem(_problem(f"return {forward}"))["op_class"]
        assert got == expect, f"{forward!r}: classify said {got}, expected {expect}"


def test_classify_precision_axis():
    # normal dtype output
    p_norm = _problem("return torch.softmax(x, dim=-1)")
    assert classify_problem(p_norm)["precision"] == "normal"
    # fp8 output -> low_bit -> routed to abstain
    p_fp8 = _problem("return torch.softmax(x, dim=-1).to(torch.float8_e4m3fn)")
    r = classify_problem(p_fp8)
    assert r["precision"] == "low_bit" and "abstain" in r["judge"]


# --- compare.judge dispatch by op_class ------------------------------------ #

def test_judge_dispatch():
    a, b = torch.zeros(4), torch.zeros(4)
    # preserve / compress -> J1 (numerical)
    assert compare.judge(a, b, op_class="preserve")["judge"] == "J1"
    assert compare.judge(a, b, op_class="compress")["judge"] == "J1"
    # J1 mismatch when tensors diverge
    assert compare.judge(torch.zeros(4), torch.ones(4), op_class="preserve")["verdict"] == "mismatch"
    # low_bit -> abstain regardless of tensors
    assert compare.judge(a, b, op_class="low_bit")["verdict"] == "abstain"
    # select needs scores; without them it abstains rather than guess
    idx = torch.tensor([[0, 1]])
    assert compare.judge(idx, idx, op_class="select", scores=None)["verdict"] == "abstain"


# --- judge_selection (J2) edge cases --------------------------------------- #

def test_j2_edge_cases():
    scores = torch.tensor([[9.0, 8.0, 7.0, 1.0, 0.0]])
    ref = torch.topk(scores, 3, dim=-1).indices  # {0,1,2}
    # exact match -> match, gap 0
    m, d = compare.judge_selection(ref, ref, scores)
    assert m and d["value_gap"] == 0.0 and d["recall"] == 1.0
    # dropped the largest (idx 0), kept idx 3 -> big harmful gap
    cand = torch.tensor([[1, 2, 3]])
    m, d = compare.judge_selection(cand, ref, scores)
    assert not m and d["value_gap"] >= 8.0 and d["recall"] < 1.0
    # all-tied scores: any size-k subset is equally correct -> match (gap 0)
    tied = torch.zeros(1, 8)
    rt = torch.topk(tied, 3, dim=-1).indices
    ct = torch.tensor([[5, 6, 7]])
    m, d = compare.judge_selection(ct, rt, tied)
    assert m and d["value_gap"] == 0.0


# --- adversarial distribution generators have the intended structure ------- #

def test_adversarial_distributions():
    import torch as t
    sel = precision_recheck._adversarial("select", 2, 256, t.float32, "cpu")
    # concentrated: the largest values sit in the first block (indices 0..7)
    top = t.topk(sel, 8, dim=-1).indices
    assert (top < 8).all(), "select adversarial must concentrate winners in block 0"
    comp = precision_recheck._adversarial("compress", 2, 256, t.float32, "cpu")
    # heavy-tail: one dominant peak at index 0, a band just below it
    assert (comp[:, 0:1] > comp[:, 1:]).all(), "compress adversarial must have a single peak"
    assert comp.shape == (2, 256)


# --- correct kernels of each class must PASS (false-alarm guard) ------------ #

def test_correct_kernels_not_flagged():
    # a CORRECT softmax on the heavy-tail adversarial input -> match (no false alarm)
    x = precision_recheck._adversarial("compress", 2, 256, torch.float32, "cpu")
    ref = torch.softmax(x, dim=-1)
    good = torch.softmax(x, dim=-1)  # identical correct kernel
    assert compare.judge(good, ref, op_class="compress")["verdict"] == "match"
    # a CORRECT top-k on the concentrated adversarial input -> match
    s = precision_recheck._adversarial("select", 2, 256, torch.float32, "cpu")
    rk = torch.topk(s, 8, dim=-1).indices
    ck = torch.topk(s, 8, dim=-1).indices
    assert compare.judge(ck, rk, op_class="select", scores=s)["verdict"] == "match"


# --- false-REJECT guard: a correct selection with a different tie-break ----- #

def test_no_false_reject_on_tiebreak():
    # correct kernel picks a DIFFERENT but equally-valid set of tied winners.
    # index recall would be < 100% (and wrongly reject); J2 value-gap says match.
    scores = torch.tensor([[5.0, 5.0, 5.0, 5.0, 1.0]])  # four-way tie at the top
    ref = torch.tensor([[0, 1]])      # reference tie-break
    cand = torch.tensor([[2, 3]])     # different valid pair of 5.0s
    r = compare.judge(cand, ref, op_class="select", scores=scores)
    assert r["verdict"] == "match" and r["value_gap"] == 0.0, f"must not false-reject, got {r}"
    # contrast: naive index recall here is 0/2 = 0% -> would have rejected
    assert r["recall"] == 0.0


# --- the WHOLE dataset matrix end-to-end through precision_recheck --------- #

def test_all_advprec_entries_matrix():
    """Drive every dataset/_advprec_* entry through precision_recheck and assert its
    meta's expected.precision_verdict. Covers the full op_class x precision x
    {correct, buggy} matrix: catch (mismatch) / control (match) / abstain / skipped.
    CPU-only (all _advprec_ kernels are pure torch)."""
    import json
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU for the whole matrix
    entries = sorted(d.name for d in DS.iterdir()
                     if d.name.startswith("_advprec_") and (d / "meta.json").exists())
    assert len(entries) >= 11, f"expected the full matrix, found {len(entries)}"
    for entry in entries:
        meta = json.loads((DS / entry / "meta.json").read_text())
        if meta.get("demo_only"):
            continue  # e.g. _advprec_magicpig_attn: multi-input attention, demo-judged
        exp = meta.get("expected", {}).get("precision_verdict")
        assert exp, f"{entry}: meta has no expected.precision_verdict"
        got = precision_recheck.precision_recheck(entry).get("verdict")
        assert got == exp, f"{entry}: precision verdict {got!r}, expected {exp!r}"


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
