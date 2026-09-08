"""Batch 6: five cases filling out FN1, FN2, FN5, FP1 and FP4.

Every case here is verified on a real GPU before it is used, because the
measured failure rate of unverified designs in this benchmark has been high: of
the twelve cases drafted in batches 4 and 5, six failed their first GPU check
for a reason in the case, not in the kernel under test -- a bug that was never
injected, an adversarial input that rounded away in float32, a mechanism
(quantization) that is idempotent and so cannot drift, a claimed downstream
equivalence that does not hold, a parameter below the threshold that triggers
it, and a wrong orthogonalisation formula.

Usage (from repo root):
    python benchmark_fn_fp/generation/generators/batch6_remaining_seeds.py
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "benchmark_fn_fp" / "triton"

# ==========================================================================
# fn16 -- FN2: fully masked row takes a branch ordinary tests never enter
# ==========================================================================
MASKED_KERNEL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _masked_softmax_kernel(X, M, Y, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(X + row * stride + cols, mask=valid, other=-float("inf"))
    keep = tl.load(M + row * stride + cols, mask=valid, other=0).to(tl.int1)
    x = tl.where(keep, x, -float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.where(keep, tl.exp(x), 0.0)
    tl.store(Y + row * stride + cols, e / tl.sum(e, axis=0), mask=valid)


def masked_softmax(x, mask):
    # Row-wise softmax over the positions the mask keeps.
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    _masked_softmax_kernel[(n_rows,)](x, mask, y, x.stride(0), n_cols,
                                      BLOCK=triton.next_power_of_2(n_cols))
    return y
"""

MASKED_PROBLEM = """Operation: row-wise softmax restricted to the positions a boolean mask keeps,
the attention-probability step of any masked or padded attention kernel.

    y[j] = exp(x[j]) / sum over kept k of exp(x[k])     if mask[j]
    y[j] = 0                                            otherwise

Input domain: the mask is arbitrary. In particular a row may keep NO positions
at all -- a fully padded query row, or a causal row before its first visible
key. For such a row the specification requires an all-zero output row: there is
no probability mass to distribute, and every downstream consumer treats the row
as contributing nothing.

Every returned value must be finite.

Does masked_softmax() satisfy this contract?
"""

MASKED_TEST = """import torch
from kernel import masked_softmax


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_rows, n_cols = 8, 64

    x = torch.randn(n_rows, n_cols, device=device)
    # Conventional test: a random mask, every row keeps something.
    mask = (torch.rand(n_rows, n_cols, device=device) > 0.3).to(torch.int32)
    mask[:, 0] = 1
    cand = masked_softmax(x, mask)
    ref = torch.softmax(x.masked_fill(mask == 0, float("-inf")), dim=1)
    common_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-2)

    # Contract-permitted rare input: one row keeps nothing.
    mask2 = mask.clone()
    mask2[3, :] = 0
    cand2 = masked_softmax(x, mask2)
    row = cand2[3]
    all_finite = bool(torch.isfinite(cand2).all().item())
    zero_row = bool((row.abs() < 1e-6).all().item())

    # Downstream: the row is used to average a value matrix.
    v = torch.randn(n_cols, 16, device=device)
    out = cand2 @ v
    downstream_finite = bool(torch.isfinite(out).all().item())

    print(f"random mask, every row keeps something: allclose={common_pass}")
    print(f"fully masked row: all finite={all_finite}, all-zero as specified={zero_row}, "
          f"row[0]={row[0].item()}")
    print(f"downstream value average finite={downstream_finite}")
    # What a conventional CI test would conclude: its random mask never empties a row.
    print(f"NAIVE_ALLCLOSE_VERDICT: {common_pass}")
    return common_pass and not (all_finite and zero_row)


if __name__ == "__main__":
    test_kernel()
"""

# ==========================================================================
# fn17 -- FN5: percentile calibration collapses on outlier channels
# ==========================================================================
PCTL_KERNEL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _quant_kernel(X, OUT, stride, N, PCTL_LEVELS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(X + row * stride + cols, mask=valid, other=0.0)

    ax = tl.abs(x)
    absmax = tl.max(ax, axis=0)
    # Calibrate on the bulk of the row rather than on its single largest value.
    thresh = absmax
    for _ in range(PCTL_LEVELS):
        below = tl.where(ax < thresh, ax, 0.0)
        thresh = tl.max(below, axis=0)

    scale = tl.where(thresh == 0.0, 1.0, thresh / 127.0)
    q = tl.floor(x / scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(OUT + row * stride + cols, q * scale, mask=valid)


def quant_dequant(x, pctl_levels: int = 2):
    # Symmetric INT8 quantize-dequantize with a per-row calibrated scale.
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    _quant_kernel[(n_rows,)](x, out, x.stride(0), n_cols,
                             PCTL_LEVELS=pctl_levels,
                             BLOCK=triton.next_power_of_2(n_cols))
    return out
"""

PCTL_PROBLEM = """Operation: symmetric INT8 quantize-dequantize with a per-row calibrated scale,
the activation quantization step of INT8 inference.

    scale = calibration(|x|) / 127
    q     = round(x / scale), clamped to [-127, 127]
    y     = q * scale

Accuracy requirement: the relative reconstruction error of each row,

    ||y - x|| / ||x||        (Euclidean norm over the row)

must not exceed 5%. That is the usual way INT8 quantization quality is stated,
and it is loose enough to absorb ordinary rounding: an 8-bit grid over the row's
dynamic range leaves a residual well under 1%. It is not loose enough to absorb
a value that has been clamped away, since a clamped entry removes its whole
magnitude from the reconstruction.

Input domain: real transformer activations, whose per-channel magnitudes are
heavy-tailed. A small number of channels carrying values one or two orders of
magnitude above the bulk is normal and documented (LLM.int8(), SmoothQuant), and
those channels are exactly the ones downstream layers depend on.

Does quant_dequant() satisfy this contract?
"""

PCTL_TEST = """import torch
from kernel import quant_dequant


def rel_error(y, x):
    return ((y - x).norm(dim=1) / x.norm(dim=1)).max().item()


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_cols = 512

    # Conventional test data: IID Gaussian, no outlier channels.
    x_iid = torch.randn(8, n_cols, device=device)
    y_iid = quant_dequant(x_iid)
    iid_rel = rel_error(y_iid, x_iid)
    iid_pass = iid_rel <= 0.05

    # Realistic activations: a few channels an order of magnitude larger, the
    # outlier structure documented by LLM.int8() and SmoothQuant.
    x_out = torch.randn(8, n_cols, device=device)
    x_out[:, 7] *= 40.0
    x_out[:, 200] *= 25.0
    y_out = quant_dequant(x_out)
    out_rel = rel_error(y_out, x_out)
    outlier_pass = out_rel <= 0.05
    worst = (y_out - x_out).abs().max().item()

    print(f"IID Gaussian rows: relative reconstruction error = {iid_rel:.4f}, within 5% = {iid_pass}")
    print(f"rows with outlier channels: relative reconstruction error = {out_rel:.4f}, "
          f"within 5% = {outlier_pass}, worst single-entry error = {worst:.2f}")
    # What a conventional CI test would conclude: it calibrates on randn, whose
    # bulk and maximum are close together.
    print(f"NAIVE_ALLCLOSE_VERDICT: {iid_pass}")
    return iid_pass and not outlier_pass


if __name__ == "__main__":
    test_kernel()
"""

# ==========================================================================
# fn18 -- FN1: segmented top-1 routing, tie resolved against the stated rule
# ==========================================================================
ROUTE_KERNEL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(Logits, Idx, stride, E: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < E
    x = tl.load(Logits + tok * stride + cols, mask=valid, other=-float("inf"))

    best = -float("inf")
    best_i = 0
    for e in range(E):
        v = tl.sum(tl.where(cols == e, x, 0.0), axis=0)
        take = v >= best
        best = tl.where(take, v, best)
        best_i = tl.where(take, e, best_i)
    tl.store(Idx + tok, best_i)


def route_top1(logits):
    # Pick one expert per token from the router logits.
    n_tokens, n_experts = logits.shape
    idx = torch.empty(n_tokens, device=logits.device, dtype=torch.int32)
    _route_kernel[(n_tokens,)](logits, idx, logits.stride(0), E=n_experts,
                               BLOCK=triton.next_power_of_2(n_experts))
    return idx
"""

ROUTE_PROBLEM = """Operation: top-1 expert routing for a Mixture-of-Experts layer. For each token,
select the expert with the largest router logit.

    expert[t] = argmax over e of logits[t, e]

Tie rule: router logits are computed in reduced precision and distinct logits
routinely round to the same representable value, so ties are a normal
occurrence rather than a curiosity. When several experts share the largest
logit, the contract requires the LOWEST expert index to win, so that routing is
reproducible across runs and across replicas holding the experts in the same
order.

The selected expert's output is what the layer returns, and different experts
are independently trained, so which one wins a tie changes the continuous
output substantially.

Does route_top1() satisfy this contract?
"""

ROUTE_TEST = """import torch
from kernel import route_top1


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_tokens, n_experts, dim = 64, 8, 32

    # Conventional test: float32 random logits, exact ties essentially never occur.
    logits = torch.randn(n_tokens, n_experts, device=device)
    cand = route_top1(logits)
    ref = logits.argmax(dim=1).to(torch.int32)
    random_pass = torch.equal(cand, ref)

    # Reduced-precision logits: distinct values collapse onto the same bf16 level.
    tied = torch.randn(n_tokens, n_experts, device=device).to(torch.bfloat16).float()
    tied[:, 2] = tied[:, 5] = tied.max(dim=1).values + 1.0
    cand_t = route_top1(tied)
    lowest_wins = bool((cand_t == 2).all().item())

    # Downstream: independently trained experts, so the choice moves the output.
    experts = torch.randn(n_experts, dim, device=device)
    got = experts[cand_t.to(torch.int64)]
    want = experts[torch.full_like(cand_t, 2, dtype=torch.int64)]
    downstream_pass = torch.allclose(got, want, rtol=1e-2, atol=1e-2)

    print(f"float32 random logits (no ties): matches torch.argmax = {random_pass}")
    print(f"tie between expert 2 and expert 5: kernel picked {cand_t[0].item()}, "
          f"lowest-index-wins = {lowest_wins}")
    print(f"downstream expert output allclose = {downstream_pass}")
    # What a conventional CI test would conclude: float32 logits do not tie.
    print(f"NAIVE_ALLCLOSE_VERDICT: {random_pass}")
    return random_pass and not lowest_wins and not downstream_pass


if __name__ == "__main__":
    test_kernel()
"""

# ==========================================================================
# fp10 -- FP1: top-k boundary tie the specification leaves open
# ==========================================================================
TOPK_KERNEL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _threshold_kernel(Scores, Out, stride, K, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(Scores + row * stride + cols, mask=valid, other=-float("inf"))

    lo = tl.min(tl.where(valid, x, float("inf")), axis=0)
    hi = tl.max(x, axis=0)
    for _ in range(40):
        mid = (lo + hi) * 0.5
        count = tl.sum(tl.where(valid & (x >= mid), 1, 0), axis=0)
        lo = tl.where(count >= K, mid, lo)
        hi = tl.where(count >= K, hi, mid)
    # Ties at the cutoff are all admitted.
    keep = valid & (x >= lo)
    tl.store(Out + row * stride + cols, tl.where(keep, x, 0.0), mask=valid)


def topk_mask(scores, k: int):
    # Zero out everything below the top-k cutoff, keeping ties at the cutoff.
    n_rows, n_cols = scores.shape
    out = torch.empty_like(scores)
    _threshold_kernel[(n_rows,)](scores, out, scores.stride(0), k, N=n_cols,
                                 BLOCK=triton.next_power_of_2(n_cols))
    return out
"""

TOPK_PROBLEM = """Operation: keep the top-k scores of every row and zero the rest, the candidate
selection step of top-k sampling.

    out[j] = scores[j]  if scores[j] is among the k largest of the row
    out[j] = 0          otherwise

Specification of ties: when several entries share the value at the k-th
position, the specification does not say which of them to keep. Admitting all
tied entries and admitting only enough to reach exactly k are both accepted
readings; the requirement is that every kept entry is at least as large as every
dropped one.

Downstream the kept entries are renormalized and used as sampling weights, so
the two readings produce different continuous outputs.

Does topk_mask() satisfy this contract?
"""

TOPK_TEST = """import torch
from kernel import topk_mask


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_cols, k = 64, 8

    # A row where the value at the cutoff is shared by several entries.
    scores = torch.randn(4, n_cols, device=device)
    scores[:, :12] = 2.0                      # 12 entries tied above the cutoff
    cand = topk_mask(scores, k)

    # Reference reading: admit exactly k, breaking the tie by lowest index.
    ref = torch.zeros_like(scores)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    ref.scatter_(1, order, scores.gather(1, order))

    raw_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-2)

    # The contract's actual requirement: no dropped entry exceeds a kept one.
    kept = cand != 0
    min_kept = torch.where(kept, cand, torch.full_like(cand, float("inf"))).min(dim=1).values
    max_dropped = torch.where(kept, torch.full_like(cand, -float("inf")), scores).max(dim=1).values
    ordering_ok = bool((min_kept >= max_dropped - 1e-6).all().item())

    n_kept = int(kept[0].sum().item())
    print(f"reference admits exactly k={k}; kernel kept {n_kept} entries")
    print(f"elementwise allclose against that one reference = {raw_pass} (expected False)")
    print(f"contract requirement (no dropped entry exceeds a kept one) = {ordering_ok} (expected True)")
    # What a conventional CI test would conclude: it compares elementwise against
    # one implementation's private tie convention.
    print(f"NAIVE_ALLCLOSE_VERDICT: {raw_pass}")
    return (not raw_pass) and ordering_ok


if __name__ == "__main__":
    test_kernel()
"""

# ==========================================================================
# fp11 -- FP4: FP8 (e4m3) round trip judged by an FP32-calibrated tolerance
# ==========================================================================
FP8_KERNEL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_roundtrip_kernel(X, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)

    # e4m3: 3 mantissa bits, so quantize to 1/8 of a binade.
    ax = tl.abs(x)
    e = tl.floor(tl.log2(tl.where(ax == 0.0, 1.0, ax)))
    step = tl.exp2(e - 3.0)
    q = tl.floor(x / step + 0.5) * step
    q = tl.where(ax == 0.0, 0.0, q)
    tl.store(OUT + offs, q, mask=mask)


def fp8_roundtrip(x):
    # Round a float32 tensor to the nearest value representable in e4m3.
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    _fp8_roundtrip_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out
"""

FP8_PROBLEM = """Operation: round a float32 tensor to the nearest value representable in the FP8
e4m3 format, the storage step of FP8 inference and training kernels.

e4m3 has 4 exponent bits and 3 mantissa bits, so within any binade it can
represent 8 evenly spaced values. Rounding to that grid is the point of the
format, not an implementation shortcut: the resulting relative error is bounded
by half a mantissa step, about 2^-4 = 6.25%.

The contract is that every value is rounded to the nearest representable e4m3
value. Deviation from the float32 input up to the format's own rounding step is
the intended behavior of the operation.

Does fp8_roundtrip() satisfy this contract?
"""

FP8_TEST = """import torch
from kernel import fp8_roundtrip


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(4096, device=device)
    y = fp8_roundtrip(x)

    # A tolerance calibrated for float32 kernels.
    fp32_tolerance_pass = torch.allclose(y, x, rtol=1e-2, atol=1e-2)

    # The error model of the declared format: half a mantissa step per binade.
    step = torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(1e-30))) - 3.0)
    within_format = bool(((y - x).abs() <= 0.5 * step + 1e-30).all().item())
    max_rel = ((y - x).abs() / x.abs().clamp_min(1e-30)).max().item()

    print(f"FP32-calibrated allclose(rtol=1e-2, atol=1e-2) = {fp32_tolerance_pass} (expected False)")
    print(f"within half an e4m3 mantissa step = {within_format} (expected True), "
          f"max relative error = {max_rel:.4f}")
    # What a conventional CI test would conclude: it applies the tolerance it uses
    # for float32 kernels.
    print(f"NAIVE_ALLCLOSE_VERDICT: {fp32_tolerance_pass}")
    return (not fp32_tolerance_pass) and within_format


if __name__ == "__main__":
    test_kernel()
"""

CASES = {
    "fn16_masked_softmax_fully_masked_row": dict(
        kernel=MASKED_KERNEL, problem=MASKED_PROBLEM, test=MASKED_TEST,
        group="FN", seed_class="FN2", failure_mode="false_negative",
        kernel_family="masked softmax / attention probabilities",
        reference="masked-softmax step of Dao-AILab/flash-attention and of every padded-attention kernel",
        mechanism="A fully masked row leaves every exponential at zero, so the kernel divides by a zero denominator and returns NaN where the contract requires an all-zero row. Random masks essentially never empty a row, so a conventional test never enters that branch, and the NaN then propagates silently through the value average.",
        naive="PASS"),
    "fn17_int8_percentile_calibration_outliers": dict(
        kernel=PCTL_KERNEL, problem=PCTL_PROBLEM, test=PCTL_TEST,
        group="FN", seed_class="FN5", failure_mode="false_negative",
        kernel_family="per-row calibrated symmetric INT8 quantization",
        reference="activation quantization as discussed in LLM.int8() (arXiv:2208.07339) and SmoothQuant (arXiv:2211.10438); per-row scales in sglang int8_kernel.py",
        mechanism="The scale is calibrated by stepping down from the maximum twice rather than on the true absmax, so the largest entries fall outside the representable range and are clamped. On IID Gaussian rows the clamped tail carries almost none of the row's energy and the reconstruction stays within 5%; on activations with the outlier channels real transformers have, the clamped outlier carries most of the row's energy and the reconstruction error is larger than the signal.",
        naive="PASS"),
    "fn18_moe_top1_routing_tie": dict(
        kernel=ROUTE_KERNEL, problem=ROUTE_PROBLEM, test=ROUTE_TEST,
        group="FN", seed_class="FN1", failure_mode="false_negative",
        kernel_family="MoE top-1 router",
        reference="top-1/top-2 expert routing in vllm-project/vllm fused_moe and in Mixtral-style MoE layers",
        mechanism="The running-best update admits a later expert on an exact tie, so the highest tied index wins where the contract requires the lowest. Float32 router logits essentially never tie; the reduced-precision logits real routers compute do.",
        naive="PASS"),
    "fp10_topk_boundary_tie_admits_all": dict(
        kernel=TOPK_KERNEL, problem=TOPK_PROBLEM, test=TOPK_TEST,
        group="FP", seed_class="FP1", failure_mode="false_positive",
        kernel_family="top-k threshold selection",
        reference="vllm-project/vllm topk_topp_triton.py threshold search",
        mechanism="When several entries share the value at the k-th position the kernel admits all of them, while the reference admits exactly k and breaks the tie by index. The specification fixes neither choice; both keep every admitted entry at least as large as every dropped one.",
        naive="FAIL"),
    "fp11_fp8_e4m3_rounding_within_format": dict(
        kernel=FP8_KERNEL, problem=FP8_PROBLEM, test=FP8_TEST,
        group="FP", seed_class="FP4", failure_mode="false_positive",
        kernel_family="FP8 e4m3 rounding",
        reference="FP8 storage in TransformerEngine and in vLLM FP8 kernels; e4m3 as specified in the OCP FP8 formats",
        mechanism="Rounding to the e4m3 grid produces relative errors up to about 6%, which is the declared behaviour of a 3-mantissa-bit format. A tolerance calibrated for float32 kernels rejects it; measured against the format's own error model every value is within half a mantissa step.",
        naive="FAIL"),
}


def main() -> None:
    for name, spec in CASES.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.py").write_text(f'"""Triton kernel under test: {name}."""' + spec["kernel"])
        (d / "problem.txt").write_text(spec["problem"])
        (d / "test.py").write_text(spec["test"])
        (d / "meta.json").write_text(json.dumps({
            "name": name,
            "benchmark_version": "fn_fp_triton_v1",
            "group": spec["group"],
            "seed_class": spec["seed_class"],
            "failure_mode": spec["failure_mode"],
            "kernel_family": spec["kernel_family"],
            "reference": spec["reference"],
            "mechanism": spec["mechanism"],
            "default_tolerance": {"rtol": 0.01, "atol": 0.01},
            "expected": {
                "ground_truth": spec["mechanism"].split(".")[0],
                "naive_allclose_verdict": spec["naive"],
                "correct_verdict": "BUGGY" if spec["group"] == "FN" else "CORRECT",
            },
            "status": "seed_v2",
            "source": "real_triton_kernel_adapted",
            "requires_gpu": True,
            "passed": None,
        }, indent=2) + "\n")
        print("wrote", name)


if __name__ == "__main__":
    main()
