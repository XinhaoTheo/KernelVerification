# Single-call trace

## System prompt

You are verifying whether a GPU (Triton) kernel implementation is correct.

You will be given:
1. A problem statement describing the operation's contract and its reference behavior.
2. The kernel implementation under test.

Decide one of:
- "reject": the implementation has a real defect / violates the stated contract.
- "trust": the implementation is correct, or is an equally valid alternative implementation.
- "needs_more_evidence": you genuinely cannot tell.

Answer with the JSON object required by the output schema.
Output schema:
{"type": "object", "required": ["verdict", "confidence", "reason"], "properties": {"verdict": {"type": "string", "enum": ["trust", "reject", "needs_more_evidence"], "description": "'reject' if the implementation has a real defect, 'trust' if it is correct or an equally valid alternative, 'needs_more_evidence' if you genuinely cannot tell."}, "confidence": {"type": "number", "description": "0.0 to 1.0."}, "reason": {"type": "string", "description": "One or two sentences."}}, "additionalProperties": false}

## User prompt

## Problem statement (contract + reference behavior)

Compute the fixed-point requantization epilogue of an integer GEMM, returning
dequantized FP32 values. This interface requires gemmlowp-compatible DOUBLE
rounding. Single-rounding requantization is a different operation.

Inputs and complete public domain:
- acc is int32 [M,C], with 1<=M<=4096 and 1<=C<=257. EVERY int32 accumulator
  value is legal, including -2^31 and 2^31-1.
- multipliers is int32 [C], with every value in [2^30,2^31-1]. These are
  positive Q31 fixed-point per-channel multipliers.
- right_shifts is int32 [C], with values in [0,31].
- zero_points is int32 [C], with values in [-128,127].
- output_scales is FP32 [C], with values exactly in
  {2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,1}.
- All tensors are ordinary torch.Tensor objects, dense, strided, contiguous,
  and on the same CUDA device. Contiguous views with nonzero storage offsets
  are legal. Inputs may share storage if each otherwise meets this contract.
  All have requires_grad=False, is_inference=False, is_neg=False, is_conj=False.
  Calls run in ordinary non-inference mode without concurrent external writes.
  This forward operation does not provide autograd.

For each accumulator a=acc[i,c], use m=multipliers[c], r=right_shifts[c],
z=zero_points[c], and s=output_scales[c]. Define the following using exact
mathematical integer arithmetic, without intermediate integer overflow:

  P = a*m
  H = floor((P + 2^30) / 2^31)
  R = nearest integer to H/2^r, with exact half ties AWAY from zero
  Q = min(127, max(-128, R + z))
  target[i,c] = (Q-z)*s

The H stage rounds to nearest with exact half ties toward POSITIVE infinity;
the R stage has its separately stated tie rule. For r=0, R=H. Since m is
positive and at most 2^31-1, the exceptional saturating product of two -2^31
operands cannot occur. Saturation is applied after the two rounding stages
and zero-point addition. The target is the requantized/dequantized result,
not the pre-quantization real number. All target values are exactly
representable in FP32 for this domain.

Return a fresh finite FP32 CUDA tensor [M,C] on acc.device, preserving all
inputs. Require absolute error <=1e-6 for every element, with relative
tolerance zero. Signed zeros are numerically equivalent. This tolerance is
smaller than the smallest nonzero quantization step in the public domain.
The implementation may fuse stages, use fixed-width product limbs, or
rearrange the arithmetic if the target above is preserved for the ENTIRE
legal domain. Correctness is one verdict for the implementation, not one
verdict per input, seed, shape or channel parameter configuration.

validate_inputs() enforces this domain and is called by run(). reference()
uses Python arbitrary-precision integers and the direct two-stage definition;
it accepts CPU inputs. make_inputs() supplies examples only. This is a
synthetic implementation of gemmlowp's fixed-point epilogue arithmetic;
it does not claim to reproduce an upstream historical defect.

Frozen environment: NVIDIA T4, PyTorch2.8.0, Triton3.4.0.


## Kernel implementation under test (kernel.py)

```python
"""Fused fixed-point rescaling and dequantization for an integer GEMM epilogue.

Synthetic Triton implementation of the gemmlowp double-rounding arithmetic.
The arithmetic specification is inspired by gemmlowp fixedpoint/fixedpoint.h;
this is not a port of an upstream kernel or a claim of a historical bug.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rescale(ACC, MULT, SHIFT, ZERO, SCALE, OUT,
             ROWS: tl.constexpr, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < ROWS * CHANNELS
    channel = index % CHANNELS
    a = tl.load(ACC + index, mask=active, other=0)
    m = tl.load(MULT + channel, mask=active, other=1073741824)
    r = tl.load(SHIFT + channel, mask=active, other=0)
    z = tl.load(ZERO + channel, mask=active, other=0)
    scale = tl.load(SCALE + channel, mask=active, other=1.0)

    # Signed upper product and modulo-2^32 lower product.
    high = tl.inline_asm_elementwise(
        "mul.hi.s32 $0, $1, $2;", constraints="=r,r,r",
        args=[a, m], dtype=tl.int32, is_pure=True, pack=1)
    low = a.to(tl.uint32) * m.to(tl.uint32)
    low_a = low + tl.full((BLOCK,), 0x40000000, tl.uint32)
    carry_a = (low_a < low).to(tl.int32)

    negative = a < 0
    low_b = tl.where(
        r == 1, tl.where(negative, 0, 0x80000000),
        tl.where((r > 1) & negative, 0x80000000, 0)).to(tl.uint32)
    high_b = tl.where(
        r >= 2, (1 << tl.maximum(r - 2, 0)) - negative.to(tl.int32), 0)
    low_final = low_a + low_b
    carry_b = (low_final < low_a).to(tl.int32)
    high_final = high + high_b + carry_a + carry_b

    q0 = ((high_final.to(tl.uint32) << 1) | (low_final >> 31)).to(tl.int32)
    shifted = high_final >> tl.maximum(r - 1, 0)
    q = tl.where(r == 0, q0, shifted)
    q = tl.minimum(tl.maximum(q, -255), 255)
    q = tl.minimum(tl.maximum(q + z, -128), 127)
    result = (q - z).to(tl.float32) * scale
    tl.store(OUT + index, result, mask=active)


def validate_inputs(acc, multipliers, right_shifts, zero_points, output_scales,
                    require_cuda=True):
    """Validate the whole public domain; CPU validation is also available."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    buffers = (acc, multipliers, right_shifts, zero_points, output_scales)
    for tensor in buffers:
        require(type(tensor) is torch.Tensor, "ordinary torch.Tensor required")
        require(tensor.layout == torch.strided and tensor.is_contiguous(),
                "contiguous strided tensors required")
        require(not tensor.requires_grad and not tensor.is_inference(),
                "ordinary non-gradient buffers required")
        require(not tensor.is_neg() and not tensor.is_conj(),
                "logical view flags are outside this domain")
        require(not require_cuda or tensor.is_cuda, "CUDA tensors required by run")
    require(all(t.device == acc.device for t in buffers), "one device required")
    require(acc.ndim == 2 and acc.dtype == torch.int32, "acc must be int32 [M,C]")
    rows, channels = acc.shape
    require(1 <= rows <= 4096 and 1 <= channels <= 257,
            "M must be in [1,4096] and C in [1,257]")
    for tensor in (multipliers, right_shifts, zero_points):
        require(tensor.dtype == torch.int32 and tensor.shape == (channels,),
                "integer channel parameters must be int32 [C]")
    require(output_scales.dtype == torch.float32 and output_scales.shape == (channels,),
            "output_scales must be FP32 [C]")
    require(bool((multipliers >= (1 << 30)).all().item()),
            "Q31 multipliers must be in [2^30,2^31-1]")
    require(bool(((right_shifts >= 0) & (right_shifts <= 31)).all().item()),
            "right shifts must be in [0,31]")
    require(bool(((zero_points >= -128) & (zero_points <= 127)).all().item()),
            "zero points must be in [-128,127]")
    allowed_scale = torch.zeros_like(output_scales, dtype=torch.bool)
    for exponent in range(-8, 1):
        allowed_scale |= output_scales == 2.0 ** exponent
    require(bool(allowed_scale.all().item()), "scales must be powers of two from 2^-8 to 1")
    return {"valid": True, "rows": rows, "channels": channels}


def run(acc, multipliers, right_shifts, zero_points, output_scales):
    """Return a fresh FP32 dequantized epilogue result."""
    validate_inputs(acc, multipliers, right_shifts, zero_points, output_scales)
    rows, channels = acc.shape
    with torch.cuda.device(acc.device):
        out = torch.empty((rows, channels), dtype=torch.float32, device=acc.device)
        _rescale[(triton.cdiv(rows * channels, 128),)](
            acc, multipliers, right_shifts, zero_points, output_scales, out,
            ROWS=rows, CHANNELS=channels, BLOCK=128, num_warps=4)
    return out


def reference(acc, multipliers, right_shifts, zero_points, output_scales):
    """Independent scalar Python-integer definition of the two rounding stages."""
    validate_inputs(acc, multipliers, right_shifts, zero_points, output_scales,
                    require_cuda=False)
    matrix = acc.cpu().tolist()
    mult = multipliers.cpu().tolist()
    shifts = right_shifts.cpu().tolist()
    zeros = zero_points.cpu().tolist()
    scales = output_scales.cpu().tolist()
    result = []
    for row in matrix:
        out_row = []
        for a, m, r, z, scale in zip(row, mult, shifts, zeros, scales):
            product = a * m
            high_round = (product + (1 << 30)) // (1 << 31)
            magnitude = abs(high_round)
            if r:
                magnitude = (magnitude + (1 << (r - 1))) // (1 << r)
            scaled = -magnitude if high_round < 0 else magnitude
            quantized = min(127, max(-128, scaled + z))
            out_row.append((quantized - z) * scale)
        result.append(out_row)
    return torch.tensor(result, dtype=torch.float32, device=acc.device)


def make_inputs(device="cuda", rows=31, channels=33, seed=0):
    """Example generator only; the legal accumulator domain is all int32 values."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    acc = torch.randint(-(1 << 31), 1 << 31, (rows, channels),
                        dtype=torch.int64, generator=generator).to(torch.int32)
    multipliers = torch.randint(1 << 30, 1 << 31, (channels,),
                                dtype=torch.int64, generator=generator).to(torch.int32)
    shifts = torch.randint(0, 32, (channels,), dtype=torch.int32, generator=generator)
    zeros = torch.randint(-128, 128, (channels,), dtype=torch.int32, generator=generator)
    exponents = torch.randint(-8, 1, (channels,), generator=generator)
    scales = torch.pow(torch.tensor(2.0), exponents).to(torch.float32)
    return tuple(t.to(device) for t in (acc, multipliers, shifts, zeros, scales))

```

Is this implementation correct?

## Provider reasoning (verbatim)

Need to analyze the kernel's rounding arithmetic. The reference: H = floor((P+2^30)/2^31), rounding to nearest with ties toward +inf (floor of +0.5). Then R = round half away from zero: sign(H)*floor((|H|+2^(r-1))/2^r).

Kernel: computes high = signed high 32 bits of a*m (i.e., floor(P/2^32)? Actually mul.hi.s32 gives floor(P/2^32) for signed). low = unsigned low.

Then adds 0x40000000 to low, carry_a. low_b / high_b: adds rounding constant for the second stage and for the first stage?

Let me think: H = floor((P + 2^30)/2^31). P = high*2^32 + low (as signed with low unsigned). P + 2^30: adding 0x40000000 to low (2^30 = 0x40000000), carry. H = floor((P+2^30)/2^31) = (high*2^32 + low_a_total)/2^31 floor = 2*high' + (low>>31) where high' = high + carry? Actually floor((h*2^32 + l)/2^31) = 2h + (l>>31). So H = 2*(high+carry_a) + (low_a>>31). Wait but floor of negative P: mul.hi.s32 for negative gives floor(P/2^32)? Yes, signed mulhi gives floor for negatives. So P = high*2^32 + low where 0<=low<2^32. Then H = floor((P+2^30)/2^31).

Now the kernel also adds low_b and high_b. Those look like the second-stage rounding constant: R = round-half-away-from-zero(H/2^r) = sign*floor((|H|+2^(r-1))/2^r). Hmm, but ties AWAY from zero for R, while H stage ties toward +inf. The kernel seems to fuse: compute (P + 2^30 + something)/2^31 then >> (r-1)? Let's check r=0 case: q0 = (high_final <<1)| (low_final>>31), which equals H computed via floor((P+2^30)/2^31) with high_final = high+carry_a (low_b, high_b, carry_b are zero when r=0? low_b: r==1 false, (r>1)&neg false → 0. high_b: r>=2 false →0. carry_b: low_final=low_a so carry_b=0). So q0 = 2*(high+carry_a) + (low_a>>31) = H. Correct for r=0.

For r>=1: R = round-half-away(H/2^r). Note H here after first rounding. The kernel's approach: shifted = high_final >> max(r-1,0). Let's check r=1: low_b = 0x80000000 if a>=0 else 0? tl.where(negative, 0, 0x80000000) — for r==1, low_b = 0x80000000 if not negative, 0 if negative. high_b = 0 (r>=2 false). So total added: 2^30 (first stage) + 2^31 (second stage half constant for nonneg). Hmm, for a<0, added 0 for second stage. R = sign(H)*floor((|H|+... Actually round half away from zero: for H>=0, R=floor((H+2^(r-1))/2^r)=floor((H+0.5)/2^r) since r=1: floor(H/2 + 0.5). For H<0, R = -floor((-H+2^(r-1))/2^r) = -floor((|H|+1)/2) i.e. round half away: -(|H|+1)//2 for odd |H|... For H<0, round half away from zero: R = -floor((|H|+2^{r-1})/2^r). With r=1: -( |H|+1 )//2.

Kernel fused: q = floor((P + 2^30 + 2^31 * (a>=0 ? 1 : 0)) / 2^{31+r-1})? Actually shifted = high_final >> (r-1), and high_final represents floor((P + added)/2^32)? Hmm: high_final = high + high_b + carry_a + carry_b = floor((P + 2^30 + low_b + high_b*2^32)/2^32). Then shifted = high_final >> (r-1) = floor(high_final / 2^{r-1}).

But the first rounding H is floor((P+2^30)/2^31), and then R = sign*floor((|H|+2^{r-1})/2^r). Fusing these: H = floor(X/2^31) where X = P+2^30. Then H/2^r rounded half away. The kernel computes floor((X + add2)/2^{31+r-1})? Not exactly — it computes floor(floor((X+add2)/2^32)/2^{r-1}) = floor((X+add2)/2^{31+r}). Wait shifted uses r-1 shift, and high_final is /2^32, so total /2^{31+r}. Hmm.

Let me define: goal R. Cases by sign of a (which equals sign of P since m>0, except a=0).

Case P>=0 (H>=... note H>=0 when P>=-2^30; if P in [-2^30+1? Actually H = floor((P+2^30)/2^31); H>=0 iff P >= -2^30... For P>=0, H>=... fine). For H>=0: R = floor((H + 2^{r-1})/2^r) = floor((floor((P+2^30)/2^31) + 2^{r-1})/2^r). Fused: floor((P + 2^30 + 2^{30+r}) / 2^{31+r}). Is that equal? floor((floor((X)/2^31) + c)/2^r) vs floor((X + c*2^31)/2^{31+r}). These are equal when... Actually floor(floor(X/2^31)/2^r + c/2^r) vs floor((X + c*2^31)/2^{31+r}) = floor(X/2^{31+r} + c/2^r). Since floor(X/2^{31+r}) vs floor(floor(X/2^31)/2^r) — these are equal (nested floor division). But the additive c/2^r with c=2^{r-1} gives 0.5 which... hmm, floor(A + 0.5) vs floor(floor(X/2^31)/2^r + 0.5): with A = X/2^{31+r}. Not generally equal, but for integers... floor(X/2^{31+r} + 0.5) rounds half up. But the target is round-half-away for H>=0: floor((H + 2^{r-1})/2^r) = round half up on H/2^r. The fused version rounds on the pre-first-rounding value X, which can differ when X mod 2^31 straddles... hmm. Actually the kernel's structure: high_final includes both the 2^30 first-stage constant and the second-stage constant, and shifted = high_final >> (r-1). Total divisor 2^{31+r}... but the first stage divides by 2^31, second by 2^r; total 2^{31+r}. But fusing both roundings into one division is generally incorrect because double rounding differs — but here it's a specific fusion: floor(floor((X)/2^31)/2^r + 1/2)?? Let me compute more carefully.

Let X = P + 2^30 + (second stage constant). Let me define kernel computation: it adds to P: 2^30 (low_a), plus low_b + high_b*2^32. Then high_final = floor((P + all)/2^32). shifted = floor(high_final / 2^{r-1}).

For r>=2, nonnegative a: low_b=0, high_b = 2^{r-2} - 0 = 2^{r-2}. So added = 2^30 + 2^{r-2}*2^32 = 2^30 + 2^{r+30}. Then high_final = floor((P + 2^30 + 2^{r+30})/2^32). shifted = floor(that / 2^{r-1}) = floor((P + 2^30 + 2^{r+30}) / 2^{31+r}).

Compare target for H>=0: R = floor((H + 2^{r-1})/2^r), H = floor((P+2^30)/2^31).

Kernel gives Rk = floor((P + 2^30 + 2^{r+30})/2^{31+r}) = floor( (P+2^30)/2^{31+r} + 1/2 ). Hmm, that's rounding (P+2^30)/2^{31+r} to nearest half-up. Target: R = round_half_up( H/2^r ) where H = round_half_up-ish of P/2^31.

Is double rounding here equal to single? Round H (from X/2^31, ties up) then round H/2^r ties away(=up for positive). Compose: R = floor((floor((X)/2^31) + 2^{r-1})/2^r). Write X = q*2^{31+r} + rem, 0<=rem<2^{31+r}. Hmm, standard result: two-stage round-half-up is NOT equivalent to single round-half-up in general? Actually floor(floor(X/2^31 + 1/2)/2^r + 1/2)... Let's just test with numbers. Let r=1, X = 3*2^31 + 2^31 - 1? Hmm.

Simpler test: r=1. H = floor(X/2^31) (with X=P+2^30, first rounding to nearest up). Target R = floor((H+1)/2). Kernel r=1: low_b = 2^31 (if a>=0), high_b=0. So added total 2^30+2^31. Rk = floor((P+2^30+2^31)/2^32) = floor(X/2^32 + 1/2) = round half up of X/2^32.

Target: H = floor(X/2^31), R = floor((H+1)/2) = round half up of H/2.

Take X such that X/2^32 is exactly half-integer? If X = 2^31*(2k+1) exactly, then X/2^32 = k + 0.5 → Rk = k+1. H = 2k+1, R = (2k+1+1)/2 = k+1. Equal. If X = 2^31*(2k+1) - 1 (just below half), Rk = k. H = 2k (since floor), R = (2k+1)//2 = k. Equal. If X = 2^31*(2k) + something... H=2k, R=k. Rk = floor((2k*2^31 + t)/2^32 + 0.5) where t in [0,2^31): = 2k + floor((t + 2^31)/2^32) = 2k + (1 if t=2^31... t<2^31 so t+2^31 < 2^32, floor 0 unless t = 2^31 exactly? t ranges [0,2^31), so t+2^31 in [2^31, 2^32), floor((...)/2^32) = 0 except t+2^31=2^32 impossible. Wait t can be up to 2^31-1, so t+2^31 <= 2^32-1, floor = 0. Hmm so Rk = 2k?? That's wrong—R should be k, Rk = 2k?

Wait I messed up. Rk = floor(X/2^32 + 1/2) with X/2^32 ≈ k + small. Let me redo: X = 2^31*(2k) + t, t∈[0,2^31). X/2^32 = k + t/2^32. Plus 0.5: k + 0.5 + t/2^32. Floor = k (since t/2^32 < 0.5). R = floor((H+1)/2) with H=2k: (2k+1)//2 = k. Equal. OK I confused myself earlier; fine.

General claim: floor(floor(X/2^31 + 1/2)/2^r + 1/2) vs kernel floor(X/2^{31+r} + 1/2)? These are the double vs single rounding. Double rounding to nearest (ties up) composed equals single rounding to nearest when... it's known that double rounding differs when the intermediate lands exactly on a tie. But here the kernel computes single rounding of X/2^{31+r} to nearest half-up. Target is double rounding. Are they equal for this domain?

Hmm wait — is the kernel actually doing that? The kernel adds the second-stage constant 2^{r+30} (for r>=2), total divisor 2^{31+r}, constant 2^{30} + 2^{r+30} = 2^30(1 + 2^r). Half of divisor 2^{31+r} is 2^{30+r}. The added constant is 2^30 + 2^{30+r}, which is 2^30 more than half. Hmm interesting: 2^{30+r} + 2^30. So Rk = floor((X0 + 2^{30+r} + 2^30)/2^{31+r}) where X0 = P. Note X = P + 2^30, so Rk = floor((X + 2^{30+r})/2^{31+r}) = round half up of X/2^{31+r}... wait X already includes 2^30. Yes: P + 2^30 + 2^{30+r} over 2^{31+r}: numerator constant = 2^30 + 2^{30+r} = 2^30(1+2^r), half of 2^{31+r} = 2^{30+r}. So constant = half + 2^30.

Hmm, so Rk = floor( (P + 2^30 + 2^{30+r}) / 2^{31+r} ) + something? Let me recompute with X = P + 2^30 (the first-stage-adjusted value). Kernel numerator: X + 2^{30+r}... wait for r>=2: added is 2^{r+30} via high_b? high_b = (1 << (r-2)) and it's added to high (the 2^32 word), so contributes 2^{r-2} * 2^32 = 2^{r+30}. Yes. So Rk = floor((X + 2^{r+30}) / 2^{r+31}).

Hmm but that's just half the divisor, i.e., round-half-up of X/2^{31+r}. But wait — where did the first-stage rounding go? The first stage rounding constant 2^30 is inside X. So the kernel computes round-half-up(X / 2^{31+r}) — a SINGLE rounding — whereas the target is double rounding: first round X/2^31 to nearest (ties up), then round that by 2^r more with ties away from zero. For nonnegative values ties away = ties up, so target = round-half-up composed twice.

Is double round-half-up = single round-half-up? floor(floor(X/2^31 + 1/2)/2^r + 1/2) vs floor(X/2^{31+r} + 1/2). Let X = 2^{31+r} * k + s, s∈[0,2^{31+r}). Single: k + [s >= 2^{30+r}]. Double: first H = floor(X/2^31 + 1/2) = floor((2^{31+r}k + s + 2^30)/2^31) = 2^r k + floor((s + 2^30)/2^31). Then R = floor(H/2^r + 1/2) = k + floor((u + 2^{r-1})/2^r) where u = floor((s+2^30)/2^31) ∈ [0, 2^r]. Hmm.

Take s = 2^{30+r} - 2^30 (just below single-rounding tie... single gives k since s < 2^{30+r}). Double: u = floor((s + 2^30)/2^31) = floor((2^{30+r})/2^31) = 2^{r-1}. Then R = k + floor((2^{r-1}+2^{r-1})/2^r) = k+1. So double gives k+1, single gives k. MISMATCH! So the kernel (single rounding) differs from target (double rounding) when s = 2^{30+r} - 2^30 exactly... wait need s such that first stage rounds up past the boundary creating a tie at second stage. Yes at s = 2^{31+r-1} - 2^30, double rounds up, single rounds down. So there's a discrepancy region. Hmm — but wait, does the kernel really do single rounding? Let me recheck.

Hmm, actually let me recheck the kernel for r>=2 carefully. Numerator additions: low_a adds 2^30 (with carry into high via carry_a). high_b = 2^{r-2} added to high. So high_final = high + 2^{r-2} + carry_a + carry_b (carry_b from low additions). low_final = low + 2^30 (+low_b=0 for nonneg, r>=2). So high_final = floor((P + 2^30 + 2^{r-2}*2^32)/2^32) = floor((P + 2^30 + 2^{r+30})/2^32). shifted = floor(high_final/2^{r-1}) = floor((P + 2^30 + 2^{r+30})/2^{r+31}).

Target double: R = floor((H + 2^{r-1})/2^r), H = floor((P+2^30)/2^31).

Compare: Rk = floor((P + 2^30 + 2^{30+r})/2^{31+r}).

Hmm, but is Rk actually equal to double rounding? Consider the decomposition: floor((P + 2^30)/2^{31+r} ...). Let me test concrete numbers: r=2, P such that... Let me pick P + 2^30 = X. Choose X = 2^{33}*k + s, s = 2^{32} - 2^30 (i.e., 3*2^30). r=2, divisor 2^{33}, half = 2^{32}. s = 3*2^30 < 2^32 → single rounds to k. Double: H = floor(X/2^31 + 1/2) = floor((2^{33}k + 3*2^30 + 2^30)/2^31) = 4k + floor(4*2^30/2^31) = 4k + 2. R = floor((4k+2 + 2)/4) = k + 1. Kernel: Rk = floor((X + 2^{32})/2^{33}) = floor((2^{33}k + 3*2^30 + 2^32)/2^33) = k + floor((3*2^30+2^32)/2^33) = k + floor(7*2^30/2^33) = k + 0 (7*2^30 = 0.875*2^33) → k. Mismatch: target k+1, kernel k.

But wait — need P = X - 2^30 achievable with integer a, m in domain, and P = a*m with m in [2^30, 2^31-1]. With m = 2^30 exactly (legal, m>=2^30): a*2^30 = P. Choose P = 2^33*k + 3*2^30 + 2^30 - 2^30... let me redo: X = P + 2^30. Want X = 2^33 + 3*2^30 (k=1). So P = 2^33 + 3*2^30 - 2^30 = 2^33 + 2^31 = 2^31*(4+1)=5*2^31. a = P/m = 5*2^31/2^30 = 10. Legal.

So a=10, m=2^30, r=2, z=0. Target: P = 5*2^31 = 10737418240. H = floor((P + 2^30)/2^31) = floor((5*2^31 + 2^30)/2^31) = 5 + floor(2^30/2^31)= 5. Hmm wait: (5*2^31 + 2^30)/2^31 = 5.5 → floor 5. H=5. R = floor((5+2)/4)? r=2: (5 + 2)/4 = 1.75 → 1? Wait 2^{r-1}=2. (5+2)//4 = 1. R=1.

Kernel: Rk = floor((P + 2^30 + 2^{32})/2^{33}) = floor((5*2^31 + 2^30 + 2^32)/2^33) = floor((10*2^32 + 2^30 + 2^32)/2^33) = floor(11*2^32 + 2^30)/2^33 → 11*2^32/2^33 = 5.5, plus 2^30/2^33 → total 5.5 + 0.0625 = 5.5625 → floor 5. So Rk = 5?? But wait then shifted = high_final >> 1... hold on, I think I mis-derived. Let me recompute high_final and shifted directly.

P = 5*2^31 = 0xA00000000. high = 5, low = 0. low_a = 0 + 2^30 = 2^30. carry_a = 0. low_b = 0 (a=10 ≥ 0, r=2 not 1, and (r>1)&negative false). high_b = (1 << (r-2)) - 0 = 1. low_final = 2^30, carry_b = 0. high_final = 5 + 1 + 0 + 0 = 6. shifted = 6 >> (r-1)=1 → 3.

Target R = 1. Kernel gives 3. Hmm, that's way off! Wait, did I mess up target? H = 5, R = round(5/4) = 1 (5/4=1.25). Kernel gives 3?? That can't be right; the kernel can't be that wrong... Let me recheck high_b. high_b = tl.where(r >= 2, (1 << tl.maximum(r - 2, 0)) - negative.to(tl.int32), 0). For r=2, negative=false: (1 << 0) - 0 = 1. high_final = 5+1 = 6, shifted = 6>>1 = 3.

Hmm wait, but maybe I'm misinterpreting: maybe high_final is meant to be (P + 2^30 + rounding2)/2^32 where the correct fused value should give floor((...)/2^{31+r})... Let's compute what the correct fused numerator should be. Target R for H>=0: floor((H + 2^{r-1})/2^r) with H = floor((P+2^30)/2^31).

Alternatively correct fusion: R = floor( (floor((P+2^30)/2^31) + 2^{r-1}) / 2^r ). Note floor((P+2^30)/2^31) = floor((P + 2^30 + 2^{30+r} - 2^30... hmm. Standard: floor((floor(N/D)+c)/E) = floor((N + c*D... no.

Let me just compute what numerator N makes floor(N/2^{31+r}) = target. Target = floor((H + 2^{r-1})/2^r) = floor((H*2^31 + 2^{30+r})/2^{31+r}). And H*2^31 <= P + 2^30 < H*2^31 + 2^31. So target = floor((floor((P+2^30)/2^31)*2^31 + 2^{30+r})/2^{31+r}).

For our example: H=5, target = floor((5*2^31 + 2^32)/2^33) = floor((10+2)*2^32/2^33)= floor(12/2)=... 12*2^32/2^33 = 6 → target = 6?? But direct computation gave R=1. Contradiction! Let me recheck: R = floor((H+2^{r-1})/2^r) with H=5, r=2: (5+2)/4 = 1.75 floor 1. But the fusion formula floor((H*2^31 + 2^{30+r})/2^{31+r}): H*2^31/2^{31+r} = H/2^r = 5/4 = 1.25; 2^{30+r}/2^{31+r} = 0.5. Sum 1.75, floor 1. I arithmetic-errored: 2^{30+r} = 2^32, divided by 2^33 = 0.5. So (5*2^31 + 2^32)/2^33 = (10*2^32 + 2^32)/2^33 = 11*2^32/2^33 = 5.5 → floor 5?? Wait 5*2^31 = 10*2^32? No: 5*2^31 = 5*2^31; 2^32 = 2*2^31. 5*2^31 + 2*2^31 = 7*2^31. 7*2^31/2^33 = 7/4 = 1.75 → 1. Good, R=1. I keep messing up.

Now kernel: high_final=6, shifted = 6>>1 = 3. But target 1. So kernel is wrong?? That seems too broken. Hmm wait — maybe I mis-derived the intended fused formula. Perhaps high_final is supposed to be floor((P + 2^30 + 2^{30+r}) / 2^32)? Then shifted = that >> (r-1) = floor((P+2^30+2^{30+r})/2^{31+r}). For our example: P+2^30+2^32 = 5*2^31 + 2^30 + 2^32 = (10 + 1 + 4)*2^30 = 15*2^30? 5*2^31 = 10*2^30; 2^32 = 4*2^30. Total 10+1+4=15*2^30. Divided by 2^33 = 8*2^30: 15/8 = 1.875 → 1. So that fusion would give 1 = correct! But kernel computed high_final = 6, meaning floor(P + 2^30 + 2^{32})/2^32? (10+1+4)*2^30 / 2^32 = 15/4 = 3.75 floor 3? But I computed high_final = 6 via components. Discrepancy: high_b = 1 added to high (=5) gives 6, but floor((P + 2^30 + 1*2^32)/2^32) = floor((10+1+4)*2^30/2^32) = floor(15/4)=3, not 6. Oh! Because carry semantics: adding 2^30 to low doesn't carry, but the "correct" numerator for high is P + 2^30 → high gets carry only if low+2^30 overflows. The first-stage constant 2^30 is a low-word addition; so high_final = high + high_b + carry = floor((P + 2^30)/2^32) + high_b roughly. floor((P+2^30)/2^32) = floor((10+1)*2^30/2^32) = floor(11/4) = 2. Plus high_b=1 → 3. But I computed high_final = high(5) + high_b(1) = 6. Ah — high = mul.hi = 5 = floor(P/2^32) = floor(10*2^30/4*2^30) = 2?? Wait P = 5*2^31 = 2.5*2^32, so high = 2, low = 2^31. I earlier said high=5, low=0 — wrong! P = 5*2^31 = 0xA_0000_0000. P/2^32 = 2.5. high=2, low = 2^31 = 0x80000000.

Redo: low_a = 2^31 + 2^30, no overflow (0xC0000000). carry_a=0. high_b = 1. high_final = 2+1 = 3. shifted = 3>>1 = 1. Target R=1. ✓.

OK so my earlier discrepancy analysis was based on arithmetic error. Let me redo the general correctness analysis.

Define P = high*2^32 + low, high = floor(P/2^32) (signed mul.hi correct for negatives), low unsigned.

Stage 1: H = floor((P + 2^30)/2^31).

Kernel r=0: q0 = (high_final<<1)|(low_final>>31) with high_final = high + carry_a (high_b=0,carry_b=0), low_final = low + 2^30 (mod 2^32). H = floor((P+2^30)/2^31). Check: P+2^30 = (high+carry_a)*2^32 + low_final (with carry_a handled). Then floor(/2^31) = 2*(high+carry_a) + (low_final>>31). Yes q0 = H. ✓ for r=0.

r=1, P>=0 (a>=0): low_b = 2^31, high_b=0. Numerator N = P + 2^30 + 2^31. q = high_final >> 0 = high_final = floor(N/2^32). Target: H = floor((P+2^30)/2^31), R = floor((H+1)/2) (H>=0). Is floor((P + 2^30 + 2^31)/2^32) = floor((floor((P+2^30)/2^31)+1)/2)? Let X = P + 2^30, H = floor(X/2^31). Claim: floor((X + 2^31)/2^32) = floor((H+1)/2). Write X = H*2^31 + t, 0<=t<2^31. X + 2^31 = H*2^31 + t + 2^31. floor(/2^32) = floor((H + 1 + t/2^31)/2). H+1 + fraction in [0,1). floor((H+1+f)/2) where f = t/2^31. If H odd: H+1 even, = 2m: floor(m + f/2)... (H+1+f)/2 = m + f/2, f/2 < 0.5 → floor = m. Target floor((H+1)/2) = m. Equal. If H even = 2m: (2m+1+f)/2 = m + (1+f)/2, (1+f)/2 ∈ [0.5, 1) → floor = m + 0 if (1+f)/2 < 1, i.e., f<1 always, and >= 0.5 → floor = m. Target floor((H+1)/2) = floor((2m+1)/2) = m. Equal. ✓. So r=1 nonneg correct.

r=1, a<0 (P<0, P≠... note a<0 → P<0; also a<0 → H could still be... H = floor((P+2^30)/2^31) could be 0 or negative). Hmm, target tie rule: AWAY from zero, so for H<0: R = -floor((|H|+1)/2) = ceil(H/2 - 0.5)... i.e. round half toward -inf? Away from zero for negatives means ties go down (more negative): R = -floor((−H+1)/2). For H = -1: R = -floor(2/2) = -1. Round(-0.5) away from zero = -1. ✓.

Kernel r=1, negative: low_b = 0 (tl.where(negative, 0, 0x80000000) → negative → 0). So N = P + 2^30, q = high_final = floor((P+2^30)/2^32)... wait shifted = high_final >> max(r-1,0) = high_final >> 0 = high_final. Hmm, but that's floor(X/2^32), not the double-rounded R!

Wait, for r=1, q = shifted = high_final >> 0 = high_final. For negative a: high_final = floor((P + 2^30)/2^32). Target R = round half away from zero of H/2, H = floor((P+2^30)/2^31).

Hmm that seems wrong: for negative a, r=1, kernel gives floor(X/2^32) where X = P+2^30. Target: H = floor(X/2^31), R = -floor((-H+1)/2).

Test: a = -1, m = 2^30, r=1, z=0. P = -2^30. X = 0. H = 0. R = 0 (round(0)=0). Kernel: high = floor(-2^30/2^32) = -1, low = 2^32 - 2^30 = 0xC0000000. low_a = 0xC0000000 + 0x40000000 = 0x1_0000_0000 → wraps to 0, carry_a = 1. low_b: r==1, negative → 0. high_b=0. low_final = 0, carry_b = 0. high_final = -1 + 0 + 1 + 0 = 0. shifted = 0. q=0. ✓.

Test negative with tie: want H odd negative, e.g., H = -1. Need floor(X/2^31) = -1: X ∈ [-2^31, 0). P = X - 2^30 ∈ [-3*2^30, -2^30). Take P = -2^30 - 1? Hmm P = a*m. Let a = -1, m = 2^30 + something? m must be ≥2^30. P = -m. X = 2^30 - m. For m = 2^30 + 2^29? X = -2^29 ∈ [-2^31,0) → H = floor(-2^29/2^31) = -1. Target: R = -floor((1+1)/2) = -1 (round(-0.5) away → -1). Kernel: P = -(2^30+2^29). high = floor(P/2^32) = -1, low = 2^32 - (2^30+2^29) = 0xA0000000... 2^32 - 0xC0000000? 2^30+2^29 = 0x60000000. low = 2^32 - 0x60000000 = 0xA0000000. low_a = 0xA0000000 + 0x40000000 = 0xE0000000, no carry. low_b = 0. high_final = -1 + 0 + 0 + 0 = -1. q = -1. ✓ matches R=-1.

Another negative test, H=-3, r=1: R = -floor(4/2) = -2. Kernel should give floor(X/2^32) where X = P+2^30, H = floor(X/2^31) = -3 → X ∈ [-3*2^31, -2*2^31). floor(X/2^32) ∈ {-2, -1}? X ∈ [-6*2^30, -4*2^30), X/2^32 ∈ [-1.5, -1) → floor = -2. ✓ equals -2. Hmm interesting, generally for negative: target R = -floor((-H+1)/2). With H negative. Let H = -k, k>0. R = -floor((k+1)/2). Kernel q = floor(X/2^32), X = H*2^31 + t, t∈[0,2^31), H=-k: X = -k*2^31 + t. q = floor((-2k + t/2^31)/2) = floor(-k + t/2^32... hmm: X/2^32 = (-k*2^31 + t)/2^32 = -k/2 + t/2^32. q = floor(-k/2 + f), f = t/2^32 ∈ [0, 0.5). If k even, k=2m: q = floor(-m + f) = -m (f<1). Target: -floor((2m+1)/2) = -m. ✓. If k odd = 2m+1: q = floor(-m - 0.5 + f), f∈[0,0.5) → value in [-m-0.5, -m) → floor = -m-1. Target: -floor((2m+2)/2) = -(m+1). ✓. Great, r=1 negative correct. Neat — for r=1 negative the kernel omits the second rounding constant because ties-away-down for negatives is achieved... yes as shown.

Also a<0 but H≥0 possible? a<0 → P<0, X = P+2^30 could be ≥0 if P ∈ [-2^30+... P negative down to -2^30+1? If P ∈ (-2^30, 0), X ∈ (0, 2^30)? Wait X = P+2^30 ∈ (0, 2^30)? P ∈ (-2^30,0) → X ∈ (0, 2^30). H = 0. Target R = 0 (round(0/2^r)). Kernel: negative branch (a<0) omits low_b. q = floor((P + 2^30)/2^{31+r})? For r=1: floor(X/2^32) = 0 for X ∈ (0,2^30). ✓. For r>=2 negative: low_b = 0x80000000, high_b = 2^{r-2} - 1. Let's analyze.

r>=2, a>=0: low_b=0, high_b = 2^{r-2}. q = high_final >> (r-1), high_final = floor((P + 2^30 + 2^{r-2}*2^32)/2^32) (carry_b=0 since low_final = low+2^30 could carry — carry_a handles that; carry_b from low_b=0 addition is 0). Actually careful: high_final = high + high_b + carry_a + carry_b where carry_a from low+2^30. So high_final = floor((P + 2^30 + high_b*2^32)/2^32). Then q = floor(high_final / 2^{r-1}) = floor((P + 2^30 + 2^{r+30})/2^{31+r}).

Target (H>=0): R = floor((H + 2^{r-1})/2^r), H = floor((P+2^30)/2^31) ≥ 0.

Check equality: Let X = P + 2^30 ≥ 0 (assuming P ≥ -2^30; if a≥0 then P≥0 so fine). H = floor(X/2^31). Claim: floor((X + 2^{31} * 2^{r-1})/2^{31+r}) = floor((floor(X/2^31) + 2^{r-1})/2^r). This is the identity floor((X + c*2^31)/2^{31+r}) = floor((floor(X/2^31)+c)/2^r)? Hmm. General: floor((floor(X/2^31)+c)/2^r) = floor((X + c*2^31)/2^{31+r})? Let X = h*2^31 + t. LHS = floor((h+c)/2^r) (since (h+c) integer... wait (floor(X/2^31)+c)/2^r = (h+c)/2^r, floor of that). RHS = floor((h*2^31 + t + c*2^31)/2^{31+r}) = floor(((h+c)*2^31 + t)/2^{31+r}) = floor((h+c)/2^r + t/2^{31+r}). Since (h+c)/2^r may be fractional with fraction f, adding t/2^{31+r} < 1/2^r ≤ ... could cross an integer boundary only if f + t/2^{31+r} ≥ 1, needs t/2^{31+r} ≥ 1 - f ≥ 1/2^r... t < 2^31, so t/2^{31+r} < 1/2^r ≤ 1/2 (r≥1). f = ((h+c) mod 2^r)/2^r, and 1 - f ≥ 1/2^r. Need t/2^{31+r} ≥ 1/2^r → t ≥ 2^31, impossible (t < 2^31). So equal. ✓. With c = 2^{r-1}: RHS numerator constant 2^{r-1}*2^31 = 2^{30+r} ✓ matches kernel. So r≥2, a≥0 correct (assuming H≥0, which holds since P≥0 → H≥... P≥0 → X≥2^30 → H≥0). ✓.

r>=2, a<0: low_b = 0x80000000, high_b = 2^{r-2} - 1. Numerator additions: 2^30 (low) + 2^31 (low_b) + (2^{r-2}-1)*2^32 (high_b). Plus carries. high_final = floor((P + 2^30 + 2^31 + (2^{r-2}-1)*2^32)/2^32) = floor((P + 2^30 + 2^31 + 2^{r+30} - 2^32)/2^32) = floor((P + 2^30 + 2^{r+30})/2^32) - 1? No: = floor((P + 2^30 + 2^{r+30} + 2^31 - 2^32)/2^32). Hmm, 2^31 - 2^32 = -2^31. So = floor((P + 2^30 + 2^{r+30} - 2^31)/2^32)? Wait: (2^{r-2}-1)*2^32 = 2^{r+30} - 2^32. Total low-word additions: 2^30 + 2^31. So overall added = 2^30 + 2^31 + 2^{r+30} - 2^32 = 2^{r+30} + 2^30 - 2^31 = 2^{r+30} - 2^30.

q = floor((P + 2^{r+30} - 2^30)/2^{31+r})? Wait need care: adding low quantities may carry, but high_final = floor(total/2^32) if we treat sum properly: sum = P + 2^30 + 2^31 + (2^{r-2}-1)*2^32, and high_final = floor(sum/2^32), then q = floor(high_final/2^{r-1}) = floor(sum/2^{31+r}). sum = P + 2^{r+30} - 2^30... let me recompute: 2^30 + 2^31 - 2^32 = 2^30 + 2^31 - 2^31 - 2^31 = 2^30 - 2^31 = -2^30. So sum = P + 2^{r+30} - 2^30. q = floor((P + 2^{30+r} - 2^30)/2^{31+r}).

Target for H<0: R = -floor((-H + 2^{r-1})/2^r) = ceil(H/2^r - 2^{r-1}/2^r)... equivalently R = -round_half_up(-H / 2^r). Alternative: R = ceil((H - 2^{r-1})/2^r)? Check: for H=-k<0: -floor((k + 2^{r-1})/2^r) = ceil((-k - 2^{r-1})/2^r). Yes since -floor(x) = ceil(-x). So R = ceil((H - 2^{r-1})/2^r) = -floor((2^{r-1} - H)/2^r).

Now kernel q = floor((P + 2^{30+r} - 2^30)/2^{31+r}). Let X = P + 2^30 (so H = floor(X/2^31), and note for a<0, H could be 0 or negative). q = floor((X + 2^{30+r} - 2^31)/2^{31+r}) = floor((X - 2^31 + 2^{30+r})/2^{31+r}).

Target: R = ceil((H - 2^{r-1})/2^r).

Case H<0: H = -k. R = ceil((-k - 2^{r-1})/2^r). Kernel: X = H*2^31 + t = -k*2^31 + t, t∈[0,2^31). q = floor((-k*2^31 + t - 2^31 + 2^{30+r})/2^{31+r}) = floor((-(k+1)*2^31 + t + 2^{30+r})/2^{31+r}) = floor( (-(k+1) + 2^{r-1})/2^r + t/2^{31+r} ).

Target R = ceil((-k - 2^{r-1})/2^r) = -floor((k + 2^{r-1})/2^r).

Set A = (-(k+1) + 2^{r-1})/2^r + t/2^{31+r}, q = floor(A). Let's verify numerically. Example: r=2, k=1 (H=-1). Target R = -floor(3/4) = -1? Hmm wait round-half-away of -1/4 = -0.25 → 0? No! R = round(H/2^r) = round(-0.25) = 0 (nearest). Wait, formula: -floor((|H| + 2^{r-1})/2^r) = -floor((1+2)/4) = -floor(0.75) = 0. I earlier wrote -floor((k+2^{r-1})/2^r) = -0 = 0. Right, R=0.

Kernel: q = floor((-(1+1) + 2)/4 + t/2^33) = floor(0 + t/2^33) = 0 (t<2^31<2^33). ✓.

Example r=2, H=-2: target = -floor(4/4) = -1 (round(-0.5) away → -1). Kernel: q = floor((-(3) + 2)/4 + t/2^33) = floor(-0.25 + small) = -1. ✓.

Example r=2, H=-6: target = -floor(8/4) = -2 (round(-1.5) away → -2). Kernel: floor((-(7)+2)/4 + eps) = floor(-1.25+eps) = -2. ✓.

Example r=2, H=-5: target = -floor(7/4) = -1 (round -1.25 → -1). Kernel: floor((-6+2)/4+eps) = floor(-1+eps) = -1. ✓ (t>0 gives eps>0; t=0 → exactly -1, floor -1 ✓).

Hmm wait, but t=0 and (-(k+1)+2^{r-1}) divisible by 2^r: e.g., r=2, k+1 = 2^{r-1}=2 → k=1 handled. Take r=3, k+1=4 → k=3, H=-3: target = -floor((3+4)/8) = -floor(0.875) = 0. Kernel: floor((-4+4)/8 + eps) = 0. ✓.

Take r=3, H=-4: target = -floor((4+4)/8) = -1. Kernel: floor((-5+4)/8+eps) = floor(-0.125+eps) = -1 (eps < 2^31/2^34 = 0.125; hmm eps = t/2^34, t<2^31 → eps < 0.125; -0.125 + eps ∈ [-0.125, 0) → floor = -1 ✓).

General proof sketch for H<0: q = floor( (-(k+1) + 2^{r-1})/2^r + δ ), δ∈[0, 2^{-r}... δ = t/2^{31+r} ∈ [0, 2^{-r}). Target R = -floor((k+2^{r-1})/2^r). Note (-(k+1)+2^{r-1})/2^r = -(k+1-2^{r-1})/2^r = -(k + 2^{r-1} - 2^r +... hmm let me relate: (k + 2^{r-1}) + (-(k+1) + 2^{r-1}) = 2^r - 1. So -(k+2^{r-1}) = (k+... let u = k + 2^{r-1}. Then -(k+1)+2^{r-1} = 2^r - 1 - u... check: -(k+1)+2^{r-1} = -(k+1) + 2^{r-1}; and 2^r - 1 - u = 2^r -1 - k - 2^{r-1} = 2^{r-1} - 1 - k. Yes equal. So the expression = (2^r - 1 - u)/2^r + δ = 1 - (u+1)/2^r + δ... (2^r -1 -u)/2^r = 1 - (u+1)/2^r. So q = floor(1 - (u+1)/2^r + δ) = 1 + floor(-(u+1)/2^r + δ) = 1 - ceil((u+1)/2^r - δ) = ... Target R = -floor(u/2^r) = -floor((u+1)/2^r) unless u ≡ 0 mod 2^r... hmm u = k + 2^{r-1}; u ≡ 0 mod 2^r possible.

q = floor(1 - (u+1)/2^r + δ). Let (u+1)/2^r = w + φ, w = floor, φ∈{0}∪(0,1). q = floor(1 - w - φ + δ). If φ=0: q = floor(1 - w + δ) = 1 - w + floor(δ)... δ<1 so q = 1-w if δ... 1-w+δ with δ∈[0,1): floor = 1-w. Hmm unless δ... floor(1-w+δ) = 1-w for δ∈[0,1). Target R = -floor(u/2^r) = -(floor((u+1)/2^r)) since u+1 ≡ 1... wait if φ=0 then u+1 ≡ 0 mod 2^r, u ≡ -1 mod 2^r, floor(u/2^r) = w - 1 (since u = (u+1) - 1 = w*2^r - 1 → floor(u/2^r) = w-1). R = -(w-1) = 1-w. q = 1-w ✓.

If φ>0: q = floor(1 - w - φ + δ), δ < 2^{-r} ≤ ... φ ≥ 1/2^r (u+1 not divisible, φ = j/2^r, j≥1). δ ∈ [0, 1/2^r). φ - δ > 0. So 1-w-φ+δ ∈ (1-w-φ, 1-w) → floor = -w if 1-w-φ+δ ≥ -w... it's < 1-w and > 1-w-1? φ<1 so 1-w-φ > -w → floor = -w... wait floor of value in (-w... value ∈ (1-w-1, 1-w) = (-w, 1-w) → floor = -w? Value could equal exactly... value > 1-w-φ ≥ 1-w-(1-1/2^r) = -w+1/2^r > -w. And value < 1-w. So floor = -w. Target: R = -floor(u/2^r). u+1 = w*2^r + (φ*2^r), φ*2^r ∈ [1, 2^r-1]. u = w*2^r + φ*2^r - 1, so floor(u/2^r) = w (since φ*2^r -1 ∈ [0, 2^r-2]). R = -w. ✓.

Great, negative a, r≥2 correct. Edge: a<0 but H=0 (P ∈ [-2^30, 0)? Actually a<0 → P<0; H = floor((P+2^30)/2^31); H=0 when P ∈ [-2^30, ... P+2^30 ∈ [0, 2^31) → P ∈ [-2^30, 2^31 - ... wait P<0 so P ∈ [-2^30, 0) → H=0). Target R = 0. Kernel: q = floor((X - 2^31 + 2^{30+r})/2^{31+r}) with X = P+2^30 ∈ [0, 2^30). q = floor((X - 2^31 + 2^{30+r})/2^{31+r}). For r=1: constant -2^31 + 2^31 = 0 → q = floor(X/2^32) = 0 ✓ (handled earlier). For r≥2: numerator ∈ [0 - 2^31 + 2^{32}, 2^30 - 2^31 + 2^{30+r}) = [2^31, ...) ≥ 0? -2^31 + 2^{30+r} with r≥2 ≥ -2^31+2^32 = 2^31 > 0. Divided by 2^{31+r}: value = 2^{31}+X ... /2^{31+r} < 2^{32}/2^{33} = 0.5 for r=2 → floor 0 ✓. Generally numerator < 2^30 + 2^{30+r} ≤ 2^{30+r} + 2^{30+r} = 2^{31+r} (for r≥1: 2^30 ≤ 2^{30+r}) → q could be... numerator max = 2^30 - 2^31 + 2^{30+r} - 1 < 2^{30+r} + ... need < 2^{31+r}: 2^30 - 2^31 + 2^{30+r} < 2^{30+r} + 2^{30+r} = 2^{31+r} ✓ (since 2^30 - 2^31 < 2^30 ≤ 2^{30+r}, strict as 2^30-2^31<0<2^{30+r}... actually 2^30 - 2^31 = -2^30 < 2^30). So q < 1, q ≥ 0 → q=0 ✓. And H could be... a<0 can H be positive? P<0, X = P+2^30 < 2^30 < 2^31 → H ≤ 0. Fine.

Now saturation and zero point: q clamped to [-255, 255] then q+z clamped [-128,127]. Pre-clamp to ±255: since |z| ≤ 128, q+z ∈ [-383, 383]; clamping q to [-255,255] first then adding z and clamping: is that same as clamping q+z directly? Target: Q = clamp(R + z). R can be huge (H up to ~2^31·2^31/2^31... P up to (2^31-1)^2 ≈ 2^62, H ≈ 2^31, R ≈ 2^31/2^r — big). Kernel clamps q to [-255,255] BEFORE adding z. Is clamp(clamp(R,-255,255)+z, -128,127) == clamp(R+z, -128,127)? If R > 255: kernel gives clamp(255+z) = 127 if z ≥ ... 255+z ≥ 127 → clamp → 127. Target: R+z > 255+z ≥ 127 → 127. ✓. If R < -255: kernel: -255+z ≤ -128 → clamp -128. Target: R+z < -255+z ≤ -128 → -128 ✓. If R ∈ [-255,255]: kernel = clamp(R+z) ✓ same. So fine. Why ±255? maybe to avoid int32 overflow? Not needed but harmless. Actually wait, is q within int32? R could be as large as ~2^31 (r=0, a=2^31-1, m=2^31-1 → P≈2^62, H≈2^31). H up to floor((2^62)/2^31)= 2^31 - stuff ≈ 2^31, exceeds int31 but fits int32? H_max = floor(((2^31-1)^2 + 2^30)/2^31) = floor((2^62 - 2^32 + 2^30 + ...)/2^31) ≈ 2^31 - 2 - ... = 2^31 - 3 or so. q0 = (high_final << 1) | ... high_final could be ~2^31? high_final = floor((P+2^30)/2^32) ≈ 2^30. q0 ≈ 2^31 - small → could it overflow int32? q0 = (high_final.to(uint32) << 1) | (low>>31) then to int32. If high_final = 2^31-1... high_final max: P_max = (2^31-1)^2 ≈ 2^62, /2^32 → ≈ 2^30. So high_final ≤ ~2^30, q0 ≤ ~2^31 - 2 → fits int32 as positive? 2^31-2 < 2^31-1 ✓. Negative side: P_min = -2^31 * (2^31-1) ≈ -2^62; but wait the problem says the exceptional case a = -2^31 with m... m ≤ 2^31-1 so P = -2^31 * m. a = -2^31 is legal! P = -2^31 * (2^31-1) ≈ -2^62. high = floor(P/2^32) ≈ -2^30. q0 ≈ -2^31 → fits int32 (≥ -2^31) ✓. Also note tl.maximum(q, -255) works.

Also mul.hi.s32 with a = -2^31: fine, m positive, product fits in 64-bit.

Wait one more: for negative a, low = a.to(uint32)*m.to(uint32) — a = -2^31 → uint32 = 2^31, times m as uint32 — product mod 2^32 correct low word ✓.

Indexing: index over rows*channels, channel = index % CHANNELS — since acc is contiguous [M,C], flat index mod C gives channel ✓. Loads of per-channel params use mask=active with other values — fine. BLOCK=128, grid covers all. out fresh tensor ✓. FP32 conversion: (q-z) int to float exact (small range), times scale (power of two) exact ✓.

r=1 negative recheck with general H<0 done. r=1 with a<0, H=0 done. r≥2 negative H<0 done. r≥2 positive done. r=0: q0 = H ✓, including negatives: q0 = 2*high_final + bit where high_final = high + carry_a. For negative P: H = floor((P+2^30)/2^31). P = high*2^32 + low (low unsigned, e.g., P=-1 → high=-1, low=2^32-1). P + 2^30 = high*2^32 + low + 2^30; if low + 2^30 ≥ 2^32, carry. q0 = 2*(high+carry) + (low_final>>31). floor((high*2^32+low_final... yes correct ✓.

One subtle: r=1 case, low_b chosen based on `negative` = a<0, but sign of H matters for tie direction; we verified math works out with a's sign (H<0 requires P<... P<0 requires a<0 since m>0; and a<0 → P<0; H≤0 for a<0 shown; a≥0 → P≥0 → X≥2^30 → H≥0 ✓... wait a≥0, H≥0? H = floor((P+2^30)/2^31), P≥0 → H≥0 ✓). Also a=0: negative=false, P=0, H = floor(2^30/2^31)=0, R=0, all paths: q = floor((0 + 2^30 + 2^{30+r})/2^{31+r}) = floor(2^{-r-1} + 0.5) = 0 for r≥0? 0.5 + 2^{-r-1} < 1 → 0 ✓.

Edge: high_b computation `(1 << tl.maximum(r - 2, 0)) - negative.to(tl.int32)` — for r≥2 branch only, r-2 ≥ 0 fine.

Edge: carry_a computed as (low_a < low) unsigned comparison — Triton uint32 comparison ✓. carry_b similar.

Also `low_b` for r>1 & negative is 0x80000000, for r==1 negative 0, r==1 nonneg 0x80000000, r==0... (r>1)&negative false, r==1 false → 0 ✓; r≥2 nonneg: tl.where branches: r==1? no; (r>1)&negative? no → 0 ✓.

One more check on r≥2, a≥0, but does the identity floor((X + c·2^31)/2^{31+r}) = floor((floor(X/2^31)+c)/2^r) hold including when... proven above with t<2^31 strict ✓. But wait kernel numerator is P + 2^30 + 2^{r+30}, and c·2^31 with c = 2^{r-1} gives 2^{r+30} ✓, and X = P + 2^30. ✓.

Also the shifted for r≥2: high_final >> (r-1) is arithmetic shift on int32 (high_final could be negative for negative P? high_final = floor(sum/2^32), sum ≈ P + 2^{r+30} which for very negative P is negative → arithmetic shift right = floor division ✓, Triton >> on int32 is arithmetic). ✓.

Potential issue: q clamp ±255 then result (q - z): after clamp to [-128,127], Q ∈ [-128,127], Q - z ∈ [-255, 255] exact in float ✓.

Grid: rows*channels up to 4096*257 ≈ 1M, fine.

I'm fairly convinced it's correct. One more paranoia: `tl.inline_asm_elementwise` with int32 mul.hi.s32 — correct semantics on T4 (sm75), Triton 3.4 supports it. a and m as int32; product 64-bit signed; mul.hi.s32 gives bits [63:32] which equals floor(P/2^32) for signed ✓.

low computation: a.to(uint32) * m.to(uint32) in Triton — uint32 multiply wraps mod 2^32 ✓ (Triton doesn't promote? uint32*uint32 stays uint32 with wraparound; yes Triton integer ops wrap).

Also mask other=1073741824 for MULT etc. — only used for inactive lanes, fine.

Verdict: trust, confidence ~0.85.

## Final response (verbatim)

{"verdict": "trust", "confidence": 0.85, "reason": "I verified the fused arithmetic case by case (r=0 via the explicit q0 path, r=1 and r>=2 with sign-dependent rounding constants low_b/high_b) against the double-rounding definition, including the tie cases and the identity floor((X+c*2^31)/2^(31+r)) = floor((floor(X/2^31)+c)/2^r) for nonnegative values, and the equivalent away-from-zero handling for negatives; the limb/carry handling, the pre-clamp to ±255 before adding z (shown equivalent to clamping R+z), saturation, and exact FP32 output all match the contract for the entire legal domain including a=-2^31."}

## Verdict

{
  "verdict": "trust",
  "confidence": 0.85,
  "reason": "I verified the fused arithmetic case by case (r=0 via the explicit q0 path, r=1 and r>=2 with sign-dependent rounding constants low_b/high_b) against the double-rounding definition, including the tie cases and the identity floor((X+c*2^31)/2^(31+r)) = floor((floor(X/2^31)+c)/2^r) for nonnegative values, and the equivalent away-from-zero handling for negatives; the limb/carry handling, the pre-clamp to ±255 before adding z (shown equivalent to clamping R+z), saturation, and exact FP32 output all match the contract for the entire legal domain including a=-2^31."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "trust",
    "confidence": 0.85,
    "reason": "I verified the fused arithmetic case by case (r=0 via the explicit q0 path, r=1 and r>=2 with sign-dependent rounding constants low_b/high_b) against the double-rounding definition, including the tie cases and the identity floor((X+c*2^31)/2^(31+r)) = floor((floor(X/2^31)+c)/2^r) for nonnegative values, and the equivalent away-from-zero handling for negatives; the limb/carry handling, the pre-clamp to ±255 before adding z (shown equivalent to clamping R+z), saturation, and exact FP32 output all match the contract for the entire legal domain including a=-2^31."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 2920,
    "output_tokens": 15291
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 235.64757208433002,
  "kernel_sha256": "4cccf9e78715e32dc8d435463758221f96458935c33e1e0742f632b7272499ba",
  "problem_sha256": "bb811a0e29d4018e97505f4467663a68c519437c59b730edee13a1dd470933ed",
  "prompt_variant": "original",
  "estimated_usd": 0.07136840000000001,
  "pricing_snapshot": {
    "input_per_million": 1.4,
    "output_per_million": 4.4,
    "cache_write_multiplier": 1.0,
    "cache_read_multiplier": 1.0,
    "basis": "published list price; cached input conservatively billed as uncached; excludes GPU",
    "checked_at": "2026-10-03",
    "source": "https://fireworks.ai/models/fireworks/glm-5p3"
  },
  "pricing": "dated list-price estimate; not invoice; excludes GPU and unreported HTTP usage"
}
