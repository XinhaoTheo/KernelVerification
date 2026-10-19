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

Compute a per-channel INT8 requantization epilogue for an integer GEMM.
The operation must reproduce the following SINGLE-rounding rule exactly.

For element a=acc[i,c], use m=multipliers[c], r=right_shifts[c], and
z=zero_points[c]. Define, with exact mathematical integer arithmetic:

  P = a*m
  R = floor((P + 2^(30+r)) / 2^(31+r))
  output[i,c] = min(127, max(-128, R + z))

Thus R is the nearest integer to a*m/2^(31+r), with exact halfway cases
rounded toward POSITIVE infinity. For example, -1.5 rounds to -1, and
+1.5 rounds to +2. Apply the zero point and INT8 saturation after rounding.
Do not substitute two-stage rounding or permit one-unit output errors.
The formula defines the required result, not a mandated implementation.

Complete public input domain:
- acc: int32 [M,C], 1<=M<=4096, 1<=C<=257. Every int32 accumulator is legal.
- multipliers: int32 [C], every value in [2^30, 2^31-1]. These are positive
  Q31 fixed-point multipliers, before the further right shift.
- right_shifts: int32 [C], every value in [0,31].
- zero_points: int32 [C], every value in [-128,127].
- Inputs are ordinary dense, strided, contiguous torch.Tensor objects on
  the same CUDA device. Contiguous views with nonzero storage offsets and
  input tensors that share storage are legal if their values satisfy the domain.
- All inputs have requires_grad=False, is_inference=False, is_neg=False,
  and is_conj=False. Use ordinary non-inference mode with no concurrent
  external writes. This forward operation does not provide autograd.

Return a fresh int8 CUDA tensor [M,C] on acc.device. Preserve every input.
Require exact integer equality at every output position: atol=0 and rtol=0.
The implementation gets one fixed accept/reject verdict over the entire
public domain; a failing legal input is evidence about the implementation.

validate_inputs() enforces the domain and run() calls it. reference() accepts
CPU inputs and applies the mathematical definition using bounded int64
arithmetic. make_inputs() supplies examples, not a restricted test domain.

This synthetic implementation uses the single-rounding form of the
fixed-point rescaling arithmetic found in TFLite. Exact compatibility is an
explicit requirement of this benchmark. It is not a claim that all TFLite
backends or all quantized GEMM APIs require identical rounding behavior.
This is not an upstream historical bug reproduction.

Frozen environment: NVIDIA T4, PyTorch 2.8.0, Triton 3.4.0.


## Kernel implementation under test (kernel.py)

```python
"""Single-rounding per-channel integer requantization.

Original synthetic Triton implementation of a fixed-point GEMM epilogue.
The task specifies exact compatibility with the stated single-rounding rule.
This file is not an upstream kernel or a historical bug reproduction.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _rescale_scalar(a, m, r, z):
    product = a.to(tl.int64) * m.to(tl.int64)
    bias = tl.full((), 1, tl.int64) << (30 + r)
    rounded = (product + bias) >> (31 + r)
    bounded = tl.minimum(tl.maximum(rounded, -255), 255).to(tl.int32)
    return tl.minimum(tl.maximum(bounded + z, -128), 127).to(tl.int8)


@triton.jit
def _requantize(ACC, MULT, SHIFT, ZERO, OUT,
                ROWS: tl.constexpr, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < ROWS * CHANNELS
    channel = index % CHANNELS
    a = tl.load(ACC + index, mask=active, other=0)
    m = tl.load(MULT + channel, mask=active, other=1073741824)
    r = tl.load(SHIFT + channel, mask=active, other=0)
    z = tl.load(ZERO + channel, mask=active, other=0)
    value = _rescale_scalar(a, m, r, z)
    tl.store(OUT + index, value, mask=active)


def validate_inputs(acc, multipliers, right_shifts, zero_points, require_cuda=True):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    buffers = (acc, multipliers, right_shifts, zero_points)
    for tensor in buffers:
        require(type(tensor) is torch.Tensor, "ordinary torch.Tensor required")
        require(tensor.layout == torch.strided and tensor.is_contiguous(),
                "contiguous strided tensors required")
        require(not tensor.requires_grad and not tensor.is_inference(),
                "ordinary non-gradient buffers required")
        require(not tensor.is_neg() and not tensor.is_conj(), "logical view flags excluded")
        require(not require_cuda or tensor.is_cuda, "CUDA tensors required by run")
        require(tensor.dtype == torch.int32, "all inputs must be int32")
    require(all(t.device == acc.device for t in buffers), "one device required")
    require(acc.ndim == 2, "acc must have shape [M,C]")
    rows, channels = acc.shape
    require(1 <= rows <= 4096 and 1 <= channels <= 257,
            "M must be 1..4096 and C must be 1..257")
    for tensor in buffers[1:]:
        require(tensor.shape == (channels,), "channel parameters must have shape [C]")
    require(bool((multipliers >= (1 << 30)).all().item()), "multipliers must be 2^30..2^31-1")
    require(bool(((right_shifts >= 0) & (right_shifts <= 31)).all().item()),
            "right shifts must be 0..31")
    require(bool(((zero_points >= -128) & (zero_points <= 127)).all().item()),
            "zero points must be -128..127")
    return {"valid": True, "rows": rows, "channels": channels}


def run(acc, multipliers, right_shifts, zero_points):
    validate_inputs(acc, multipliers, right_shifts, zero_points)
    rows, channels = acc.shape
    with torch.cuda.device(acc.device):
        output = torch.empty((rows, channels), dtype=torch.int8, device=acc.device)
        _requantize[(triton.cdiv(rows * channels, 128),)](
            acc, multipliers, right_shifts, zero_points, output,
            ROWS=rows, CHANNELS=channels, BLOCK=128, num_warps=4,
            enable_fp_fusion=False)
    return output


def reference(acc, multipliers, right_shifts, zero_points):
    """Direct single-rounding definition, with exact signed int64 arithmetic."""
    validate_inputs(acc, multipliers, right_shifts, zero_points, require_cuda=False)
    a = acc.cpu().to(torch.int64)
    m = multipliers.cpu().to(torch.int64)
    r = right_shifts.cpu().to(torch.int64)
    z = zero_points.cpu().to(torch.int64)
    product = a * m
    bias = torch.bitwise_left_shift(torch.ones_like(r), 30 + r)
    rounded = torch.bitwise_right_shift(product + bias, 31 + r)
    return torch.clamp(rounded + z, -128, 127).to(torch.int8).to(acc.device)


def make_inputs(device="cuda", rows=37, channels=65, seed=0):
    """Example generator; every int32 accumulator belongs to the public domain."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(-(1 << 31), 1 << 31, (rows, channels),
                      generator=g, dtype=torch.int64).to(torch.int32)
    m = torch.randint(1 << 30, 1 << 31, (channels,), generator=g,
                      dtype=torch.int64).to(torch.int32)
    r = torch.randint(0, 32, (channels,), generator=g, dtype=torch.int32)
    z = torch.randint(-128, 128, (channels,), generator=g, dtype=torch.int32)
    return tuple(t.to(device) for t in (a, m, r, z))

```

Is this implementation correct?

## Provider reasoning (verbatim)

Check arithmetic: product up to 2^31*2^31 = 2^62, plus bias 2^61 fits int64. floor division via arithmetic right shift matches floor for negative numbers (Python-like floor). Bias = 2^(30+r), shift 31+r — matches. Halfway rounds toward +inf: floor((P+bias)/2^(31+r)) — yes.

Overflow concerns: product int64 max ~2^62, plus bias 2^61 fine.

Inner clamp to ±255 before adding z: rounded could exceed ±255 only if a*m/2^(31+r) large — with m≥2^30 and r=0, a*m/2^31 = a/2 roughly, so |R| up to 2^30. Clamping to ±255 then to [-128,127] with z in [-128,127]: is it correct? If R=300, z=-200? z ≥ -128. Clamped to 255, +z could be 127 → clamp to 127. Correct answer: R+z ≥ 172 → 127. OK. If R=-300, clamp to -255, z=127 → -128, correct answer -173→-128. Fine. If R=200, z=-128: correct 72; clamped 255-128=127→127. Wrong! 200-128=72 vs 127. Can R=200 with z=-128? R = a*m >> (31+r). With m in [2^30, 2^31), a arbitrary int32. R = a*m/2^(31+r) roughly a/2 when r=0. So R=200 when a≈400. That's legal. Then output should be 72, kernel gives 127. Bug!

Wait, but bounded clamp to ±255 then bounded+z clamped — yes, real defect. E.g. a=400, m=2^30, r=0: P=400·2^30=2^30·400, bias=2^30, R=floor((400·2^30+2^30)/2^31)=floor(401/2)=200. z=-128 → 72. Kernel: bounded=200 (within 255), +z=72 → correct. Hmm wait, 200 is within ±255, so no clamp applied. The clamp only matters if |R|>255. If R=300, z=-128: true answer 172→127; kernel: clamp 255, +z=127→127. Correct coincidentally? R=300,z=-128 → 172→127; kernel 127. OK. R=300, z=0 → 127 both. R=-300, z=127 → -173→-128; kernel -255+127=-128→-128. OK. Actually since final clamp to [-128,127], any |R|>255 will map... R>255: true value R+z; with z≤127, R+z > 128? R≥256, z≥-128: R+z ≥128 → 127 always. Kernel: 255+z ∈ [127,382] → clamped ≥127... wait 255 + (-128) = 127, and clamped to ≤127 → 127. Correct. R<-255: R+z ≤ -256+127 = -129 → -128. Kernel: -255+z ∈ [-383,-128] → clamp -128. Correct. So the ±255 clamp is safe. Good.

Shift semantics: Triton `>>` on int64 — arithmetic shift, floor. Yes.

So kernel is correct. Check store/load indexing assumes contiguity — validated. Everything fine.

## Final response (verbatim)

{"verdict": "trust", "confidence": 0.9, "reason": "The kernel reproduces the exact single-rounding formula: int64 product (max ~2^62, no overflow), bias 2^(30+r), arithmetic right shift (floor for negatives), and the intermediate ±255 clamp is provably harmless because final clamping to [-128,127] makes any |R|>255 map identically (R>255 implies R+z≥128, R<-255 implies R+z≤-129). Halfway cases round toward +inf as required."}

## Verdict

{
  "verdict": "trust",
  "confidence": 0.9,
  "reason": "The kernel reproduces the exact single-rounding formula: int64 product (max ~2^62, no overflow), bias 2^(30+r), arithmetic right shift (floor for negatives), and the intermediate ±255 clamp is provably harmless because final clamping to [-128,127] makes any |R|>255 map identically (R>255 implies R+z≥128, R<-255 implies R+z≤-129). Halfway cases round toward +inf as required."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "trust",
    "confidence": 0.9,
    "reason": "The kernel reproduces the exact single-rounding formula: int64 product (max ~2^62, no overflow), bias 2^(30+r), arithmetic right shift (floor for negatives), and the intermediate ±255 clamp is provably harmless because final clamping to [-128,127] makes any |R|>255 map identically (R>255 implies R+z≥128, R<-255 implies R+z≤-129). Halfway cases round toward +inf as required."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 2164,
    "output_tokens": 904
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 14.898962208069861,
  "kernel_sha256": "4a29a6ff53a62c31b6fc146bf297ae466e783070ee0de108371d1723101a1608",
  "problem_sha256": "da272fe227bbed73afc3ca9e3d1c9201ba0be991aae8840f8e923d6235bb4a57",
  "prompt_variant": "original",
  "estimated_usd": 0.007007200000000001,
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
