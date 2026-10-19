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

Compute an unnormalized Walsh-Hadamard transform for each row of an int16
input tensor, returning int32 coefficients in increasing sequency order.

The mathematical target is defined independently of any fast algorithm:
1. Start with H_1 = [[1]] and recursively construct
       H_(2n) = [[H_n, H_n],
                 [H_n, -H_n]].
2. For each row of H_N, count how many adjacent entries have different
   signs. These counts are all distinct and range from 0 to N-1.
3. Sort the rows by that count, in ascending order, to obtain W_N.
4. Return out[b,k] = sum_j W_N[k,j] * x[b,j].

Thus output coefficient k uses a basis row with exactly k sign changes.
No normalization factor is applied, and output order is part of correctness.

Inputs and complete public domain:
- x is an ordinary torch.Tensor (not a subclass), with dtype int16 and
  shape [B,N], where 1<=B<=128 and N is one of {32,64,128}.
- EVERY stored int16 value is legal: the full range [-32768,32767].
- x is dense, strided, contiguous and on a CUDA device. Contiguous views
  with nonzero storage offsets are legal.
- x has requires_grad=False, is_inference=False, is_neg=False and
  is_conj=False. Calls run in ordinary non-inference mode, without
  concurrent external writes to the input. No autograd is provided.

Return a fresh int32 tensor of shape [B,N] on x.device, preserving x.
Every element must equal the mathematical integer target exactly:
absolute tolerance = 0 and relative tolerance = 0.
The largest possible absolute result is at most N*32768<=4194304, which
fits in int32. Approximate transforms, other coefficient orderings and
normalized transforms do not satisfy this task. Implementations may use
any algorithm that satisfies the same target over the whole legal domain.
Correctness is one verdict for the implementation, not a verdict per input,
input distribution, seed or size.

validate_inputs() enforces this domain and is called by run(). reference()
builds the Sylvester matrix, sorts its rows by sign-change count and uses
int64 products and sums before exact int32 conversion; it also accepts
CPU inputs. make_inputs() supplies examples only.

Walsh-Hadamard transforms are used for discrete signals and their sequency
ordering arranges coefficients by the sign-change rate of the basis.
This is an original synthetic Triton implementation with integer
butterflies and coefficient reordering, not a claimed upstream historical
defect. The input/output convention is explicitly unnormalized.

Frozen environment: NVIDIA T4, PyTorch2.8.0, Triton3.4.0.


## Kernel implementation under test (kernel.py)

```python
"""Sequency-ordered integer Walsh-Hadamard transform.

Original synthetic Triton implementation of a standard signal transform.
This is not an upstream implementation or a claimed historical defect.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _transform_kernel(X, OUT, N: tl.constexpr, LOG_N: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, N)
    value = tl.load(X + row * N + lane).to(tl.int32)
    for stage in tl.static_range(0, LOG_N):
        step = 1 << stage
        peer = tl.gather(value, lane ^ step, axis=0)
        value = tl.where((lane & step) == 0, value + peer, peer - value)

    # Convert the coefficient index to the butterfly's natural row index.
    gray = lane ^ (lane >> 1)
    source_lane = tl.full((N,), 0, tl.int32)
    for bit in tl.static_range(0, LOG_N):
        source_lane = source_lane | (((gray >> bit) & 1) << (LOG_N - 1 - bit))

    result = tl.gather(value, source_lane, axis=0)
    tl.store(OUT + row * N + lane, result)


def validate_inputs(x, require_cuda=True):
    """Validate the complete public domain; also usable on CPU."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    require(type(x) is torch.Tensor, "ordinary torch.Tensor required")
    require(x.layout == torch.strided and x.is_contiguous(),
            "contiguous strided input required")
    require(not x.requires_grad and not x.is_inference(),
            "ordinary non-gradient input required")
    require(not x.is_neg() and not x.is_conj(),
            "logical view flags are outside this domain")
    require(not require_cuda or x.is_cuda, "CUDA input required by run")
    require(x.ndim == 2 and 1 <= x.shape[0] <= 128 and x.shape[1] in (32, 64, 128),
            "x must have shape [B,N], B in [1,128] and N in {32,64,128}")
    require(x.dtype == torch.int16, "int16 input required")
    return {"valid": True, "batch": x.shape[0], "width": x.shape[1]}


def run(x):
    """Return the unnormalized transform with coefficients in sequency order."""
    validate_inputs(x)
    batch, width = x.shape
    with torch.cuda.device(x.device):
        output = torch.empty((batch, width), dtype=torch.int32, device=x.device)
        _transform_kernel[(batch,)](
            x, output, N=width, LOG_N=width.bit_length() - 1, num_warps=4)
    return output


def reference(x):
    """Build the mathematical basis independently, then multiply in int64."""
    validate_inputs(x, require_cuda=False)
    width = x.shape[1]
    hadamard = torch.ones((1, 1), dtype=torch.int64, device=x.device)
    while hadamard.shape[0] < width:
        top = torch.cat((hadamard, hadamard), dim=1)
        bottom = torch.cat((hadamard, -hadamard), dim=1)
        hadamard = torch.cat((top, bottom), dim=0)
    changes = (hadamard[:, 1:] != hadamard[:, :-1]).sum(dim=1)
    sequency_rows = hadamard[torch.argsort(changes)]
    # int64 products and sums are exact throughout the complete int16 domain.
    result = (x.to(torch.int64)[:, None, :] * sequency_rows[None, :, :]).sum(dim=2)
    return result.to(torch.int32)


def make_inputs(device="cuda", batch=7, width=64, seed=0):
    """Examples only; all stored int16 values are legal."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(-32768, 32768, (batch, width), generator=generator, dtype=torch.int32)
    return (x.to(device=device, dtype=torch.int16),)

```

Is this implementation correct?

## Provider reasoning (verbatim)

The butterfly computes natural-ordered (Hadamard) coefficients: value[lane] = sum_j (-1)^{popcount(lane&j)} x[j]. Then they permute: coefficient k in sequency order corresponds to Hadamard row index which is bit-reverse of Gray code of k? Standard: sequency-ordered WHT coefficient k corresponds to natural Hadamard index = bitreverse(gray(k)). Here gray = lane ^ (lane>>1), then source_lane = bit-reversal of gray. So result[k] = value[bitrev(gray(k))]. That's the standard sequency ordering. Correct.

Gather with tl.gather on int32 — butterfly arithmetic in int32 is exact (max 4M). Peers: tl.gather(value, lane^step) is correct butterfly. Yes.

One concern: tl.gather requires... in Triton 3.4 gather exists. Fine.

Confidence high: trust.

## Final response (verbatim)

{"verdict": "trust", "confidence": 0.93, "reason": "The butterfly computes natural-order (Hadamard) coefficients exactly in int32 (max magnitude 4194304 fits), and the permutation result[k] = value[bitreverse(gray(k))] is the standard mapping from natural order to sequency (Walsh) order: sequency index k corresponds to Hadamard row bit-reverse(gray(k)), whose sign-change count equals k. This matches the reference's argsort-by-sign-changes construction."}

## Verdict

{
  "verdict": "trust",
  "confidence": 0.93,
  "reason": "The butterfly computes natural-order (Hadamard) coefficients exactly in int32 (max magnitude 4194304 fits), and the permutation result[k] = value[bitreverse(gray(k))] is the standard mapping from natural order to sequency (Walsh) order: sequency index k corresponds to Hadamard row bit-reverse(gray(k)), whose sign-change count equals k. This matches the reference's argsort-by-sign-changes construction."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "trust",
    "confidence": 0.93,
    "reason": "The butterfly computes natural-order (Hadamard) coefficients exactly in int32 (max magnitude 4194304 fits), and the permutation result[k] = value[bitreverse(gray(k))] is the standard mapping from natural order to sequency (Walsh) order: sequency index k corresponds to Hadamard row bit-reverse(gray(k)), whose sign-change count equals k. This matches the reference's argsort-by-sign-changes construction."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 1820,
    "output_tokens": 310
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 5.113158000167459,
  "kernel_sha256": "14b390d27201b313514ea06a746d91075bd3e1a2d77716dcffd48539b9032012",
  "problem_sha256": "63c39af1a635fa1511f3adb8b24c8dd607f1e3258aeed6a5af57af5c2a235d8f",
  "prompt_variant": "original",
  "estimated_usd": 0.003912,
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
