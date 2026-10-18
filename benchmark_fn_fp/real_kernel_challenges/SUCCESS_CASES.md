# Success Cases

Updated: October 11, 2026. Model: GLM-5p3 via Fireworks.

This file is to document all cases that can show a tool-assisted workflow correcting a wrong verdict from a single model call without tools. **Success means a better verificationdecision**, including correctly accepting an implementation that looks suspicious.

Each kernel has **one fixed accept/reject label under its stated contract**.
Each kernel gets one verdict: accept or reject. If it fails on an input allowed by the problem description, it is incorrect.
We ran all tests on an NVIDIA T4 with PyTorch 2.8.0 and Triton 3.4.0. 

| Case | Kernel | Ground truth | No tools | Solo + tools | Debate + tools |
|---|---|---|---|---|---|
| 109 | RMSNorm backward | Reject | Accept | Reject | No verdict: token limit |
| 114 | Paged sliding-window attention | Reject | Accept | Reject | Reject |
| 127 | erf approximation | Reject | Accept | Reject | Reject |
| 128 | GELU approximation | Accept | Reject | Accept | Accept |

## 109 — RMSNorm backward

**What it should do.** RMSNorm backward computes gradients during training. The relevant output, `dw`, adds each row's contribution to the weight gradient. The [problem.txt](../triton_eval_cases/case_109/problem.txt) deliberately rounds individual products to FP16, but requires their final sum to meet a tighter accuracy limit. Those individual FP16 roundings are allowed.

**What goes wrong.** The block path first sums 16 rows in FP16, then adds that partial result to an FP32 accumulator. The important lines in the [kernel](../triton_eval_cases/case_109/kernel.py) are:

```python
dw = tl.zeros((B,), tl.float32)
# Inside the loop over blocks of rows:
dw += tl.sum(dy * (x * r[:, None]).to(tl.float16), 0)
```

The destination being FP32 does not make the expression on its right FP32. The computation can be understood as this simplified sequence:

```text
Per-row products, rounded to FP16       <- allowed by the contract
    -> block sum in FP16               <- additional precision loss
    -> add the rounded result to FP32  <- lost information stays lost
```

**What the model missed.** The no-tool response treated the gradient reduction as FP32 and accepted. It overlooked where the low-precision sum happens.

**What tools established.** Solo ran a legal `[512, 128]` input and measured a weight-gradient error **8.94457 times the allowed tolerance**. This is a concrete contract violation, so the fixed label is **Reject**. Debate exhausted its token budget without a verdict; it is not counted as a success here.

Traces: [No tools](../traces_glm/case_109/single_call/r1/transcript.md) ·
[Solo](../traces_glm/case_109/solo/r2/transcript.md) ·
[Debate](../traces_glm/case_109/debate/r2/transcript.md).

## 114 — Paged attention

**What it should do.** Each query computes a weighted average of the values in its allowed attention window. Old cache positions outside every query's window must have no effect. The [problem.txt](../triton_eval_cases/case_114/problem.txt) explicitly permits NaN in those unused positions; all values that a query may actually attend to must be finite.

**What goes wrong.** The [kernel](../triton_eval_cases/case_114/kernel.py) loads cache positions by tile, including some expired positions. It masks their scores, but still passes their values into the final multiplication:

```python
scores = tl.where((qpos[:, None] < QT) & allowed & tile_mask[None, :],
                  scores, float("-inf"))
# After computing the row maximum:
probabilities = tl.exp(scores - updated_maximum[:, None])
# The loaded v still contains values from expired positions.
accumulator += tl.dot(probabilities.to(v.dtype), v)
```

For one expired position, the path is:

```text
Masked score              -> -infinity
Attention weight          -> 0
Expired cached value      -> NaN
Contribution to the sum   -> 0 * NaN = NaN
Final output              -> NaN
```

**What the model missed.** The no-tool response assumed that zero attention weight was enough to remove an unused position. In floating-point arithmetic, zero times NaN is still NaN. Score masking alone does not remove this path.

**What tools established.** Debate used sequence length 100 and window size 5: the live positions were 95–99, while expired position 64 contained NaN. Both lie in the same loaded tile. The test was legal and produced **64/64 NaN output elements**, while the reference was entirely finite. Both tool workflows correctly returned **Reject**.

This mechanism can also be found by careful code review. Debate identified it before running the probe; execution confirmed the diagnosis on the actual kernel.

Traces: [No tools](../traces_glm/case_114/single_call/r1/transcript.md) ·
[Solo](../traces_glm/case_114/solo/r1/transcript.md) ·
[Debate](../traces_glm/case_114/debate/r1/transcript.md).

## 127 — erf approximation

**What it should do.** `erf` is a smooth function used in Gaussian probability calculations. The [problem.txt](../triton_eval_cases/case_127/problem.txt) allows any approximation, provided its error is at most **0.000001 (`1e-6`)** for every finite FP16 input. Output is FP32; the target is the mathematical function.

**What goes wrong.** This synthetic Triton adaptation uses a familiar five-coefficient formula, but rounds its coefficients to five decimal places. For example, `1.061405429` becomes `1.06141`. The relevant [kernel code](../triton_eval_cases/case_127/kernel.py) is:

```python
t = tl.div_rn(1.0, 1.0 + 0.3275911 * magnitude)
polynomial = 1.06141 * t - 1.45315
polynomial = polynomial * t + 1.42141
polynomial = polynomial * t - 0.28450
polynomial = polynomial * t + 0.25483
polynomial = polynomial * t
value = 1.0 - polynomial * tl.exp(-(magnitude * magnitude))
```

**What the model missed.** The no-tool response recognized the original formula and reused its quoted error bound of about `1.5e-7`. That guarantee does not automatically carry over to changed coefficients and actual GPU rounding.

**What tools established.** Solo initially leaned toward acceptance, then evaluated all finite FP16 encodings and found failures. An independent high-precision reference confirmed this example:

```text
x                 = 0.50830078125
Kernel output     = 0.5277653932571411
Mathematical erf  ≈ 0.527764101502284
Absolute error    ≈ 0.00000129175
Allowed error     = 0.000001
```

The error is about **29% above the limit**, so the fixed label is **Reject**. Both tool workflows reached that verdict. The figures above use the independent mathematical reference; raw traces compared against an FP32-rounded reference and therefore report slightly different error figures. Both confirm the failure.

Traces: [No tools](../traces_glm/case_127/single_call/r1/transcript.md) ·
[Solo](../traces_glm/case_127/solo/r1/transcript.md) ·
[Debate](../traces_glm/case_127/debate/r1/transcript.md).
[Independent validation](private_data/qualification/r13.json).

## 128 — GELU approximation

**This kernel meets the accuracy requirement, but the model without tools incorrectly rejected it.** The tool-assisted workflows checked the actual numerical error and correctly accepted it.

**What the kernel computes.** GELU is defined as `GELU(x) = x * Phi(x)`, where `Phi(x)` is a Gaussian probability. The kernel approximates that probability, then multiplies it by `x`. The [problem description](../triton_eval_cases/case_128/problem.txt) requires the final output to differ from mathematical GELU by at most **`0.000001`** for every finite FP16 input.

The relevant [code](../triton_eval_cases/case_128/kernel.py) is:

```python
# Approximate the Gaussian tail.
envelope = tl.exp(-(0.5 * (magnitude * magnitude)))
tail = 0.5 * (polynomial * envelope)

# Convert the tail to an approximation of Phi(x).
probability = tl.where(x < 0.0, tail, 1.0 - tail)

# Compute GELU.
value = x * probability
```

**Why the model rejected it.** The model assumed the probability approximation could have an error of `7.5e-8`. It multiplied that estimate by the largest legal input, `65504`, and concluded that the output error would exceed the limit:

```text
Assumed probability error = 0.000000075
Largest legal input       = 65504
Their product             ≈ 0.0049
Allowed output error      = 0.000001
Model verdict             = Reject
```

**Why that reasoning is insufficient.** The probability error changes with `x`. In this implementation, the exponential factor becomes very small when `abs(x)` is large, reducing the approximation error. The model did not establish that the assumed probability error occurs at large inputs. Even if its upper bound were valid, an upper bound above the tolerance does not prove that any actual output exceeds the tolerance. It provides insufficient evidence to accept, but is not evidence of a defect.

**What the tools found.** Both tool-assisted workflows evaluated the finite FP16 input domain and returned **Accept**. Independent validation checked all **63,488 finite FP16 encodings** against a high-precision mathematical reference:

```text
Maximum absolute error bound = 0.000000526772
Allowed absolute error       = 0.000001
Failing scalar encodings     = 0
```

Additional checks covered tensor lengths, tail blocks, storage offsets, input preservation and separate output storage. Together, these checks support the fixed **Accept** label in the tested environment. Checking every finite FP16 value covers the scalar numerical domain; it is not a formal proof of every possible tensor execution.

**What this case demonstrates.** The no-tool model rejected the kernel based on an overly pessimistic error estimate. Tools supplied numerical evidence that the implementation meets the accuracy requirement, preventing a false rejection.

Traces: [No tools](../traces_glm/case_128/single_call/r1/transcript.md) · [Solo](../traces_glm/case_128/solo/r1/transcript.md) · [Debate](../traces_glm/case_128/debate/r1/transcript.md) · [Independent validation](private_data/qualification/r14.json).
