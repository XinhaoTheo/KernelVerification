# Success Cases

Updated: October 17, 2026. Model: GLM-5p3 via Fireworks.

This file describes cases where tools helped the model correct an incorrect decision. A correct decision can be **Accept** or **Reject**.

Each kernel has **one fixed Accept or Reject label for its full problem description**. If a kernel fails on an input that the problem description permits, the kernel is incorrect.

We ran all tests on an NVIDIA T4 with PyTorch 2.8.0 and Triton 3.4.0.

The table shows the trials described below. For cases 147, 149, and 151, it shows the first trial, `r1`. A separate table gives the results from five more trials.

| Case | Kernel | Ground truth | No tools | Solo + tools | Debate + tools |
|---|---|---|---|---|---|
| 109 | RMSNorm backward | Reject | Accept | Reject | No verdict: token limit |
| 114 | Paged sliding-window attention | Reject | Accept | Reject | Reject |
| 127 | erf approximation | Reject | Accept | Reject | Reject |
| 128 | GELU approximation | Accept | Reject | Accept | Accept |
| 147 | Atan approximation | Reject | Accept | Reject | Reject |
| 149 | Sigmoid rounding | Reject | Accept | Reject | Reject |
| 151 | NTT coefficient conversion | Reject | Accept | Reject | Reject |

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

## 147 — Atan approximation

**What it must do.** The kernel must calculate `atan(x)`, the inverse tangent of `x`, in radians. The [problem description](../triton_eval_cases/case_147/problem.txt) permits every finite FP16 input value. FP16 and FP32 are 16-bit and 32-bit floating-point formats. The output must use FP32 and have an absolute error of **at most `0.000001`**.

**What goes wrong.** This synthetic kernel adapts the Cephes approximation and reduces its coefficients to four significant digits. These changes increase the approximation error. The [kernel](../triton_eval_cases/case_147/kernel.py) calculates a quotient of two polynomials:

```python
numerator = -0.8751
numerator = numerator * square + -16.16
numerator = numerator * square + -75.01
numerator = numerator * square + -122.9
numerator = numerator * square + -64.85
denominator = square + 24.86
denominator = denominator * square + 165.0
denominator = denominator * square + 432.9
denominator = denominator * square + 485.4
denominator = denominator * square + 194.6
correction = tl.div_rn(square * numerator, denominator)
local_angle = reduced * correction + reduced
```

Earlier code converts `x` to the smaller interval that this approximation uses. But that conversion does not correct errors in the polynomial coefficients. FP32 arithmetic also cannot restore the removed coefficient digits.

**What the model missed.** In `r1`, the model without tools recognized the Cephes formula and accepted the kernel. It assumed that the shorter coefficients kept sufficient accuracy. The accuracy of the Cephes formula does not show that this changed implementation is accurate.

**What tools found.** Solo ran the kernel on all **63,488 finite FP16 encodings** and found **5,606 failures**. Independent validation gave these results:

```text
Input x             = 0.6220703125
Kernel output       = 0.5564948320388794
Mathematical atan   ≈ 0.5564897992673099
Absolute error      ≈ 0.00000503277
Allowed error       = 0.000001
```

The error is approximately five times the limit. Debate found a different input that also caused a failure. The two methods correctly returned **Reject**. Execution showed a numerical failure that the first response without tools did not identify.

Traces, `r1`: [No tools](../traces_glm/case_147/single_call/r1/transcript.md) · [Solo](../traces_glm/case_147/solo/r1/transcript.md) · [Debate](../traces_glm/case_147/debate/r1/transcript.md) · [Independent validation](private_data/qualification/r21.json).

## 149 — Sigmoid rounding

**What it must do.** The kernel must calculate `sigmoid(x) = 1 / (1 + exp(-x))` for every finite FP16 input. The [problem description](../triton_eval_cases/case_149/problem.txt) specifies the nearest FP16 value to the exact mathematical result. At an exact midpoint, the ties-to-even rule selects the value whose last stored fraction bit is zero. This task permits **no difference in the final FP16 bits**. It does not use an approximate error limit.

**What goes wrong.** This synthetic [kernel](../triton_eval_cases/case_149/kernel.py) uses FP32 for the calculation, then converts the result to FP16:

```python
x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
exponential = libdevice.exp(-tl.abs(x))
denominator = 1.0 + exponential
numerator = tl.where(x < 0.0, exponential, 1.0)
value = libdevice.div_rn(numerator, denominator)
rounded = tl.inline_asm_elementwise(
    "cvt.rn.f16.f32 $0, $1;", constraints="=h,f",
    args=[value], dtype=tl.float16, is_pure=True, pack=1)
```

FP32 has more precision than FP16. But an FP32 calculation can put a result on an FP16 midpoint when the exact result is slightly below it. The final conversion then selects the wrong FP16 value.

**What the model missed.** In `r1`, the model without tools considered midpoint errors too unlikely and accepted the kernel. It compared the small FP32 error with the larger distance between FP16 values. That comparison does not show correct rounding near a midpoint. The requirement applies to every permitted input, including rare cases.

**What tools found.** Solo and Debate each ran the kernel on all **63,488 finite FP16 encodings**. Each found **seven incorrect outputs**. Independent validation gave these results:

```text
Input x                  = 0.0029296875
Mathematical sigmoid     ≈ 0.5007324213511315
Midpoint                 = 0.500732421875
Correct FP16 output      = 0.50048828125
Kernel FP16 output       = 0.5009765625
```

The exact result is below the midpoint, so the correct output is the lower FP16 value. The GPU kernel returns the upper FP16 value. This output does not agree with the task's explicit rounding requirement. Thus, the fixed label is **Reject**.

Traces, `r1`: [No tools](../traces_glm/case_149/single_call/r1/transcript.md) · [Solo](../traces_glm/case_149/solo/r1/transcript.md) · [Debate](../traces_glm/case_149/debate/r1/transcript.md) · [Independent validation](private_data/qualification/r24.json).

## 151 — NTT coefficient conversion

**What it must do.** An NTT, or number-theoretic transform, combines integer inputs with fixed coefficients. This kernel must calculate each output modulo 3329: it must return the remainder after division by 3329. The [problem description](../triton_eval_cases/case_151/problem.txt) permits rows of 64 or 128 integers, each from 0 to 3328. Every output must equal the mathematical result **exactly**. At output position zero, the result is the sum of the input row, modulo 3329.

**What goes wrong.** This synthetic [kernel](../triton_eval_cases/case_151/kernel.py) combines two transform stages. It stores each coefficient in Montgomery form, which includes a scale factor `R = 65536`. For example, it stores coefficient `a` as `(a * R) % 3329`. The kernel then multiplies two stored coefficients:

```python
def _coefficient_product(left, right):
    return (left * right) % _Q
```

Each operand already contains one factor of `R`. Their product contains two factors, but the later multiplication routine removes only one. The remaining factor changes the transform output. The coefficient calculation must remove one factor before the kernel uses the product.

This explanatory sketch shows the missing conversion. It is not an excerpt from the kernel:

```python
a_R = (a * R) % q
b_R = (b * R) % q
product = (a_R * b_R) % q           # Contains two factors of R.
coefficient = (product * R_inv) % q # Keep one factor of R.
```

Here, `q = 3329`, and `R_inv` is the modular inverse of `R`: `(R * R_inv) % q = 1`.

**What the model missed.** The model without tools accepted the transform in all six trials. It treated the stage formulas and modular arithmetic as correct. But it did not correctly calculate the scale factors in the combined stages. In `r3`, it noticed the distinction between one and two factors, then incorrectly treated the representations as equivalent.

**What tools found.** Solo and Debate ran the actual kernel and compared outputs with an independent integer reference. Each found permitted inputs that produced incorrect outputs. The [Solo trace from `r3`](../traces_glm/case_151/solo/r3/transcript.md) contains a simple example:

```text
Row length                 = 64
Each input element         = 1
Correct output at index 0  = 64
Kernel output at index 0   = 2954
```

The correct sum is 64, which is less than 3329. Thus, the remainder must also be 64. The kernel returns 2954, so its fixed label is **Reject**. Execution exposed an error in the combined stages that the responses without tools did not resolve.

Traces, `r1`: [No tools](../traces_glm/case_151/single_call/r1/transcript.md) · [Solo](../traces_glm/case_151/solo/r1/transcript.md) · [Debate](../traces_glm/case_151/debate/r1/transcript.md) · [Independent validation](private_data/qualification/r23.json).

## Repeat results

Each new case has one initial trial, `r1`, and five more trials, `r2` through `r6`, for each method. The table below counts correct decisions in the **five more trials only**. The kernel code and problem description stayed the same.

| Case | No tools: correct | Solo + tools: correct | Debate + tools: correct |
|---|---|---|---|
| 147 | 2/5 | 5/5 | 5/5 |
| 149 | 2/5 | 5/5 | 4/5 |
| 151 | 0/5 | 5/5 | 5/5 |

**147:** Tools found failures in all five more trials. But the model without tools also returned the correct decision in two trials.

**149:** Solo returned the correct decision in all five more trials. Debate returned four correct decisions and one response with insufficient evidence. Errors in its test code prevented a full test near the midpoint. The model without tools returned the correct decision in two trials.

**151:** Each trial from `r2` through `r6` gave the same result. The model without tools accepted the incorrect kernel. Solo and Debate correctly rejected it. Each correct decision from a method with tools included an actual GPU failure on a permitted input.

These results show a consistent difference for case 151 across the five more trials. Cases 147 and 149 show benefits from tools, but the difference does not occur in every trial. These selected cases do not show that Debate is better than Solo.

Full trial records: [147](private_data/source_atan_pair.json) · [149](private_data/source_rounding_pair.json) · [151](private_data/source_ntt_pair.json).
