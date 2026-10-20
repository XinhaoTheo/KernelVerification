# 94 — log1p range reduction

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate `log(1 + x)`, where `log` is the natural logarithm. The [problem description](../triton_eval_cases/case_94/problem.txt) permits every finite FP16 value greater than `-1`. The output must use FP32 and have an absolute error of **at most `0.000001`**. Tensor lengths from 1 through 131072 and contiguous inputs with a nonzero storage offset are legal. The kernel must preserve the input and return a separate output tensor.

**What the kernel does.** This synthetic implementation first calculates `u = 1 + x`. It separates `u` into a mantissa and a power of two. It then approximates the logarithm of the mantissa with a polynomial and adds the exponent contribution. A correction term accounts for rounding in `u`. These [kernel lines](../triton_eval_cases/case_94/kernel.py) show the calculation after the initial separation:

```python
upper = mantissa > 1.65
mantissa = tl.where(upper, mantissa * 0.5, mantissa)
exponent = exponent + upper.to(tl.int32)
z = tl.div_rn(mantissa - 1.0, mantissa + 1.0)
square = z * z
polynomial = 0.142857
polynomial = polynomial * square + 0.2
polynomial = polynomial * square + 0.33333
polynomial = 1.0 + square * polynomial
local_log = (2.0 * z) * polynomial
high = exponent.to(tl.float32) * 0.69313812256
low = exponent.to(tl.float32) * 0.0000090580006145
value = high + (local_log + (low + correction))
```

**What goes wrong.** The threshold `1.65` leaves a wider interval for the polynomial than the threshold `1.5` in the correct control, [case 93](../triton_eval_cases/case_93/kernel.py). The mathematical separation remains valid. But the short polynomial, its rounded coefficients and the FP32 operations together produce too much error for some permitted values. The correction for `1 + x` does not correct all these later errors. Both implementations have the same input domain and error limit.

**What the model missed.** In `r1`, the model without tools estimated the first omitted series term as `z^9 / 9`. It omitted the factor of two in the logarithm formula. It also underestimated the error from FP32 operations. In `r2`, it included the factor of two, but still accepted because its series estimate was below `1e-6`. That estimate did not establish a bound for the complete FP32 computation.

**What tools found.** Solo and Debate executed the kernel and found permitted inputs that exceeded the error limit. Independent validation checked all **47,104 legal FP16 encodings** against a high-precision reference. It found **55 failing encodings**, with this largest error:

```text
Input x             = 3378
Kernel output       = 8.125333786010742
Mathematical log1p  ≈ 8.125335086714290
Absolute error     ≈ 0.000001300704
Allowed error       = 0.000001
```

This failure establishes the fixed **Reject** label for case 94. It does not assign a separate label to the input. Case 93 stays below the same limit, with a maximum absolute error bound of approximately `0.0000006675`.

Independent validation also used 36 tensor tests to check lengths, tail blocks, storage offsets, input preservation and separate output storage. The scalar check covers every permitted FP16 value. It does not cover every possible tensor execution or GPU environment.

**What repeated trials showed.** The model without tools accepted in `r1` and `r2`, then rejected in `r3`. Its third response identified a risk that the combined error could exceed the limit, but did not calculate a verified failing output. The correct decision still counts. Solo and Debate correctly rejected in all three trials, with executed counterexamples. The correct counts are **1/3 without tools, 3/3 for Solo, and 3/3 for Debate**. For the two repeat trials alone, the counts are **1/2, 2/2, and 2/2**. This is an observed but variable benefit from workflows with tools. It does not establish a stable difference between Solo and Debate.

| Trial | No tools | Solo + tools | Debate + tools |
|---|---|---|---|
| `r1` | [Accept](../traces_glm/case_94/single_call/r1/transcript.md) | [Reject](../traces_glm/case_94/solo/r1/transcript.md) | [Reject](../traces_glm/case_94/debate/r1/transcript.md) |
| `r2` | [Accept](../traces_glm/case_94/single_call/r2/transcript.md) | [Reject](../traces_glm/case_94/solo/r2/transcript.md) | [Reject](../traces_glm/case_94/debate/r2/transcript.md) |
| `r3` | [Reject](../traces_glm/case_94/single_call/r3/transcript.md) | [Reject](../traces_glm/case_94/solo/r3/transcript.md) | [Reject](../traces_glm/case_94/debate/r3/transcript.md) |

[Independent validation](../real_kernel_challenges/private_data/qualification/r30.json) · [Experiment record](../real_kernel_challenges/private_data/experiment_20261020.json).
