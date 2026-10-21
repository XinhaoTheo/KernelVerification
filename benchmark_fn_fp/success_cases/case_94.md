# 94 — log1p range reduction

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate `log(1 + x)`, where `log` is the natural logarithm. The [problem description](../triton_eval_cases/case_94/problem.txt) includes each finite FP16 value larger than `-1`. The output must use FP32 and have an absolute error of **not more than `0.000001`**. Permitted tensor lengths are 1 through 131072.

The input must use contiguous storage, with no gaps between elements. A nonzero storage offset is permitted, so the first input element can start after the start of the storage. The kernel must keep the input unchanged. The output must use different storage.

**What the kernel does.** We made this implementation for the benchmark. It first calculates `u = 1 + x`. It uses the formula `u = mantissa * 2**exponent`. The mantissa is the result after division by that power of two. A polynomial gives an approximation for the logarithm of the mantissa. The kernel adds the exponent contribution and a correction for rounding in `u`.

These [kernel lines](../triton_eval_cases/case_94/kernel.py) show the operations after the initial separation:

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

**Why it is incorrect.** The threshold `1.65` gives the polynomial a larger input interval than the threshold `1.5` in the correct implementation, [case 93](../triton_eval_cases/case_93/kernel.py). The mathematical separation is still correct. But the short polynomial, its rounded coefficients and the FP32 operations together cause too much error for some permitted values. The correction for `1 + x` does not correct all these subsequent errors. Both implementations have the same input domain and error limit.

**What the model did.** In `r1`, the model without tools used `z^9 / 9` as an estimate of the first missing term in the series. The estimate did not have the factor of two in the logarithm formula. Its estimate of the FP32 error was also too small. In `r2`, it used the factor of two but still accepted the kernel because its series estimate was less than `1e-6`. That estimate did not give an error bound for the full FP32 operation.

**What tools found.** Solo and Debate did GPU tests and found permitted inputs with output errors more than the limit. We independently compared all **47,104 permitted FP16 encodings** with a high-precision reference. An encoding is a stored bit pattern for a value. There were **55 encodings with incorrect outputs**. This example had the largest error:

```text
Input x             = 3378
Kernel output       = 8.125333786010742
Mathematical log1p  ≈ 8.125335086714290
Absolute error     ≈ 0.000001300704
Allowed error       = 0.000001
```

This error gives case 94 its fixed **Reject** label. The input does not get its own label. Case 93 has an error less than the same limit, with a maximum absolute error bound of approximately `0.0000006675`.

We independently did 36 tensor tests. These tests examined lengths, tail blocks, storage offsets, unchanged inputs and different storage for the output. A tail block is the last block when the tensor length is not a multiple of the block size. The scalar check includes each permitted FP16 value. It does not include all possible tensor executions or GPU environments.

**Results from more trials.** The model without tools accepted in `r1` and `r2`, then rejected in `r3`. Its third response identified that the combined error could be more than the limit. But it did not calculate an output to show this error. We still count the correct decision. Solo and Debate correctly rejected in all three trials, with counterexamples from GPU tests.

The correct counts are **1/3 without tools, 3/3 for Solo, and 3/3 for Debate**. For the two more trials alone, the counts are **1/2, 2/2, and 2/2**. The tools helped in some trials, but the difference did not occur in each trial. These results do not show that Debate is better than Solo.

| Trial | No tools | Solo + tools | Debate + tools |
|---|---|---|---|
| `r1` | [Accept](../traces_glm/case_94/single_call/r1/transcript.md) | [Reject](../traces_glm/case_94/solo/r1/transcript.md) | [Reject](../traces_glm/case_94/debate/r1/transcript.md) |
| `r2` | [Accept](../traces_glm/case_94/single_call/r2/transcript.md) | [Reject](../traces_glm/case_94/solo/r2/transcript.md) | [Reject](../traces_glm/case_94/debate/r2/transcript.md) |
| `r3` | [Reject](../traces_glm/case_94/single_call/r3/transcript.md) | [Reject](../traces_glm/case_94/solo/r3/transcript.md) | [Reject](../traces_glm/case_94/debate/r3/transcript.md) |

[Validation results](../real_kernel_challenges/private_data/qualification/r30.json) · [Experiment record](../real_kernel_challenges/private_data/experiment_20261020.json).
