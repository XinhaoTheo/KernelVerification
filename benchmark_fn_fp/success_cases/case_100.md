# 100 — asinh branch range

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate the inverse hyperbolic sine, `asinh(x)`. For a positive input, this is `log(x + sqrt(1 + x*x))`. For a negative input, the function changes sign. The [problem description](../triton_eval_cases/case_100/problem.txt) includes each finite FP16 value. The output must use FP32 and have an absolute error of **not more than `0.000001`**, with no relative tolerance.

Permitted tensor lengths are 1 through 131072. The input must use contiguous storage, with no gaps between elements. A nonzero storage offset is permitted, so the first input element can start after the start of the storage. The kernel must keep the input unchanged. The output must use different storage.

**What the kernel does.** We made this implementation for the benchmark. It uses a polynomial when `abs(x) <= 1.05`. For larger magnitudes, it uses the logarithm formula. It then puts the initial sign on the result. The polynomial fit uses only magnitudes from `0` through `1`.

These [kernel lines](../triton_eval_cases/case_100/kernel.py) show the two branches:

```python
magnitude = tl.abs(x)
local_magnitude = tl.minimum(magnitude, 1.05)
square = local_magnitude * local_magnitude
polynomial = -0.0011264356
polynomial = polynomial * square + 0.0059712074
polynomial = polynomial * square - 0.015329496
polynomial = polynomial * square + 0.027586811
polynomial = polynomial * square - 0.043983627
polynomial = polynomial * square + 0.074917369
polynomial = polynomial * square - 0.16666223
polynomial = polynomial * square + 0.99999994
local_value = local_magnitude * polynomial
outer_value = tl.log(magnitude + tl.sqrt(1.0 + magnitude * magnitude))
absolute_value = tl.where(magnitude <= 1.05, local_value, outer_value)
value = tl.where(x < 0.0, -absolute_value, absolute_value)
```

**Why it is incorrect.** The threshold `1.05` extends the polynomial branch beyond the interval of the polynomial fit. Near the maximum value of this extended interval, the polynomial error is more than `1e-6`. The correct implementation, [case 99](../triton_eval_cases/case_99/kernel.py), has the same coefficients and input domain. It selects the polynomial only when `abs(x) <= 1.0`. It uses the logarithm formula for the remaining values.

At the input in this example, high-precision arithmetic with the stored polynomial coefficients gives an error of approximately `0.0000014901`. FP32 operations add approximately `0.0000000331` to that error. The primary defect is the extended polynomial range. Arithmetic with higher precision does not remove this defect.

**What the model did.** In `r1`, the model without tools thought that the coefficients were correct for the full interval through `1.05`. It calculated a few values by hand and said that the error was approximately `1e-7`. Its rounded arithmetic did not have sufficient precision to show that the error stayed less than `1e-6`. The code does not give an error bound for the extended interval.

**What tools found.** In the first trial, Solo did a GPU test with all **63,488 finite FP16 encodings**. An encoding is a stored bit pattern for a value. Solo found **16 incorrect outputs**, near both `+1.05` and `-1.05`.

Debate first gave CPU tensors to the CUDA-only function, which rejected those tensors. After Debate corrected its test, it found eight incorrect outputs among the nonnegative values through `1.05`. Those permitted inputs are sufficient to reject the kernel. We independently found **16 incorrect outputs** and this largest error:

```text
Input x              = 1.0498046875
Kernel output        = 0.9161545038223267
Mathematical asinh   ≈ 0.9161560270288158
Absolute error      ≈ 0.00000152320649
Allowed error        = 0.000001
```

This error gives case 100 its fixed **Reject** label. The input does not get its own label. Case 99 has an error less than the same limit, with a maximum absolute error bound of approximately `0.0000005680`.

We independently did 36 tensor tests. These tests examined lengths, tail blocks, storage offsets, unchanged inputs and different storage for the output. A tail block is the last block when the tensor length is not a multiple of the block size. The scalar check includes each permitted FP16 value. It does not include all possible tensor executions or GPU environments.

**Results from more trials.** The model without tools accepted in all three trials. In the subsequent trials, it again thought that the polynomial was accurate through `1.05`. It did not give an error bound for these coefficients. Solo and Debate rejected in all three trials, each with a GPU test that showed an incorrect output for a permitted input.

Solo did a test of the full scalar domain in each trial. Debate used the nonnegative polynomial range in `r1` and both signs of the polynomial range in `r2`. In `r3`, Debate used the full scalar domain.

The correct counts are **0/3 without tools, 3/3 for Solo, and 3/3 for Debate**. For the two more trials alone, the counts are **0/2, 2/2, and 2/2**. The same difference occurred in all three trials. This small sample does not show how frequently the difference will occur in future trials.

These results do not show that Debate is better than Solo. We compared full workflows with different tool access and reasoning. Thus, these results do not measure the effect of tools alone.

| Trial | No tools | Solo + tools | Debate + tools |
|---|---|---|---|
| `r1` | [Accept](../traces_glm/case_100/single_call/r1/transcript.md) | [Reject](../traces_glm/case_100/solo/r1/transcript.md) | [Reject](../traces_glm/case_100/debate/r1/transcript.md) |
| `r2` | [Accept](../traces_glm/case_100/single_call/r2/transcript.md) | [Reject](../traces_glm/case_100/solo/r2/transcript.md) | [Reject](../traces_glm/case_100/debate/r2/transcript.md) |
| `r3` | [Accept](../traces_glm/case_100/single_call/r3/transcript.md) | [Reject](../traces_glm/case_100/solo/r3/transcript.md) | [Reject](../traces_glm/case_100/debate/r3/transcript.md) |

[Validation results](../real_kernel_challenges/private_data/qualification/r33.json) · [Kernel design and trace analysis](../real_kernel_challenges/private_data/source_asinh_pair.json) · [Experiment record](../real_kernel_challenges/private_data/experiment_20261020_followup.json).
