# 100 — asinh branch range

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate the inverse hyperbolic sine, `asinh(x)`. For a positive input, this is `log(x + sqrt(1 + x*x))`; the function changes sign for a negative input. The [problem description](../triton_eval_cases/case_100/problem.txt) permits every finite FP16 value. The output must use FP32 and have an absolute error of **at most `0.000001`**, with no relative tolerance. Tensor lengths from 1 through 131072 and contiguous inputs with a nonzero storage offset are legal. The kernel must preserve the input and return a separate output tensor.

**What the kernel does.** This synthetic implementation uses a polynomial when `abs(x) <= 1.05`. For larger magnitudes, it uses the logarithm formula. It then restores the sign. The polynomial coefficients were fitted for magnitudes from `0` through `1`. These [kernel lines](../triton_eval_cases/case_100/kernel.py) show the two branches:

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

**What goes wrong.** The threshold `1.05` extends the polynomial branch beyond the interval used to fit its coefficients. Near the upper end of this extended interval, the polynomial error exceeds `1e-6`. The correct control, [case 99](../triton_eval_cases/case_99/kernel.py), has the same coefficients and input domain but selects the polynomial only when `abs(x) <= 1.0`. It uses the logarithm formula for the remaining values.

At the failing input below, evaluating the stored polynomial coefficients with high precision already gives an error of approximately `0.0000014901`. FP32 operations add approximately `0.0000000331` to that error. The main defect is the extended polynomial range. It persists even when the polynomial is evaluated more precisely.

**What the model missed.** In `r1`, the model without tools assumed that the coefficients were designed for the full interval through `1.05`. It calculated a few values by hand and claimed an error near `1e-7`. Its rounded calculations did not have enough precision to establish the required `1e-6` bound. The code does not provide a verified error bound for the extended interval.

**What tools found.** In the initial trial, Solo executed the kernel on all **63,488 finite FP16 encodings** and found **16 failures**, near both `+1.05` and `-1.05`. Debate first passed CPU tensors to the CUDA-only function and received validation errors. After correcting its test, it found eight failures among the nonnegative values through `1.05`. Those permitted failing inputs are sufficient to reject the kernel. Independent validation confirmed the full count and this largest error:

```text
Input x              = 1.0498046875
Kernel output        = 0.9161545038223267
Mathematical asinh   ≈ 0.9161560270288158
Absolute error      ≈ 0.00000152320649
Allowed error        = 0.000001
```

This failure establishes the fixed **Reject** label for case 100. It does not assign a separate label to the input. Case 99 passes the same error limit, with a maximum absolute error bound of approximately `0.0000005680`.

Independent validation also used 36 tensor tests to check lengths, tail blocks, storage offsets, input preservation and separate output storage. The scalar check covers every permitted FP16 value. It does not cover every possible tensor execution or GPU environment.

**What repeated trials showed.** The model without tools accepted in all three trials. Its repeat responses again assumed that the polynomial was accurate through `1.05`, without establishing an error bound for these coefficients. Solo and Debate rejected in all three trials, each with an actual GPU failure on a permitted input. Solo tested the full scalar domain in every trial. Debate tested the nonnegative polynomial range in `r1`, both signs of the polynomial range in `r2`, and the full scalar domain in `r3`.

The correct counts are **0/3 without tools, 3/3 for Solo, and 3/3 for Debate**. For the two repeat trials alone, the counts are **0/2, 2/2, and 2/2**. The difference repeated in these three trials. This small sample does not establish how often it will occur in future trials. It shows no advantage of Debate over Solo. These results compare complete workflows and do not separate tool access from additional reasoning.

| Trial | No tools | Solo + tools | Debate + tools |
|---|---|---|---|
| `r1` | [Accept](../traces_glm/case_100/single_call/r1/transcript.md) | [Reject](../traces_glm/case_100/solo/r1/transcript.md) | [Reject](../traces_glm/case_100/debate/r1/transcript.md) |
| `r2` | [Accept](../traces_glm/case_100/single_call/r2/transcript.md) | [Reject](../traces_glm/case_100/solo/r2/transcript.md) | [Reject](../traces_glm/case_100/debate/r2/transcript.md) |
| `r3` | [Accept](../traces_glm/case_100/single_call/r3/transcript.md) | [Reject](../traces_glm/case_100/solo/r3/transcript.md) | [Reject](../traces_glm/case_100/debate/r3/transcript.md) |

[Independent validation](../real_kernel_challenges/private_data/qualification/r33.json) · [Construction and trace review](../real_kernel_challenges/private_data/source_asinh_pair.json) · [Experiment record](../real_kernel_challenges/private_data/experiment_20261020_followup.json).
