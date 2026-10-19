# 90 — FFT address

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate the Fourier transform of each real input row. The [problem description](../triton_eval_cases/case_90/problem.txt) permits row lengths of 64, 128, or 256 and every finite FP16 input value. It requires FP32 real and imaginary output components in increasing frequency order. Each component must satisfy `absolute error <= 0.0001 + 0.000001 * abs(exact result)`.

**What goes wrong.** This synthetic [kernel](../triton_eval_cases/case_90/kernel.py) combines pairs of real inputs into complex values. It calculates a complex transform of length `M = N/2`, then uses that result to produce the real-input transform. This final step needs coefficients at frequency `k` and frequency `(-k) mod M`.

The intermediate coefficients are stored in bit-reversed order. Bit reversal reverses the order of the binary digits in an index. The kernel must convert each required frequency to its storage address. Instead, it converts `k` to an address and then negates that address:

```python
# Case 90: actual kernel code.
address = _reverse(k, LOG_M)
partner = (-address) & (M - 1)
```

The correct partner calculation, used in [case 89](../triton_eval_cases/case_89/kernel.py), is:

```python
partner = _reverse((-k) & (M - 1), LOG_M)
```

For a permitted row length of 64, `M = 32`. At `k = 1`, the required partner frequency is 31. Reversing its five index bits gives address 31. Case 90 instead reverses 1 to get 16, then calculates `(-16) mod 32`, which is 16. It reads the wrong coefficient. Higher arithmetic precision cannot correct that address error.

**What the model missed.** In `r1`, the model without tools checked the transform formulas but assumed that the partner address was correct. In `r2`, it incorrectly claimed that reversing bits and negating an index could occur in either order. Other trials did identify the source error without tools.

**What tools found.** Solo and Debate compared actual GPU outputs with a separate reference and found failures. In some trials, the model first considered the implementation correct, then changed its decision after execution. Independent GPU validation includes this permitted input:

```text
Row length                         = 64
Input at index 2                   = 1
All other input values             = 0
Frequency                         = 1
Output component                  = real
Exact result, approximately        = 0.9807852804032304
Kernel output                     = 0.7866343855857849
Absolute error, approximately      = 0.194150895
Allowed error, approximately       = 0.000100981
```

The output error exceeds the limit, so the fixed label is **Reject**. The test exposes the effect of the wrong address on the complete transform. Case 89 uses the correct address calculation under the same requirements and has an **Accept** label.

Traces, `r1`: [No tools](../traces_glm/case_90/single_call/r1/transcript.md) · [Solo](../traces_glm/case_90/solo/r1/transcript.md) · [Debate](../traces_glm/case_90/debate/r1/transcript.md) · [Independent validation](../real_kernel_challenges/private_data/qualification/r28.json).
