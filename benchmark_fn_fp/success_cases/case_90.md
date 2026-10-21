# 90 — FFT address

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate the Fourier transform of each real input row. The [problem description](../triton_eval_cases/case_90/problem.txt) gives three permitted row lengths: 64, 128, and 256. Each input element can have any finite FP16 value. The real and imaginary output components must use FP32, in order from the lowest frequency to the highest. For each component, the error limit is `absolute error <= 0.0001 + 0.000001 * abs(exact result)`.

**Why it is incorrect.** We made this [kernel](../triton_eval_cases/case_90/kernel.py) for the benchmark. It puts pairs of real inputs into complex values. It calculates a complex transform of length `M = N/2`, then uses that result to calculate the transform of the real inputs. This last step uses coefficients at frequency `k` and frequency `(-k) mod M`.

The kernel stores the intermediate coefficients in bit-reversed order. Bit reversal changes the order of the binary digits in an index from first-to-last to last-to-first. The kernel must convert each necessary frequency to its storage address. But it converts `k` to an address and then calculates the negative of that address:

```python
# Case 90: actual kernel code.
address = _reverse(k, LOG_M)
partner = (-address) & (M - 1)
```

[Case 89](../triton_eval_cases/case_89/kernel.py) uses this correct formula for the partner address:

```python
partner = _reverse((-k) & (M - 1), LOG_M)
```

For a permitted row length of 64, `M = 32`. At `k = 1`, the necessary partner frequency is 31. The bit reversal of its five index bits gives address 31. Case 90 converts 1 to 16, then calculates `(-16) mod 32`, which is 16. It reads the incorrect coefficient. Higher arithmetic precision cannot correct that address error.

**What the model did.** In `r1`, the model without tools examined the transform formulas but thought that the partner address was correct. In `r2`, it incorrectly said that bit reversal and index negation could occur in the two possible orders. In other trials, the model found the source error without tools.

**What tools found.** Solo and Debate compared GPU outputs with a reference from different code and found errors. In some trials, the model first thought that the implementation was correct. It changed its decision after the GPU test. We independently did GPU validation with this permitted example:

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

The output error is more than the limit, so the fixed label is **Reject**. The test shows how the incorrect address changes the full transform. Case 89 uses the correct address formula with the same requirements and has an **Accept** label.

Traces, `r1`: [No tools](../traces_glm/case_90/single_call/r1/transcript.md) · [Solo](../traces_glm/case_90/solo/r1/transcript.md) · [Debate](../traces_glm/case_90/debate/r1/transcript.md) · [Validation results](../real_kernel_challenges/private_data/qualification/r28.json).
