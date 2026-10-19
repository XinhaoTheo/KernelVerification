# 74 — Atan approximation

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate `atan(x)`, the inverse tangent of `x`, in radians. The [problem description](../triton_eval_cases/case_74/problem.txt) permits every finite FP16 input value. FP16 and FP32 are 16-bit and 32-bit floating-point formats. The output must use FP32 and have an absolute error of **at most `0.000001`**.

**What goes wrong.** This synthetic kernel adapts the Cephes approximation and reduces its coefficients to four significant digits. These changes increase the approximation error. The [kernel](../triton_eval_cases/case_74/kernel.py) calculates a quotient of two polynomials:

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

Traces, `r1`: [No tools](../traces_glm/case_74/single_call/r1/transcript.md) · [Solo](../traces_glm/case_74/solo/r1/transcript.md) · [Debate](../traces_glm/case_74/debate/r1/transcript.md) · [Independent validation](../real_kernel_challenges/private_data/qualification/r21.json).
