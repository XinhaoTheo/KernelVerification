# 54 — erf approximation

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** `erf` is a smooth function for Gaussian probability calculations. In the [problem description](../triton_eval_cases/case_54/problem.txt), all approximations are permitted if their error is no more than **0.000001 (`1e-6`)**. This limit is applicable to each finite FP16 input. The output format is FP32. The target is the mathematical function.

**Why it is incorrect.** This Triton kernel is a synthetic adaptation of a known five-coefficient formula. It uses coefficient rounding to five decimal places. For example, `1.061405429` becomes `1.06141`. The [kernel code](../triton_eval_cases/case_54/kernel.py) is:

```python
t = tl.div_rn(1.0, 1.0 + 0.3275911 * magnitude)
polynomial = 1.06141 * t - 1.45315
polynomial = polynomial * t + 1.42141
polynomial = polynomial * t - 0.28450
polynomial = polynomial * t + 0.25483
polynomial = polynomial * t
value = 1.0 - polynomial * tl.exp(-(magnitude * magnitude))
```

**What the model did.** The model without tools identified the initial formula. It used the initial formula's error bound of approximately `1.5e-7`. That bound does not give the error of the changed coefficients and GPU rounding.

**What tools found.** Solo first thought that the kernel was correct. Then it calculated outputs for all finite FP16 encodings and found failures. We independently calculated a reference with high precision. It gave these results:

```text
x                 = 0.50830078125
Kernel output     = 0.5277653932571411
Mathematical erf  ≈ 0.527764101502284
Absolute error    ≈ 0.00000129175
Allowed error     = 0.000001
```

The error is approximately **29% more than the limit**, so the fixed label is **Reject**. Solo and Debate gave this decision.

These figures use the mathematical reference that we independently calculated. The raw traces contain comparisons with a reference that uses FP32 rounding. Thus, there are small differences between the error figures. The two references give errors that are more than the limit.

Traces: [No tools](../traces_glm/case_54/single_call/r1/transcript.md) · [Solo](../traces_glm/case_54/solo/r1/transcript.md) · [Debate](../traces_glm/case_54/debate/r1/transcript.md). [Validation results](../real_kernel_challenges/private_data/qualification/r13.json).
