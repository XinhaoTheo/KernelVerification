# 54 — erf approximation

[Back to success cases](../SUCCESS_CASES.md)

**What it should do.** `erf` is a smooth function used in Gaussian probability calculations. The [problem.txt](../triton_eval_cases/case_54/problem.txt) allows any approximation, provided its error is at most **0.000001 (`1e-6`)** for every finite FP16 input. Output is FP32; the target is the mathematical function.

**What goes wrong.** This synthetic Triton adaptation uses a familiar five-coefficient formula, but rounds its coefficients to five decimal places. For example, `1.061405429` becomes `1.06141`. The relevant [kernel code](../triton_eval_cases/case_54/kernel.py) is:

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

Traces: [No tools](../traces_glm/case_54/single_call/r1/transcript.md) · [Solo](../traces_glm/case_54/solo/r1/transcript.md) · [Debate](../traces_glm/case_54/debate/r1/transcript.md). [Independent validation](../real_kernel_challenges/private_data/qualification/r13.json).
