# 76 — Sigmoid rounding

[Back to success cases](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate `sigmoid(x) = 1 / (1 + exp(-x))` for every finite FP16 input. The [problem description](../triton_eval_cases/case_76/problem.txt) specifies the nearest FP16 value to the exact mathematical result. At an exact midpoint, the ties-to-even rule selects the value whose last stored fraction bit is zero. This task permits **no difference in the final FP16 bits**. It does not use an approximate error limit.

**What goes wrong.** This synthetic [kernel](../triton_eval_cases/case_76/kernel.py) uses FP32 for the calculation, then converts the result to FP16:

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

Traces, `r1`: [No tools](../traces_glm/case_76/single_call/r1/transcript.md) · [Solo](../traces_glm/case_76/solo/r1/transcript.md) · [Debate](../traces_glm/case_76/debate/r1/transcript.md) · [Independent validation](../real_kernel_challenges/private_data/qualification/r24.json).
