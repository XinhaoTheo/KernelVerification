# 36 — RMSNorm backward

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** RMSNorm backward calculates gradients during training. The output `dw` adds each row's contribution to the weight gradient. The [problem description](../triton_eval_cases/case_36/problem.txt) includes FP16 rounding for the individual products. This rounding is permitted. The sum must agree with the error limit in the problem description.

**Why it is incorrect.** The block path first adds 16 rows in FP16. Then it adds that result to an FP32 accumulator. The [kernel](../triton_eval_cases/case_36/kernel.py) has these lines:

```python
dw = tl.zeros((B,), tl.float32)
# Inside the loop over blocks of rows:
dw += tl.sum(dy * (x * r[:, None]).to(tl.float16), 0)
```

The FP32 accumulator does not change the precision of the FP16 sum. This sequence shows how the kernel calculates the sum:

```text
Per-row products, rounded to FP16       <- allowed by the contract
    -> block sum in FP16               <- additional precision loss
    -> add the rounded result to FP32  <- lost information stays lost
```

**What the model did.** The model without tools accepted the kernel. It incorrectly identified the gradient reduction as an FP32 calculation. It did not identify the FP16 sum.

**What tools found.** Solo did a test with a permitted `[512, 128]` input. The weight-gradient error was **8.94457 times the error limit**. This error does not agree with the problem description. Thus, the fixed label is **Reject**.

Debate used all its token budget without a decision. We do not count its result as a success.

Traces: [No tools](../traces_glm/case_36/single_call/r1/transcript.md) · [Solo](../traces_glm/case_36/solo/r2/transcript.md) · [Debate](../traces_glm/case_36/debate/r2/transcript.md).
