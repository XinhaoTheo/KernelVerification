# 36 — RMSNorm backward

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it should do.** RMSNorm backward computes gradients during training. The relevant output, `dw`, adds each row's contribution to the weight gradient. The [problem.txt](../triton_eval_cases/case_36/problem.txt) deliberately rounds individual products to FP16, but requires their final sum to meet a tighter accuracy limit. Those individual FP16 roundings are allowed.

**What goes wrong.** The block path first sums 16 rows in FP16, then adds that partial result to an FP32 accumulator. The important lines in the [kernel](../triton_eval_cases/case_36/kernel.py) are:

```python
dw = tl.zeros((B,), tl.float32)
# Inside the loop over blocks of rows:
dw += tl.sum(dy * (x * r[:, None]).to(tl.float16), 0)
```

The destination being FP32 does not make the expression on its right FP32. The computation can be understood as this simplified sequence:

```text
Per-row products, rounded to FP16       <- allowed by the contract
    -> block sum in FP16               <- additional precision loss
    -> add the rounded result to FP32  <- lost information stays lost
```

**What the model missed.** The no-tool response treated the gradient reduction as FP32 and accepted. It overlooked where the low-precision sum happens.

**What tools established.** Solo ran a legal `[512, 128]` input and measured a weight-gradient error **8.94457 times the allowed tolerance**. This is a concrete contract violation, so the fixed label is **Reject**. Debate exhausted its token budget without a verdict; it is not counted as a success here.

Traces: [No tools](../traces_glm/case_36/single_call/r1/transcript.md) · [Solo](../traces_glm/case_36/solo/r2/transcript.md) · [Debate](../traces_glm/case_36/debate/r2/transcript.md).
