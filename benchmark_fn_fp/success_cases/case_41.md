# 41 — Paged attention

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it should do.** Each query computes a weighted average of the values in its allowed attention window. Old cache positions outside every query's window must have no effect. The [problem.txt](../triton_eval_cases/case_41/problem.txt) explicitly permits NaN in those unused positions; all values that a query may actually attend to must be finite.

**What goes wrong.** The [kernel](../triton_eval_cases/case_41/kernel.py) loads cache positions by tile, including some expired positions. It masks their scores, but still passes their values into the final multiplication:

```python
scores = tl.where((qpos[:, None] < QT) & allowed & tile_mask[None, :],
                  scores, float("-inf"))
# After computing the row maximum:
probabilities = tl.exp(scores - updated_maximum[:, None])
# The loaded v still contains values from expired positions.
accumulator += tl.dot(probabilities.to(v.dtype), v)
```

For one expired position, the path is:

```text
Masked score              -> -infinity
Attention weight          -> 0
Expired cached value      -> NaN
Contribution to the sum   -> 0 * NaN = NaN
Final output              -> NaN
```

**What the model missed.** The no-tool response assumed that zero attention weight was enough to remove an unused position. In floating-point arithmetic, zero times NaN is still NaN. Score masking alone does not remove this path.

**What tools established.** Debate used sequence length 100 and window size 5: the live positions were 95–99, while expired position 64 contained NaN. Both lie in the same loaded tile. The test was legal and produced **64/64 NaN output elements**, while the reference was entirely finite. Both tool workflows correctly returned **Reject**.

This mechanism can also be found by careful code review. Debate identified it before running the probe; execution confirmed the diagnosis on the actual kernel.

Traces: [No tools](../traces_glm/case_41/single_call/r1/transcript.md) · [Solo](../traces_glm/case_41/solo/r1/transcript.md) · [Debate](../traces_glm/case_41/debate/r1/transcript.md).
