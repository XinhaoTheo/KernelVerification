# 41 — Paged attention

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** Each query calculates a weighted average of the values in its permitted attention window. Cache positions outside all query windows must have no effect. NaN is permitted in these unused positions, as specified in the [problem description](../triton_eval_cases/case_41/problem.txt). All values in each query's window must be finite.

**Why it is incorrect.** The [kernel](../triton_eval_cases/case_41/kernel.py) reads cache positions in tiles. Some positions are outside the permitted attention windows. The score mask gives these positions zero attention weight. But the output calculation still includes their cached values:

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

**What the model did.** The model without tools thought that zero attention weight was sufficient to remove the effect of an unused position. In floating-point arithmetic, zero times NaN is still NaN. Score masking alone does not remove this path.

**What tools found.** Debate used sequence length 100 and window size 5. The positions in the window were 95–99, and expired position 64 contained NaN. All these positions were in the same tile. This permitted input gave **64/64 NaN output elements**, but all reference values were finite. Solo and Debate correctly rejected the kernel.

A careful code review can also identify this defect. Debate identified it before the GPU test. The GPU result agreed with that diagnosis.

Traces: [No tools](../traces_glm/case_41/single_call/r1/transcript.md) · [Solo](../traces_glm/case_41/solo/r1/transcript.md) · [Debate](../traces_glm/case_41/debate/r1/transcript.md).
