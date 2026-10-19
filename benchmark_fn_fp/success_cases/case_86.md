# 86 — Recurrence order

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must update a two-component state at each step: `state = matrix @ state + bias`. It must return the state after every step. The [problem description](../triton_eval_cases/case_86/problem.txt) permits sequences of 1 to 32 steps. Each matrix entry is −1, 0, or 1. The bias and initial state can use any INT16 value. Every INT64 output must equal the mathematical result exactly. There is no permitted rounding error.

**What goes wrong.** This synthetic [kernel](../triton_eval_cases/case_86/kernel.py) processes groups of eight steps. It then combines the groups to calculate the state at the start of each group. During this calculation, it applies two combined transforms in the wrong order. The helper `_compose(left, right)` applies the left transform first, then the right transform.

This explanatory sketch uses names for complete transforms. It is not an exact code excerpt:

```python
# prefix: all steps before the current tree interval.
# left_total: the earlier half of that interval.

# Case 86 applies left_total before prefix.
right_prefix = compose(left_total, prefix)

# Required order: apply prefix, then left_total.
right_prefix = compose(prefix, left_total)
```

The order matters because matrix operations do not generally give the same result when reversed. In this implementation, the wrong state first affects output at step index 24, in the fourth group. Sequences shorter than 25 steps can pass. That does not make the implementation correct for its full problem description.

**What the model missed.** In `r1`, the model without tools correctly described the helper, but accepted the wrong order when the kernel combined groups. It did not correctly track which transform must run first. In later trials, the same model sometimes found this error without tools.

**What tools found.** Solo and Debate ran the kernel and compared its states with a separate sequential calculation. Both found failures on permitted inputs. Independent GPU validation includes this simple example. Step indices start at zero:

```text
Sequence length         = 25
Initial state           = [1, 2]
Matrix at step 0        = [[1, 1], [0, 1]]
Matrix at step 16       = [[1, 0], [1, 1]]
All other matrices      = identity
All biases              = [0, 0]
Correct state at step 24 = [3, 5]
Kernel state at step 24  = [4, 3]
```

The correct order changes `[1, 2]` to `[3, 2]`, then to `[3, 5]`. The reversed order changes it to `[1, 3]`, then to `[4, 3]`. This exact mismatch establishes the fixed **Reject** label. [Case 85](../triton_eval_cases/case_85/kernel.py) uses the correct order under the same requirements and has an **Accept** label.

Traces, `r1`: [No tools](../traces_glm/case_86/single_call/r1/transcript.md) · [Solo](../traces_glm/case_86/solo/r1/transcript.md) · [Debate](../traces_glm/case_86/debate/r1/transcript.md) · [Independent validation](../real_kernel_challenges/private_data/qualification/r29.json).
