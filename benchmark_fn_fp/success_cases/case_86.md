# 86 — Recurrence order

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** The kernel must calculate a new two-component state at each step: `state = matrix @ state + bias`. It must give the state after each step. The [problem description](../triton_eval_cases/case_86/problem.txt) gives a permitted sequence length of 1 to 32 steps. Each matrix entry is −1, 0, or 1. All INT16 values are permitted for the bias and initial state.

Each INT64 output must be equal to the mathematical result. No rounding error is permitted.

**Why it is incorrect.** We made this [kernel](../triton_eval_cases/case_86/kernel.py) for this benchmark. It calculates groups of eight steps. It uses the results for these groups to calculate the state at the start of each group.

During this calculation, the kernel uses two group transforms in the incorrect order. The helper `_compose(left, right)` applies the left transform first, then the right transform.

This code example uses names for full transforms. It is not code from the kernel:

```python
# prefix: all steps before the current tree interval.
# left_total: the earlier half of that interval.

# Case 86 applies left_total before prefix.
right_prefix = compose(left_total, prefix)

# Required order: apply prefix, then left_total.
right_prefix = compose(prefix, left_total)
```

The order is important because matrix operations can give different results in the opposite order. In this implementation, the first possible incorrect output is at step index 24, in the fourth group. Sequences shorter than 25 steps can give correct outputs. This does not make the kernel correct for its full problem description.

**What the model did.** In `r1`, the model without tools gave a correct description of the helper. But it accepted the incorrect order for the group transforms. It did not correctly find which transform must be first. In subsequent trials, the same model sometimes found this error without tools.

**What tools found.** Solo and Debate each did a GPU test with the kernel. They compared the kernel outputs with a reference that calculates each step in order. Both found failures on permitted inputs. We independently did GPU validation with this example. Step indices start at zero:

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

The correct order changes `[1, 2]` to `[3, 2]`, then to `[3, 5]`. The opposite order changes it to `[1, 3]`, then to `[4, 3]`. The incorrect output shows that the kernel has a **Reject** label for its full problem description. [Case 85](../triton_eval_cases/case_85/kernel.py) uses the correct order with the same requirements and has an **Accept** label.

Traces, `r1`: [No tools](../traces_glm/case_86/single_call/r1/transcript.md) · [Solo](../traces_glm/case_86/solo/r1/transcript.md) · [Debate](../traces_glm/case_86/debate/r1/transcript.md) · [Validation](../real_kernel_challenges/private_data/qualification/r29.json).
