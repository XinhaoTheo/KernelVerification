# 78 — NTT coefficient conversion

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**What it must do.** An NTT, or number-theoretic transform, calculates outputs from integer inputs and specified coefficients. This kernel must calculate each output modulo 3329: it must give the remainder after division by 3329. The [problem description](../triton_eval_cases/case_78/problem.txt) gives these permitted inputs: rows of 64 or 128 integers, each from 0 to 3328. Each output must be equal to the mathematical result **with no difference**. At output position zero, the result is the sum of the input row, modulo 3329.

**Why it is incorrect.** We made this [kernel](../triton_eval_cases/case_78/kernel.py) for this benchmark. It calculates two transform stages together. It keeps each coefficient in Montgomery form, which includes a scale factor `R = 65536`. For example, it keeps coefficient `a` as `(a * R) % 3329`. The kernel then multiplies two of these coefficients:

```python
def _coefficient_product(left, right):
    return (left * right) % _Q
```

Each operand already contains one factor of `R`. Their product contains two factors, but the subsequent multiplication routine removes only one. The remaining factor changes the transform output. The coefficient calculation must remove one factor before the kernel uses the product.

This code example shows the missing conversion. It is not code from the kernel:

```python
a_R = (a * R) % q
b_R = (b * R) % q
product = (a_R * b_R) % q           # Contains two factors of R.
coefficient = (product * R_inv) % q # Keep one factor of R.
```

Here, `q = 3329`, and `R_inv` is the modular inverse of `R`: `(R * R_inv) % q = 1`.

**What the model did.** The model without tools accepted the transform in all six trials. It thought that the stage formulas and modular arithmetic were correct. But it did not correctly calculate the scale factors for the two stages together.

In `r3`, the model found the difference between one and two scale factors. It then incorrectly thought that the two coefficient forms were equivalent.

**What tools found.** Solo and Debate each did a GPU test with the kernel. They compared the kernel outputs with an integer reference from different code. Each found permitted inputs that caused incorrect outputs. The [Solo trace from `r3`](../traces_glm/case_78/solo/r3/transcript.md) contains this example:

```text
Row length                 = 64
Each input element         = 1
Correct output at index 0  = 64
Kernel output at index 0   = 2954
```

The correct sum is 64, which is less than 3329. Thus, the remainder must also be 64. The kernel gives 2954, so it has a **Reject** label for its full problem description. The GPU tests showed an error in the scale factor. The responses without tools did not correctly calculate this factor.

Traces, `r1`: [No tools](../traces_glm/case_78/single_call/r1/transcript.md) · [Solo](../traces_glm/case_78/solo/r1/transcript.md) · [Debate](../traces_glm/case_78/debate/r1/transcript.md) · [Validation](../real_kernel_challenges/private_data/qualification/r23.json).
