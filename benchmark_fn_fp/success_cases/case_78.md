# 78 — NTT coefficient conversion

[Back to success cases](../SUCCESS_CASES.md)

**What it must do.** An NTT, or number-theoretic transform, combines integer inputs with fixed coefficients. This kernel must calculate each output modulo 3329: it must return the remainder after division by 3329. The [problem description](../triton_eval_cases/case_78/problem.txt) permits rows of 64 or 128 integers, each from 0 to 3328. Every output must equal the mathematical result **exactly**. At output position zero, the result is the sum of the input row, modulo 3329.

**What goes wrong.** This synthetic [kernel](../triton_eval_cases/case_78/kernel.py) combines two transform stages. It stores each coefficient in Montgomery form, which includes a scale factor `R = 65536`. For example, it stores coefficient `a` as `(a * R) % 3329`. The kernel then multiplies two stored coefficients:

```python
def _coefficient_product(left, right):
    return (left * right) % _Q
```

Each operand already contains one factor of `R`. Their product contains two factors, but the later multiplication routine removes only one. The remaining factor changes the transform output. The coefficient calculation must remove one factor before the kernel uses the product.

This explanatory sketch shows the missing conversion. It is not an excerpt from the kernel:

```python
a_R = (a * R) % q
b_R = (b * R) % q
product = (a_R * b_R) % q           # Contains two factors of R.
coefficient = (product * R_inv) % q # Keep one factor of R.
```

Here, `q = 3329`, and `R_inv` is the modular inverse of `R`: `(R * R_inv) % q = 1`.

**What the model missed.** The model without tools accepted the transform in all six trials. It treated the stage formulas and modular arithmetic as correct. But it did not correctly calculate the scale factors in the combined stages. In `r3`, it noticed the distinction between one and two factors, then incorrectly treated the representations as equivalent.

**What tools found.** Solo and Debate ran the actual kernel and compared outputs with an independent integer reference. Each found permitted inputs that produced incorrect outputs. The [Solo trace from `r3`](../traces_glm/case_78/solo/r3/transcript.md) contains a simple example:

```text
Row length                 = 64
Each input element         = 1
Correct output at index 0  = 64
Kernel output at index 0   = 2954
```

The correct sum is 64, which is less than 3329. Thus, the remainder must also be 64. The kernel returns 2954, so its fixed label is **Reject**. Execution exposed an error in the combined stages that the responses without tools did not resolve.

Traces, `r1`: [No tools](../traces_glm/case_78/single_call/r1/transcript.md) · [Solo](../traces_glm/case_78/solo/r1/transcript.md) · [Debate](../traces_glm/case_78/debate/r1/transcript.md) · [Independent validation](../real_kernel_challenges/private_data/qualification/r23.json).
