# 55 — GELU approximation

[Back to success cases](../SUCCESS_CASES.md)

**This kernel meets the accuracy requirement, but the model without tools incorrectly rejected it.** The tool-assisted workflows checked the actual numerical error and correctly accepted it.

**What the kernel computes.** GELU is defined as `GELU(x) = x * Phi(x)`, where `Phi(x)` is a Gaussian probability. The kernel approximates that probability, then multiplies it by `x`. The [problem description](../triton_eval_cases/case_55/problem.txt) requires the final output to differ from mathematical GELU by at most **`0.000001`** for every finite FP16 input.

The relevant [code](../triton_eval_cases/case_55/kernel.py) is:

```python
# Approximate the Gaussian tail.
envelope = tl.exp(-(0.5 * (magnitude * magnitude)))
tail = 0.5 * (polynomial * envelope)

# Convert the tail to an approximation of Phi(x).
probability = tl.where(x < 0.0, tail, 1.0 - tail)

# Compute GELU.
value = x * probability
```

**Why the model rejected it.** The model assumed the probability approximation could have an error of `7.5e-8`. It multiplied that estimate by the largest legal input, `65504`, and concluded that the output error would exceed the limit:

```text
Assumed probability error = 0.000000075
Largest legal input       = 65504
Their product             ≈ 0.0049
Allowed output error      = 0.000001
Model verdict             = Reject
```

**Why that reasoning is insufficient.** The probability error changes with `x`. In this implementation, the exponential factor becomes very small when `abs(x)` is large, reducing the approximation error. The model did not establish that the assumed probability error occurs at large inputs. Even if its upper bound were valid, an upper bound above the tolerance does not prove that any actual output exceeds the tolerance. It provides insufficient evidence to accept, but is not evidence of a defect.

**What the tools found.** Both tool-assisted workflows evaluated the finite FP16 input domain and returned **Accept**. Independent validation checked all **63,488 finite FP16 encodings** against a high-precision mathematical reference:

```text
Maximum absolute error bound = 0.000000526772
Allowed absolute error       = 0.000001
Failing scalar encodings     = 0
```

Additional checks covered tensor lengths, tail blocks, storage offsets, input preservation and separate output storage. Together, these checks support the fixed **Accept** label in the tested environment. Checking every finite FP16 value covers the scalar numerical domain; it is not a formal proof of every possible tensor execution.

**What this case demonstrates.** The no-tool model rejected the kernel based on an overly pessimistic error estimate. Tools supplied numerical evidence that the implementation meets the accuracy requirement, preventing a false rejection.

Traces: [No tools](../traces_glm/case_55/single_call/r1/transcript.md) · [Solo](../traces_glm/case_55/solo/r1/transcript.md) · [Debate](../traces_glm/case_55/debate/r1/transcript.md) · [Independent validation](../real_kernel_challenges/private_data/qualification/r14.json).
