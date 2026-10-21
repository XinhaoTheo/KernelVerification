# 55 — GELU approximation

[Back to Tools vs No Tools](../SUCCESS_CASES.md)

**This kernel agrees with the accuracy requirement, but the model without tools incorrectly rejected it.** Solo and Debate compared the numerical error with the limit and correctly accepted the kernel.

**What the kernel does.** The GELU formula is `GELU(x) = x * Phi(x)`, where `Phi(x)` is a Gaussian probability. The kernel uses an approximation for this probability. Then it multiplies the result by `x`. The [problem description](../triton_eval_cases/case_55/problem.txt) has an absolute error limit of **`0.000001`** from mathematical GELU. This limit is applicable to each finite FP16 input.

The [code](../triton_eval_cases/case_55/kernel.py) is:

```python
# Approximate the Gaussian tail.
envelope = tl.exp(-(0.5 * (magnitude * magnitude)))
tail = 0.5 * (polynomial * envelope)

# Convert the tail to an approximation of Phi(x).
probability = tl.where(x < 0.0, tail, 1.0 - tail)

# Compute GELU.
value = x * probability
```

**What the model did.** The model used `7.5e-8` as its probability error estimate. It multiplied this estimate by the maximum permitted input, `65504`. The result was more than the error limit, so the model rejected the kernel:

```text
Assumed probability error = 0.000000075
Largest legal input       = 65504
Their product             ≈ 0.0049
Allowed output error      = 0.000001
Model verdict             = Reject
```

**Why that decision is incorrect.** The probability error changes with `x`. In this implementation, the exponential factor becomes very small when `abs(x)` is large. This factor decreases the approximation error. The model did not show that its probability error estimate occurs at large inputs.

A correct upper bound can be more than the error limit without a failure in the output. This upper bound alone does not show whether the kernel is correct.

**What tools found.** Solo and Debate did tests with the finite FP16 input domain and accepted the kernel. We independently compared all **63,488 finite FP16 encodings** with a mathematical reference that has high precision:

```text
Maximum absolute error bound = 0.000000526772
Allowed absolute error       = 0.000001
Failing scalar encodings     = 0
```

More checks included tensor lengths, tail blocks and storage offsets. We also made sure that the inputs did not change and that the output used different storage. These results agree with the fixed **Accept** label in the environment of the tests. Tests with all finite FP16 values include all scalar input values. They do not give a formal proof of all possible tensor executions.

**What this case shows.** The model without tools rejected the kernel because its error estimate was too large. The numerical results from tools agreed with the accuracy requirement. These results prevented an incorrect rejection.

Traces: [No tools](../traces_glm/case_55/single_call/r1/transcript.md) · [Solo](../traces_glm/case_55/solo/r1/transcript.md) · [Debate](../traces_glm/case_55/debate/r1/transcript.md) · [Validation results](../real_kernel_challenges/private_data/qualification/r14.json).
