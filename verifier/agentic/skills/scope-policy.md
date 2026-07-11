# Scope Policy

Classify claims by the benchmark input contract before deciding whether they can
support a reject verdict.

## Benchmark Contract First

For this verifier, `in_scope` means the case is required by the benchmark or
artifact input contract, not merely by general PyTorch behavior. Prefer evidence
from `test.py`, `get_inputs`, `get_init_inputs`, seed files, `meta.json`, or
other artifact metadata that describes the actual verification domain.

`problem.txt` can explain the operator, but by itself it is usually not enough
to make a stress/generalization case decisive when `test.py` narrows the input
domain.

## Default Out-of-Scope Or Unknown

Treat the following as `unknown` or `out_of_scope` unless the benchmark contract
explicitly requires them:

- non-contiguous views or unusual strides;
- shapes, sizes, or ranks different from `get_inputs` / `test.py`;
- dtypes different from the benchmark inputs;
- module configuration changes such as a different LayerNorm `eps`;
- NaN/Inf-only rows or other undefined/reference-nonfinite cases;
- autograd, aliasing, zero-size tensors, broadcasting, or host/device variants.

These cases may be useful generalization notes, but they cannot by themselves
justify rejecting a benchmark kernel.

## Reference-Undefined Cases

If the PyTorch/reference computation is also NaN, non-finite, raises the same
exception, or has unspecified tie-breaking, the evidence is not a confirmed
correctness failure. Mark it `inconclusive`, `rebutted`, or narrow the claim to a
metric/contract artifact.

## Reject Discipline

A reject verdict requires a confirmed claim that is both `in_scope` and backed
by scope evidence from the benchmark/test input domain. Claims supported only by
operator generality or stress cases should lead to `trust`, `needs_more_evidence`,
or a non-decisive generalization note, not `reject`.


Do not use "not forbidden" or "no explicit constraint" as a substitute for
benchmark coverage. If `test.py` fixes `features = 64`, then `features = 0` is
not an in-scope decisive case unless another benchmark artifact explicitly
generates or requires it.
