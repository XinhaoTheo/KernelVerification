# Metric Selection

Choose the verification metric from the operator's correctness contract, not
from convenience.

## Continuous Outputs

For matmul, elementwise ops, reductions, and normalizations with continuous
numeric outputs, elementwise error can be useful evidence. A probe should report
at least max absolute error, max relative error when meaningful, shape, dtype,
and the tested input distribution.

## Saturating Or Lossy Outputs

For softmax, sigmoid, GELU, ReLU, SiLU, FP8, FP4, INT4, or any lossy/saturating
path, final-output allclose is not enough by itself. Prefer metrics that expose
where the error lives:

- active region versus dead or saturated region;
- dominant entries versus tails;
- quantized-reference mismatch versus FP32-reference mismatch;
- saturation, zero, and code/tie counts for low-bit formats.

## Selection Outputs

For sort, top-k, argmax, sparse attention indices, or any discrete selection,
state which metric is authoritative before judging the claim:

- exact index order;
- selected index set;
- selected values;
- value gap between missed and selected elements;
- downstream output after consuming the selected indices.

Index mismatch alone is weak evidence when the contract permits ties or
non-stable ordering. Missing a value far above the cutoff is stronger evidence
than swapping two tied boundary values.

## Probe Result Format

Runtime probes should print a final JSON object that includes the chosen metric,
the reason that metric matches the claim, and enough raw counts or extrema for
Judge to distinguish a real correctness failure from a metric artifact.
