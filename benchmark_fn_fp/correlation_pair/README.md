# Two-path quantization: correlation witness

**2026-09-22 update:** Both actual Triton kernels have now been verified on T4,
with ten identical repeated outputs per case and matching CPU input hashes.
GPU errors are 0.2241158064 (case_36, reject) and 0.0336471468 (case_37, trust).
Source-only and tool-enabled trials are recorded separately in `traces/`;
the derived [REPORT.md](REPORT.md) and `scoreboard.json` contain their results.
The construction history below predates these paid evaluation trials.

The separate Fireworks GLM-5.3 replication uses three arms on both frozen cases:
source-only, solo with tools, and four-role debate with tools. Its prespecified
settings are in [FIREWORKS_PROTOCOL.md](FIREWORKS_PROTOCOL.md), records are in
`traces_fireworks/`, and [REPORT_FIREWORKS.md](REPORT_FIREWORKS.md) is generated
with `python benchmark_fn_fp/correlation_pair/report.py --fireworks`. Token-limit
exhaustion is recorded separately from a wrong verdict. Opus records are retained.
Completed six-trial findings: [FIREWORKS_FINDINGS.md](FIREWORKS_FINDINGS.md).

This is a new pair of finite-workload numerical-compliance cases, separate from
the existing benchmark and the earlier 24-case pilot. It tests whether a verifier
can determine the final error when the two branch errors have identical marginal
statistics but different correlation. See [FINDINGS.md](FINDINGS.md) for the
measured source-only failures and successful tool-enabled resolutions. The solo
tool control also succeeds, so this does not establish a debate-specific benefit.

## Operator and contract

The operator is `y = A @ x + B @ x`. The candidate Triton implementation performs
symmetric per-row low-bit quantization of each matrix, computes both GEMVs, and
adds their outputs. This is a synthetic implementation of ordinary projection
and addition operations, not a copied production kernel or an alleged upstream
bug. The reference uses the original FP32 inputs evaluated in FP64. The fixed
relative L2 budget is 0.1, with the denominator floor specified in problem.txt.
That budget is an exploratory benchmark choice, not a production standard.

Only the fully specified input instance is in scope. A passing instance does
not imply this approximate kernel is correct on arbitrary inputs.

## Construction

Both inputs share exactly the same x and A; B differs only by a row permutation.
Therefore B has identical elements, row multiset, norm, extrema, and singular
values in both cases. Each branch's scalar error multiset and error norm are
also unchanged. The final reference norms are nearly equal.

On the construction side, the per-row branch errors are measured once. B's rows
are matched either in opposite error order or in the same error order. This
changes error correlation without changing either marginal distribution. The
candidate pool uses a predeclared 10% threshold and 25% clearance requirement;
the first seed qualifies. There was no adaptive search against model responses.

The verifier sees the entire common kernel, the complete input generator, the
PCG64 seed, and the literal 64-entry permutation. Nothing required to reconstruct
the inputs is hidden. The sorting/selection recipe belongs to the construction
side, just like an adversarial test generator or answer key. The two source files
differ only in the permutation. No semantic pass/fail names are exposed.

## Local measurements

| Quantity | case_36 | case_37 |
| --- | ---: | ---: |
| Branch A error L2 | 0.849963 | 0.849963 |
| Branch B error L2 | 0.961688 | 0.961688 |
| Error-vector cosine | +0.95282 | -0.96345 |
| Final relative L2 error | 22.4116% | 3.3647% |
| Budget | 10% | 10% |
| CPU witness label | reject | trust |

Assuming independent branch errors gives approximately
`sqrt(0.849963**2 + 0.961688**2)/7.988 = 16.1%` for BOTH cases. That estimate
would reject both and misclassify case_37. This establishes a failure of this
specific shortcut, not a failure of an LLM: a model may recognize that the
correlation is unknown and correctly request evidence.

## Validation and limits

The initial construction executed locally with NumPy 1.26.4 without API or GPU
calls. Subsequent paid evaluation is recorded in REPORT.md and the update above.

* CPU FP32 emulation provides candidate labels, cross-checked using a sequential
  FP32 reduction instead of NumPy's vectorized reduction.
* FP64 reference sums are checked against independent scalar `math.fsum`.
* The actual verifier-visible generator is executed and its arrays checked
  against saved NPZ tensors and SHA-256 hashes.
* The exact row multiset of B and both branch error norms are checked.
* A conservative reduction-order sensitivity bound is recorded, conditional on
  identical products and quantization codes. It does not validate GPU lowering,
  GPU division or compilation.

The initial construction was CPU-only. See `validation_gpu.json` for the later
GPU confirmation and `REPORT.md` for model evaluations. These remain isolated
from the production scoreboard so exploratory trials are not merged into the
original benchmark results.

The permutation is deliberately selected from measured errors. This is a
controlled synthetic witness, not evidence that such alignments are common in
model activations. All information is exposed, but the finite seeded arrays
still favor a system that can compute; this alone says nothing about the benefit
of multiple agents. With the reference and metric supplied, a fixed numerical
script can decide the cases too.

## Reproduction and artifacts

```sh
/opt/anaconda3/bin/python benchmark_fn_fp/correlation_pair/build.py
/opt/anaconda3/bin/python benchmark_fn_fp/correlation_pair/validate_cpu.py
```

`eval_cases/` contains only source and contract. `answer_key_cpu.json` contains
construction-side measurements, `search_log.json` records candidates, and
`validation_cpu.json` records the independent checks. The NPZ files are exact
materialized input data; the model can reconstruct them from its source alone.

`build.py` now refuses to overwrite evaluated artifacts. Reconstruct in a fresh
directory containing the build scripts if needed; use validate_cpu.py to check
the frozen current artifacts without changing them.

The initial acceptance gates were GPU verification of these two kernels, followed
by two adequately budgeted source-only calls. An inconclusive verdict
must remain distinct from a wrong answer or a token-budget exhaustion. Only if
this establishes a useful difficulty should tool-enabled agents be evaluated.
