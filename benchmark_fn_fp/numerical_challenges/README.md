# Numerical challenges beyond the case_36/case_37 pair

Twenty-four synthetic finite-workload cases test twelve numerical mechanisms. They are
kept separate from the original FN/FP benchmark: the question is whether the
published, fully reconstructible input meets a final numerical error contract.
They make no claim that a production kernel has a bug or that these inputs occur
frequently in real model activations.

| Family | Case | T4 relative L2 error | Allowed error | Ground truth |
| --- | --- | ---: | ---: | --- |
| Attention sensitivity to logit quantization | case_38 | 0.809816% | 2% | trust |
| Attention sensitivity to logit quantization | case_39 | 3.652389% | 2% | reject |
| State rounding in a nonnormal recurrence | case_40 | 0.076903% | 0.2% | trust |
| State rounding in a nonnormal recurrence | case_41 | 0.490784% | 0.2% | reject |
| Sequential floating-point summation order | case_42 | 90.019872% | 10% | reject |
| Sequential floating-point summation order | case_43 | 0% | 10% | trust |
| Raw-moment LayerNorm cancellation | case_44 | 0.024262% | 2% | trust |
| Raw-moment LayerNorm cancellation | case_45 | 14.481965% | 2% | reject |
| Fixed-iteration SPD solve | case_46 | 2.363685% | 8% | trust |
| Fixed-iteration SPD solve | case_47 | 12.440400% | 8% | reject |
| FP32 Horner polynomial evaluation | case_48 | 0.003373% | 0.02% | trust |
| FP32 Horner polynomial evaluation | case_49 | 0.047293% | 0.02% | reject |

All labels were verified on a Tesla T4 with Torch 2.8.0, Triton 3.4.0 and NumPy
1.26.4 before model evaluation. Each case was executed ten times, with independent
CPU references, reproducible input hashes and input nonmutation checks. See
[`private_data/validation_gpu.json`](private_data/validation_gpu.json) for exact errors and source hashes.

The additional twelve case_50–case_61 cases are described in [OZ_DESIGN.md](OZ_DESIGN.md),
with a separate [protocol](OZ_PROTOCOL.md) and [results](OZ_REPORT.md). The
[global case index](../CASE_INDEX.md) includes these 24 public workloads, the
earlier case_36/case_37 pair, and every other dataset, with links to their contracts and records. The table above
covers case_38–case_49; case_50–case_61 appear in that complete index.

## Mechanisms

case_38/case_39 quantize logits before softmax and multiply the resulting probabilities by
values. The probability error itself is identical between cases. A different
public row permutation of the values changes how that error affects the final
attention output. Reference norms remain around four, away from the metric floor.

case_40/case_41 run 64 updates of a 16-dimensional linear recurrence, rounding each state to
FP16. A nonnormal transition matrix can transiently amplify rounding error even
though its eigenvalues have magnitude below one. Two public seeds produce
different trajectories and final relative errors. Independent 80-digit Decimal
references check the FP64 result; three CPU reduction orders agree on the final
FP16 output for both selected workloads.

case_42/case_43 sum rows containing exactly balanced large positive/negative terms and small
positive terms. The kernel accumulates sequentially in FP32. Both cases have the
same per-row mathematical sums, but their public column permutations change the
rounding path. One order loses most of the small signal; the other yields the
exact result. Source-only classification requires establishing the actual order,
which depends on the specified PCG64 permutation and literal index sequence.

case_44/case_45 compute LayerNorm variance from FP32 raw moments, E[x²] − E[x]², on values
clustered near 64. Different seeds change cancellation and accumulation error.
The stable centered FP64 reference is cross-checked with shifted math.fsum.

case_46/case_47 solve the same 16×16 SPD system with 64 Richardson iterations. Only the RHS
seed changes. Its projection onto slow eigenmodes controls the finite-iteration
error; NumPy solve and independent 80-digit Decimal elimination agree.

case_48/case_49 evaluate degree-48 polynomials with FP32 Horner multiply/add rounding
separately. Only the seed changes coefficients and evaluation points. FP64
Horner is checked against Decimal direct power summation, both using the actual
stored FP32 coefficients. Near cancellation makes the final error depend on the
particular values and evaluation path.

## Evaluation and records

[`PROTOCOL.md`](PROTOCOL.md) fixes the initial three-arm comparison and repeat
gate for case_38–case_43 with a 64K per-call output cap. Subsequent case_38–case_43 low-effort 8K trials
are separately documented in [`EXPERIMENT_NUMERICAL_LOW.md`](EXPERIMENT_NUMERICAL_LOW.md).
The case_44–case_49 [`extension protocol`](EXTENSION_PROTOCOL.md) fixes low-effort 32K trials
for all three arms plus a separate default-reasoning 64K source-only control.
Its [design and full CPU search counts](EXTENSION_DESIGN.md) explain each new
mechanism; the [probe audit](EXTENSION_TRACE_AUDIT.md) records both decisive
runtime evidence and recovered diagnostic mistakes.
The solo and debate arms have GPU execution tools; the single-call arm does not.
Total compute budgets of solo and debate are not matched. The report must state
separately whether tools help and whether debate improves on solo.

Model results are generated in [`REPORT.md`](REPORT.md), with the case_44–case_49 extension
reported separately in [`EXTENSION_REPORT.md`](EXTENSION_REPORT.md). All canonical traces
are in `../traces_glm/<case>/<arm>/<trial>/`, alongside older GLM experiments.
Each record includes dataset provenance, full raw API calls and token usage;
tool runs also retain probe code, outputs and agent state. Never overwrite a
trial or change a case after its first model evaluation.

The private `families/` builders and `private_data/answer_key_*.json` files are only for dataset
construction and scoring. Agent containers receive `eval_cases/` alone. Formal
CPU candidate searches are retained; selection occurs before model calls in the
initial batch. The incomplete early case_58/case_59 and case_60/case_61 parameter-exploration records
are disclosed in [OZ_EARLY_PARAMETER_EXPLORATION.md](OZ_EARLY_PARAMETER_EXPLORATION.md).
Later adaptive selection and repeat results must remain explicit.

Given the stated reference and metric, a fixed numerical script can solve these
cases. Successful tool use here does not establish a unique need for agents or
debate, nor does this small selected collection establish general accuracy.

## case_44–case_49 completed extension results

The 48 prespecified experimental runs are terminal. Under the common GLM low
reasoning / 32768-token setting, source-only was correct in 11/18 attempts
(three repetitions per case); solo and debate were each correct in 12/12
(two repetitions per case). All 42 low-setting runs produced explicit verdicts.
The first two source rounds were each 3/6, and the third was 5/6.

| Pair | Source-only correct / 6 | Solo correct / 4 | Debate correct / 4 |
|---|---:|---:|---:|
| case_44/case_45: LayerNorm | 3/6 | 4/4 | 4/4 |
| case_46/case_47: iterative solve | 5/6 | 4/4 | 4/4 |
| case_48/case_49: polynomial | 3/6 | 4/4 | 4/4 |

I and M satisfy the exploratory low-setting replication gate: source-only made
three and two explicit errors respectively, with both tool arms correct in
both repetitions. The separate default-reasoning / 65536-token control produced
wrong verdicts for case_44/case_46, correct verdicts for case_45/case_47, and provider 504 errors for
case_48/case_49. Those failures are not classification errors and do not establish an case_48/case_49
gap under default reasoning. No paid retry replaced a failed control slot.

The recorded API estimate for this extension is $1.055612, excluding Modal GPU
cost and unknown charges for the two 504 calls that returned no usage. Full
requests and returned responses, plus failure traces, are retained. Both rounds
of all 24 tool runs have been audited for actual GPU measurement evidence;
recovered probe mistakes remain documented. See the [complete results](EXTENSION_REPORT.md)
and [trace audit](EXTENSION_TRACE_AUDIT.md). There is no observed final-label
accuracy advantage for debate over solo in these runs.

## case_50–case_61 completed extension results

All 96 prespecified runs are terminal. The twelve new cases passed independent
reference checks and ten repeated Modal T4 validations each before model calls.
Under common GLM low reasoning / 32768-token per-call settings, source-only was
correct in 13/36 attempts, with 19 explicit errors and four abstentions; solo
and debate were each correct in 24/24. Source-only has three repetitions per
case, and each tool arm has two. Total compute budgets are not matched.

| Pair | Mechanism | Source-only correct / 6 | Solo correct / 4 | Debate correct / 4 |
|---|---|---:|---:|---:|
| case_50/case_51 | Near-collinear projection and residual normalization | 1/6 | 4/4 | 4/4 |
| case_52/case_53 | Quantized nearest-neighbor routing | 3/6 | 4/4 | 4/4 |
| case_54/case_55 | Oscillatory quadrature | 2/6 | 4/4 | 4/4 |
| case_56/case_57 | Spectral truncation | 1/6 | 4/4 | 4/4 |
| case_58/case_59 | Ill-conditioned FP32 log-determinant | 3/6 | 4/4 | 4/4 |
| case_60/case_61 | Squared-distance cancellation and RBF regression | 3/6 | 4/4 | 4/4 |

The explicit errors comprise nine rejections of compliant cases and ten
acceptances of noncompliant cases. All four abstentions belong to case_56/case_57 and are
not counted as explicit errors. case_50/case_51/case_52/case_55/case_59/case_61 meet the fixed replication gate,
covering five families. These selected workloads demonstrate a gap for the
tool-enabled arms under this configuration, with no observed final-label
advantage for debate over solo. They do not establish generalization.

The separate default-reasoning / 65536-token source-only control returned four
correct verdicts, two explicit errors (Q and Z), and six provider 504 errors
(case_50/case_54/case_55/case_57/case_58/case_59). Failed calls remain in their original slots and were not retried.
These failures are not classification errors. Each default-control case has
only one attempt, so case_52/case_61 supply limited cross-configuration evidence.

The recorded API estimate for case_50–case_61 is $2.054085, excluding Modal GPU cost and
unknown charges for the six 504 calls. All 384 API call records are retained,
including all 378 returned responses with usage. The 90 runs with returned
responses have complete raw capture; the six provider failures have partial
failure traces. All 48 tool runs were audited for decisive GPU evidence;
recovered probe failures and inaccurate auxiliary explanations are preserved.

See the [case_50–case_61 results](OZ_REPORT.md), [case_50–case_61 design](OZ_DESIGN.md),
[case_50–case_53 trace audit](OZ_TRACE_AUDIT_OPQR.md),
[case_54–case_57 trace audit](OZ_TRACE_AUDIT_STUV.md),
[case_58–case_61 trace audit](OZ_TRACE_AUDIT_WXYZ.md), and the
[global case index](../CASE_INDEX.md).
