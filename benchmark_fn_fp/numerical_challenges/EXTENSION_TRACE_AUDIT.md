# case_44–case_49 extension trace audit

Storage update (2026-09-30): Batch labels and repeat shorthand below retain their original experimental meaning; original labels are saved in `trace_meta.json.original_trial`. Canonical links use per-case/arm `rN` folders; see [trace naming and migration](../TRACES.md).

Trials: `extension_low32_r1` and `extension_low32_r2`. Reviews: 2026-09-24 UTC.
The main sections audit r1; the final section independently audits r2.

This is a read-only review of saved probe source, stdout/stderr, tool events,
claim evidence, and final verdicts. It made no new model or GPU calls and did
not edit original traces. Numerical comparisons use the frozen
[`private_data/validation_gpu.json`](private_data/validation_gpu.json), rather than treating an agent's
verdict as its own ground truth.

## Scope and core measurement checks

All twelve completed runs below have been inspected. All directly execute
the public case's GPU kernel and use the specified FP64 reference. Their
decisive measured errors agree with the frozen GPU oracle to numerical
reduction precision. case_47 debate has an incorrect, inactive denominator floor,
documented below; its actual denominator still equals the contract denominator.

| Case | Arm | Decisive probe | Measured relative L2 | Contract threshold | Verdict |
|---|---|---|---:|---:|---|
| case_44 | solo | [t7](../traces_glm/case_44/solo/r1/probes/t7_probe.py) | 0.00024261613799325 | 0.02 | trust |
| case_45 | solo | [t7](../traces_glm/case_45/solo/r1/probes/t7_probe.py) | 0.14481964583989307 | 0.02 | reject |
| case_46 | solo | [t6](../traces_glm/case_46/solo/r1/probes/t6_probe.py) | 0.023636854556236472 | 0.08 | trust |
| case_47 | solo | [t7](../traces_glm/case_47/solo/r1/probes/t7_probe.py) | 0.12440399903958174 | 0.08 | reject |
| case_48 | solo | [t6](../traces_glm/case_48/solo/r1/probes/t6_probe.py) | 0.00003372867387109279 | 0.0002 | trust |
| case_49 | solo | [t7](../traces_glm/case_49/solo/r1/probes/t7_probe.py) | 0.0004729252164019362 | 0.0002 | reject |
| case_44 | debate | [t12](../traces_glm/case_44/debate/r1/probes/t12_probe.py), [t13](../traces_glm/case_44/debate/r1/probes/t13_probe.py) | 0.00024261613799325 | 0.02 | trust |
| case_45 | debate | [t12](../traces_glm/case_45/debate/r1/probes/t12_probe.py) | 0.14481964583989307 | 0.02 | reject |
| case_46 | debate | [t12](../traces_glm/case_46/debate/r1/probes/t12_probe.py) | 0.023636854556236472 | 0.08 | trust |
| case_47 | debate | [t13](../traces_glm/case_47/debate/r1/probes/t13_probe.py) | 0.12440399903958174 | 0.08 | reject |
| case_48 | debate | [t12](../traces_glm/case_48/debate/r1/probes/t12_probe.py) | 0.00003372867387109279 | 0.0002 | trust |
| case_49 | debate | [t8](../traces_glm/case_49/debate/r1/probes/t8_probe.py) | 0.0004729252165482065 | 0.0002 | reject |

case_44/case_45 probes use the centered population variance reference with epsilon 1e-5.
case_46/case_47 use `np.linalg.solve` on the actual supplied FP32 matrix and RHS cast to
FP64; the reference is the system's solution, not the 64-step iterate. M and
case_49 solo use the direct polynomial power sum on the stored FP32 coefficients
and points cast to FP64, including the already-rounded constant coefficient.
case_49 debate uses FP64 Horner on the same stored inputs. The FP64 evaluation-order
differences alter the relative-error measurement by less than 2e-13.

Most probes call `make_inputs()` directly. case_45 debate reconstructs the exact
published PCG64 seed 782406 and input formula before calling the public
`kernel.run`; its inputs and decisive result match the public workload. L
constructs the CPU reference using `make_inputs_numpy()` and GPU inputs using
`make_inputs("cuda")`; these share the same deterministic public generator.
No reviewed decisive measurement substitutes a different input or a CPU
emulation for the actual GPU output.

case_46/case_47/case_48/case_49 debate and case_49 solo were unavailable at the initial inspection; they
were explicitly inspected in the completion review. case_49 solo, case_49 debate, and case_48
debate include complete output arrays: reconstructing their FP32 bytes yields
the exact output SHA256 values recorded by the frozen GPU oracle. The separate
case_44 solo smoke trial is outside this audit's scope. The final section separately
covers every r2 repetition after its run archive became available.

## case_44 debate: incorrect auxiliary denominator diagnostic, valid core verdict

In [case_44 debate t12](../traces_glm/case_44/debate/r1/probes/t12_probe.py),
`eff_den` is computed as `norm(output) / norm(x - mean64)`. This estimates the
reciprocal of the effective denominator. Estimating the denominator requires
the reverse ratio, with the small FP32-versus-FP64 mean difference also taken
into account.

Consequently, its saved `eff_den_from_output = 7.816455355342365` and
`den_rel_err = 60.1117729789066` are invalid denominator diagnostics. They must
not be reported as a real kernel denominator or a real 6011% denominator
error. The same probe's independently reconstructed `den_f32 =
0.1279352307319641` and reference `den64 = 0.12790424781228815` are consistent
with approximately 0.0242% denominator error.

The core measured `rel_l2 = 0.00024261613799325` is computed directly from the
actual GPU output and the correct FP64 reference, independently of those
incorrect auxiliary fields. It matches the frozen oracle. A second actual
GPU probe, t13, reproduces the same full-output error while measuring the
mean contribution separately. Thus the trust verdict remains supported.

The agent also explicitly recognized the problem in its saved
[claim evidence](../traces_glm/case_44/debate/r1/claims.json):
the claim c1 evidence summary calls `eff_den_from_output` and `den_rel_err`
“a probe-side formula artifact” and states that the direct relative-L2
comparison is decisive. The original probe was not corrected or rerun for
this diagnostic. This is a recovered auxiliary explanation error, not a
measurement failure or an ignored contract violation.

## case_47 debate: incorrect inactive floor and limited rounding diagnostic

[case_47 debate t13](../traces_glm/case_47/debate/r1/probes/t13_probe.py)
uses `max(norm(x), 0.04)` instead of the required
`max(norm(x), 0.001*sqrt(16))`, whose floor is 0.004. This is a metric-expression
error. For the only in-scope input, `norm(x) = 18.216566495477316`, so both
expressions produce the same denominator. The measured error
0.12440399903958174 exactly matches the frozen GPU oracle and exceeds 0.08.
The correct contract floor was also used in this run's separate t12 FP64
iteration diagnostic. No alternative low-norm input is in scope; thus the
typo does not change this case's reject verdict, but the t13 code should not
be described as a generally faithful implementation of the contract metric.

The t13 field `fp32_rounding_extra = -3.396286464563136e-08` subtracts the FP64
iteration's error norm from the GPU output's error norm. This is the signed
change in total relative error, not the norm of the FP32-versus-FP64 state
difference. The final statement that rounding contributes approximately
3e-8 needs this qualification; those two norms alone do not establish a
3e-8 bound on the rounding error vector. The decisive reject measurement
does not depend on this decomposition.

case_46 debate t13 also reports the difference between two error norms as a
`rounding_increment`, but additionally reports the explicit maximum absolute
FP32-versus-FP64 simulated state difference (1.40e-7). Its CPU reduction order
need not be bitwise identical to Triton's. The actual GPU verdict measurement
is the independent t12 execution.

## Trace and exception handling observations

- All reviewed probes exited with code 0, did not time out, and have empty
  stderr. All twelve reviewed run archives have empty `runner_error.txt`.
- case_46 solo t6 prints a Python dictionary rather than JSON. The recorder retains
  `json_parse_error: last stdout line is not JSON`, and no parsed JSON result
  artifact is available. Its full stdout contains the correct numeric error,
  finite flag, and shape; the verdict accurately uses those values. This is
  a structured-output limitation, not a failed GPU execution.
- case_44/case_45/case_47/case_49 solo and case_44/case_45/case_46/case_47/case_48 debate initially made invalid `record_claim` calls
  lacking `scope_rationale`. The tool recorded recoverable ledger errors;
  subsequent calls supplied the missing information and produced the claims
  used by the probes. These exceptions are retained in the trace and do not
  invalidate later successful runtime evidence.
- case_45 debate t13, case_46 debate t13, case_47 debate t12, case_48 debate t13, and case_49 debate t9
  are CPU-only explanatory diagnostics. Their companion decisive probes
  actually execute the GPU kernel. GPU permission alone is not evidence that
  every diagnostic ran a GPU kernel.

This audit supports measurement and verdict correctness for the explicitly
reviewed rows. It does not establish that debate improves accuracy over solo,
nor that every intermediate explanatory statement is correct.

## Second-round confirmation: `extension_low32_r2`

All twelve r2 tool runs have now been inspected independently through their
saved probe source, outputs, process exit status, and verdict. Every row below
has evidence of calling the public GPU kernel on the exact fixed input and
comparing against the required FP64 reference with the correct contract
denominator. The core measurements agree with the frozen oracle; the largest
relative-error difference is 2.26e-13, arising from FP64 polynomial summation
order. case_48/case_49 solo and debate additionally save complete output arrays, whose
reconstructed FP32 bytes exactly match the oracle output hashes.

| Case | Arm | Core GPU probe | Relative L2 | Verdict |
|---|---|---|---:|---|
| case_44 | solo | [t7](../traces_glm/case_44/solo/r2/probes/t7_probe.py) | 0.00024261613799325 | trust |
| case_44 | debate | [t9](../traces_glm/case_44/debate/r2/probes/t9_probe.py) | 0.00024261613799325 | trust |
| case_45 | solo | [t7](../traces_glm/case_45/solo/r2/probes/t7_probe.py) | 0.14481964583989307 | reject |
| case_45 | debate | [t8](../traces_glm/case_45/debate/r2/probes/t8_probe.py) | 0.14481964583989307 | reject |
| case_46 | solo | [t7](../traces_glm/case_46/solo/r2/probes/t7_probe.py), stdout evidence | 0.023636854556236472 | trust |
| case_46 | debate | [t12](../traces_glm/case_46/debate/r2/probes/t12_probe.py) | 0.023636854556236472 | trust |
| case_47 | solo | [t7](../traces_glm/case_47/solo/r2/probes/t7_probe.py) | 0.12440399903958174 | reject |
| case_47 | debate | [t12](../traces_glm/case_47/debate/r2/probes/t12_probe.py) | 0.12440399903958174 | reject |
| case_48 | solo | [t7](../traces_glm/case_48/solo/r2/probes/t7_probe.py) | 0.00003372867393290552 | trust |
| case_48 | debate | [t12](../traces_glm/case_48/debate/r2/probes/t12_probe.py) | 0.000033728673707072835 | trust |
| case_49 | solo | [t7](../traces_glm/case_49/solo/r2/probes/t7_probe.py) | 0.0004729252164019362 | reject |
| case_49 | debate | [t12](../traces_glm/case_49/debate/r2/probes/t12_probe.py) | 0.0004729252165880834 | reject |

The correct verdicts are supported by measured results, but this does **not**
mean every probe process exited successfully:

- case_46 solo t7 ran the GPU kernel and correct FP64 solve, then printed the full
  error, reference norm, finite flag, and shape to stdout. Its subsequent
  JSON print failed because `np.bool_` is not serializable, producing exit 1.
  No follow-up probe was run. The saved verdict explicitly recognizes that
  serialization failure and relies on the already printed measurement, which
  matches the oracle. This is valid raw measurement evidence with an unclean
  probe-process exit, not a clean JSON probe success.
- case_44 debate t8 and case_45 solo t6 failed by passing CUDA tensors to NumPy; corrected
  t9 and t7 respectively reran the public GPU kernels and measured the result
  successfully. case_48 debate t10 failed during JSON serialization of `np.float32`;
  corrected t12 reran and produced a complete successful JSON result. Original
  failed probe sources and stderr remain preserved. Their empty initial stdout
  was not used as numerical evidence.

All twelve r2 run archives have empty `runner_error.txt`. case_46/case_47's auxiliary
rounding-shift diagnostics still compare error norms rather than bounding the
rounding error vector; the decisive contract checks use direct GPU output
versus the exact-system reference and do not rely on that interpretation.
Thus both rounds' 24 tool-arm verdicts have actual numerical evidence, with
the probe-quality limitations above retained rather than erased.
