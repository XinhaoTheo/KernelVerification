# case_54–case_57 tool-trace audit

Storage update (2026-09-30): Batch labels and repeat shorthand below retain their original experimental meaning; original labels are saved in `trace_meta.json.original_trial`. Canonical links use per-case/arm `rN` folders; see [trace naming and migration](../TRACES.md).

Scope: `case_54` through `case_57`, both `solo` and `debate`, trials
`oz_low32_r1` and (when available) `oz_low32_r2` under
`benchmark_fn_fp/traces_glm/<case>/<arm>/rN/`.

Review updated 2026-09-24 UTC after all eight r1 tool runs were collected. The
initial snapshot had only reservation metadata; it has been superseded for
all r1 runs. All eight r2 runs have also been reviewed.
No traces were changed, and
this review used no model, network, or GPU calls.

| Case | Family | Oracle label | Budget | r1 solo | r1 debate | r2 solo | r2 debate |
|---|---|---|---:|---|---|---|---|
| case_54 | Continuous oscillatory integral / midpoint approximation | trust | 0.035 | Verified t7 | Verified t8/t9 | Verified t6 | Verified t15 after t12 import fix |
| case_55 | Continuous oscillatory integral / midpoint approximation | reject | 0.035 | Verified t8 after t7 syntax fix | Verified t14/t15 after two fixes | Verified t7 | Verified t12/t15; diagnostic corrected |
| case_56 | Full Fourier signal / retained-frequency approximation | trust | 0.15 | Verified t7; harmless warning | Verified t10 | Verified t8 after JSON serialization fix | Verified t12/t13 |
| case_57 | Full Fourier signal / retained-frequency approximation | reject | 0.15 | Verified t6 | Verified t8 after t7 API fix | Verified t7 | Verified t10 |

## Completed r1 evidence

All eight reviewed runs execute the public `kernel.run` on CUDA tensors generated
by the corresponding public input generator. case_56-debate generates the public
NumPy arrays and explicitly uploads them; its probe records `device: cuda`.
The other runs use `make_inputs()`. The stored kernel source in
`run.json` hashes to the frozen GPU validation source. Their decisive probes
exit successfully, write parsed JSON metrics, and the final verdict cites that
successful evidence. Probe source/stdout/stderr/result hashes match the tool
event artifacts, including the failed attempts retained for case_55-solo, case_55-debate,
and case_57-debate.

| Case/arm | Decisive probe | Measured relative L2 | Frozen T4 relative L2 | Result |
|---|---|---:|---:|---|
| case_54 solo | t7 | 0.001552519962372429 | 0.001552519962372429 | trust |
| case_54 debate | t8 | 0.001552519962372429 | 0.001552519962372429 | trust |
| case_55 solo | t8 | 0.11888436797488026 | 0.11888436797488026 | reject |
| case_55 debate | t14 | 0.11888436797488026 | 0.11888436797488026 | reject |
| case_56 solo | t7 | 0.07170935573821624 | 0.07170935573821630 | trust |
| case_56 debate | t10 | 0.07170935573821624 | 0.07170935573821630 | trust |
| case_57 solo | t6 | 0.23046426609697562 | 0.23046426609697562 | reject |
| case_57 debate | t8 | 0.23046426609697562 | 0.23046426609697562 | reject |

The quadrature probes cast stored FP32 amplitudes/frequencies/phases to FP64
and use the correct continuous closed-form integral. They use the correct
relative L2 denominator floor (`1e-12`) and tolerance (`0.035`). Their recorded
output vectors also reproduce the frozen T4 output hashes. case_54-debate t9 adds
a per-term quadrature diagnostic; its FP64 per-term approximation is not used
as a substitute for the actual GPU output in decisive t8. case_55-debate t14 likewise
compares the actual GPU output to the continuous reference; its extra midpoint
reference is only diagnostic. case_55-debate t15 runs a separate CUDA sine probe to
quantify the `tl.sin` approximation error, which is supplementary to t14.

The spectral probes sum all sixteen reference modes in FP64, using stored FP32
coefficients and offset. case_57-solo calls the deterministic public generator once
for NumPy reference inputs and once for CUDA kernel inputs; both calls use the
same fixed seed and generator. Their metric uses the correct denominator
floor (`1e-12`) and tolerance (`0.15`). Direct-sum and FFT reference arithmetic
explain the `5.56e-17` metric difference for case_56-solo; it is immaterial to the
decision. All eight recorded verdicts agree with the frozen oracle labels.

Recovered issues and wording limitations:

- case_55-solo t7 has an unmatched parenthesis and exits before execution. Its t8
  replacement runs successfully, computes the contract metric, and supplies
  the final decision's evidence. The final reason calls the angular-frequency
  range "220 Hz"; the public formula uses radians per unit interval. This is
  a units error in the prose, not in the reference calculation or measured
  threshold violation.
- case_57-debate t7 calls nonexistent `Tensor.float64()` after invoking the kernel.
  Its t8 replacement uses `.cpu().numpy().astype(np.float64)` and successfully
  reruns the kernel and full reference. The failed t7 is not used as numerical
  evidence for the verdict.
- case_56-solo t7 emits NumPy's deprecation warning for conversion of a length-one
  array to `float`. It still exits zero and produces the correct scalar offset,
  full reference, and metric. No recovery was required in this environment.
- case_55-debate t12 passes an invalid `axis` argument to `np.sin`; t13 contains an
  invalid broadcast in its supplementary sine diagnostic. Both errors are
  retained. Corrected t14 and t15 respectively exit zero, provide the intended
  contract and diagnostic measurements, and are cited by the final verdict.
  The t14 source catches potential kernel exceptions, but its saved output
  contains a real GPU vector and non-null measured error; no exception was
  hidden in the successful evidence.
- case_56-debate's final prose describes the cutoff as "sanctioned" by a descriptive
  clause. The operative acceptance criterion is still the numerical bound;
  its explicit 16-mode comparison establishes that bound without relying on
  the wording interpretation.

All 52 captured LLM call directories across the eight runs contain request,
response, and metadata files. Counts are case_54-solo 5, case_54-debate 6, case_55-solo 6,
case_55-debate 10, case_56-solo 5, case_56-debate 9, case_57-solo 4, and case_57-debate 7. This is a
persistence inventory, not an audit of provider-internal HTTP retries or billing.

## Collected r2 evidence

All eight reviewed r2 runs independently execute the frozen public kernel on CUDA
and measure the contract's full reference and relative L2 metric. Fixed NumPy
and CUDA generator calls use the same seed and stored FP32 values. Frozen
kernel hashes, tool-event probe source and artifact hashes, and oracle labels
match. For case_54/case_55, saved output vectors also reproduce frozen T4 output hashes.

| Case/arm | Decisive probe | Measured relative L2 | Result |
|---|---|---:|---|
| case_54 solo | t6 | 0.001552519962372429 | trust |
| case_54 debate | t15 | 0.001552519962372429 | trust |
| case_55 solo | t7 | 0.11888436797488026 | reject |
| case_55 debate | t12, corroborated by t15 | 0.11888436797488026 | reject |
| case_56 solo | t8 | 0.07170935573821625 | trust |
| case_56 debate | t12, corroborated by t13 | 0.07170935573821624 | trust |
| case_57 solo | t7 | 0.23046426609697562 | reject |
| case_57 debate | t10 | 0.23046426609697562 | reject |

All these metric values equal the frozen T4 metric within `5.56e-17`.
Recovered errors and auxiliary limitations are retained:

- case_54-debate t12 fails because `make_inputs` was not imported. Corrected t15
  successfully runs CUDA and reports `kernel_error: null`; its closed-form
  reference and measured contract error are correct. Supplementary t13 uses
  `norm(err)` on the 4-by-8 per-term error matrix and labels it total quadrature
  L2. That is the Frobenius norm of component errors, not the L2 norm after
  summing the terms per output row. Its comment and the final prose's
  `~0.0033` total-error claim are inaccurate. The decisive t15 computes the
  correct four-output metric; the label remains independently supported.
- case_55-debate t13 fails on undefined variable `fp32`; corrected diagnostic t15
  succeeds. Its t12 decisive contract metric was already successful. The t12
  field `per_row_abs_err` actually contains signed differences, and
  `dtype_shape_ok` is hardcoded rather than checked. The preserved output and
  public kernel still establish the correct four-element output, and the
  numeric contract test is valid. The supplementary t15 "fraction of total"
  divides relative norms normalized by different reference norms; it is not a
  literal error decomposition fraction. Its recorded raw differences correctly
  show that FP32 arithmetic is negligible compared with quadrature error.
- case_56-solo t7 successfully runs the kernel but cannot serialize `np.bool_` into
  JSON and also emits a scalar-conversion warning. Corrected t8 converts the
  Boolean explicitly, extracts `offset[0]`, and reruns successfully with clean
  stderr and the proper all-sixteen-mode reference.
- case_56-debate t12 measures the full contract; t13 additionally compares against
  six retained modes to isolate arithmetic error. That diagnostic does not
  replace the full reference. Its `fp32_error_below_tolerance` field tests a
  stricter `1e-3` auxiliary threshold rather than the displayed `0.15`; both
  are satisfied and the independently measured full-contract result is valid.
- case_57-debate t10 also computes the correct orthogonal-basis dropped-mode energy.
  The final prose says its ratio matches measured error "within 1e-8", whereas
  the stored difference is about `1.28e-8`. This small rounding-description
  error does not affect the decisive `0.230464... > 0.15` contract check.

Request, response, and metadata files are present for all 58 captured calls in
the reviewed r2 runs: case_54-solo 4, case_54-debate 10, case_55-solo 5, case_55-debate 10, case_56-solo 6,
case_56-debate 9, case_57-solo 5, and case_57-debate 9.

For remaining and subsequent collected traces, inspect each run for:

- Execution of the frozen public kernel on CUDA using its own `make_inputs()`;
  connect that execution to persisted probe source, stdout, and tool results.
- A reference formed from the actual stored FP32 inputs. case_54/case_55 require the
  continuous closed-form integral, not a more finely sampled surrogate;
  case_56/case_57 require all sixteen Fourier modes, not the six-mode approximation.
- The contract's relative L2 error and denominator floor; compare the measured
  error and decision with `private_data/validation_gpu.json` without treating a correct
  final label alone as proof of verification.
- Probe exceptions, unsuccessful imports, or failed GPU calls, including
  whether a later successful probe actually recovered the missing evidence.
- Completeness of the request/response capture and final verdict provenance.
