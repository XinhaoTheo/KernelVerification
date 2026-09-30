# Independent construction audit: case_54–case_61

Date: 2026-09-23. Read-only review of the four new families, separate from their
builders. No model API calls, network calls, GPU calls, or changes to cases were
made in this audit. The table below reports CPU results; GPU validation remains
the responsibility of `validate_modal.py` and `private_data/validation_gpu.json`.

| Cases | Mechanism | CPU errors (pass / fail) | Tolerance | Maximum independent-reference difference |
|---|---|---:|---:|---:|
| case_54 / case_55 | Midpoint integration of oscillatory functions | 0.00155251996 / 0.11888438878 | 0.035 | 1.11e-16 |
| case_56 / case_57 | Truncation of Fourier reconstruction | 0.07170934896 / 0.23046426303 | 0.15 | 2.05e-15 |
| case_58 / case_59 | FP32 elimination log determinant | 2.0235510e-7 / 0.000711919995 | 0.0001 | 7.76e-12 |
| case_60 / case_61 | Expanded squared distances propagated through RBF regression | 0.00014232152 / 0.29240636097 | 0.05 | 4.16e-17 |

All eight cases passed the following independent checks:

- Extracted and executed the public `make_inputs_numpy` functions using the AST,
  then compared every generated array with the corresponding private generator.
  All arrays were identical and matched the recorded input SHA256 values.
- Recomputed kernel and problem SHA256 values and matched each answer key.
  Public problem text exactly matched its family template.
- Within each pair, public kernel files differed only in the PCG64 seed. Problem
  text and numerical tolerance were identical. Public metadata contained only
  the case name and the unverified/null status; it contained no label.
- Recomputed both references and CPU emulation. References agreed at
  `rtol=1e-10, atol=1e-12`; errors reproduced the answer keys exactly. Every
  passing case was below `0.75 * tolerance` and every failing case exceeded
  `1.25 * tolerance`.
- Each search log contained the complete formal 256-seed sweep and declared
  that model feedback was not used to select cases.

The mathematical contracts and metrics are consistent with the implementations
of the private references. case_54/case_55 use the continuous integral of the supplied FP32
coefficients, independently evaluated with Decimal80 cosine arithmetic; the
32-point implementation is an approximation. case_56/case_57 use all sixteen Fourier modes,
with independent inverse-FFT and direct compensated trigonometric references;
the Parseval omitted-energy calculation provides an additional check. case_58/case_59 use
the actual supplied FP32 SPD matrix, with independent NumPy `slogdet` and
Decimal80 elimination/determinant logarithm. case_60/case_61 evaluate distances by subtracting
the actual stored coordinates before squaring, and compare the final normalized
RBF prediction; their independent reference uses Decimal80 distances, exponentials,
and aggregation. All metric denominators match the public contracts.

No substantive correctness or answer-leakage issue was found. The Triton sources
have no apparent shape, indexing, mask, or unsupported-control-flow problem.
CPU emulation does not prove GPU identity: `tl.sin`, `tl.cos`, `tl.log`, `tl.exp`
and reduction order can differ from NumPy. The numerical margins above are large
relative to ordinary approximation differences, but actual repeated T4 execution
is still required before these labels can be used in model experiments.
