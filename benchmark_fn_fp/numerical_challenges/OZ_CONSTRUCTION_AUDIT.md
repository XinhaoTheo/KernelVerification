# case_50–case_61 construction audit

## case_50–case_53: independent read-only review

Reviewed 2026-09-23. This section covers only `orthogonal_projection` (case_50/case_51)
and `quantized_routing` (case_52/case_53), written by a different construction agent.
No cases, answer keys, search logs, or GPU records were changed. No network,
model request, or GPU execution was performed for this review.

No correctness issue was found in the four frozen cases.

| Case | Mechanism | Frozen label | CPU error | T4 error, all 10 repeats | Budget |
|---|---|---|---:|---:|---:|
| case_50 | Normalize a nearly cancelled orthogonal residual | trust | 0.001543649485 | 0.001543649485 | 0.01 |
| case_51 | Same normalized projection | reject | 0.060455262807 | 0.060455264512 | 0.01 |
| case_52 | Quantized nearest-neighbour routing | trust | 0 | 0 | 0.1 |
| case_53 | Same approximate routing | reject | 1.340184002324 | 1.340184002324 | 0.1 |

Checks performed:

- Read both public contracts and Triton kernels. The contracts define references
  on the actual stored FP32 input arrays and give the same relative L2 metric
  and denominator floor as their oracle modules.
- For case_50/case_51, algebraically recentering `b - 1.125*u` preserves the mathematical
  projection and avoids cancellation in the reference. Recomputed FP64
  recentered references agree with the independently evaluated, uncentered
  80-digit Decimal formula to at most `5.56e-17` on the selected cases.
  The normalized residuals are nonzero. Separate FP32 operations and sequential
  sums in the emulator match the launch's disabled fusion contract. The small
  CPU/T4 difference for case_51 is about `1.71e-9` in the error metric.
- For case_52/case_53, both references use original, unquantized FP32 coordinates. FP64
  and Decimal80 squared distances choose identical embeddings. The public
  lowest-index tie rule matches both `np.argmin` and the kernel's masked
  minimum-index reduction. The final gather leaves embedding values unchanged.
- Extracted and executed each public NumPy generator without importing Torch
  or Triton. Its output equals the private generator bit for bit, has FP32
  dtype, and matches every stored input hash in the answer key and GPU record.
  Public kernel/problem hashes also match both records.
- Each pair has identical problem text and identical public source after
  replacing its seed literal. No label, reference output, or selected error
  appears in the public case files.
- Recomputed every logged candidate: 64 calibration plus 256 formal seeds for
  case_50/case_51, and 256 formal seeds for case_52/case_53. All recorded candidate dictionaries were
  reproduced exactly. The stored selected seeds match the implemented
  selection rules. The case_50/case_51 rule chooses minimum/maximum errors; case_52/case_53 chooses
  the first robust pass and fail after top-two-distance gap filtering.
- Both pairs satisfy the construction margins: pass error at most `0.75`
  times budget, fail error at least `1.25` times budget. case_52/case_53 also satisfy the
  reference gap `>1e-5` and quantized gap `>=1/64` requirements. GPU records
  preserve the same labels, all ten output hashes are identical per case,
  and input nonmutation was checked by the existing validator.

Scope: these are deliberately selected fixed-workload examples, not estimates
of accuracy on random inputs or a population sample. Construction review and
GPU label agreement do not establish any no-tool/solo/debate performance gap;
that requires the separately recorded model experiments. Threshold provenance
and the absence of model feedback are recorded in the construction logs; this
review verified their contents and reproducibility, not an external timeline.
