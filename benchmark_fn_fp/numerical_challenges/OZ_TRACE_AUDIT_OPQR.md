# Tool-trace audit: case_50–case_53

Updated: 2026-09-24 UTC, `oz_low32_r1` and `oz_low32_r2` collections. This is a read-only
review of saved probe code, stdout/stderr, tool events, claim evidence, source
hashes, and final verdicts. No GPU probe or API was re-executed. Both rounds are
fully reviewed for all four cases and both tool arms; other trials are outside
this audit.

## Round 1

| Case / arm | Audit status | Actual successful GPU probe | Observed error | Frozen GPU error | Verdict |
|---|---|---|---:|---:|---|
| case_50 / solo | Reviewed | `t7` | 0.00154364948338 | 0.00154364948476 | trust |
| case_50 / debate | Reviewed | `t10` | 0.00154364948338 | 0.00154364948476 | trust |
| case_51 / solo | Reviewed | `t7` | 0.060455 (printed to 6 decimals) | 0.0604552645120 | reject |
| case_51 / debate | Reviewed; diagnostic caveat below | `t12`, `t13` | 0.0604552645278 | 0.0604552645120 | reject |
| case_52 / solo | Reviewed | `t7` | 0 | 0 | trust |
| case_52 / debate | Reviewed | `t15` | 0 | 0 | trust |
| case_53 / solo | Reviewed | `t8` | 1.34018400923 | 1.34018400232 | reject |
| case_53 / debate | Reviewed | `t12` | 1.34018400232 | 1.34018400232 | reject |

All eight round-1 runs are complete and reviewed.

## Execution, reference, and input scope

All eight reviewed runs import the public `kernel.py` from the corresponding
`/root/numerical_cases/case_*` directory and call its actual GPU `run` function.
Their inputs come directly from public `make_inputs` or `make_inputs_numpy`,
with no changed seeds, injected values, or alternate workloads. The kernel and
problem text returned by `load_artifact` match the frozen source SHA256 values.
All recorded probe-code/stdout/stderr/JSON artifact hashes were checked against
the actual saved bytes. Successful decisive probes exited with code 0 and did
not time out.

case_50/case_51 references compute the specified projection in FP64 from actual stored FP32
inputs, then normalize its residual. They use the correct relative-L2 metric
and 0.01 threshold. Some comments call this reference "recentred", but the code
actually evaluates the direct formula. Its observed errors differ from the
independent frozen reference by at most about 1.6e-11, which is immaterial to
either label. case_50 debate additionally compares explicit sequential FP32 CPU
emulation to the actual GPU output and records bitwise equality.

case_52/case_53 references compute unquantized squared distances in FP64 and select the
lowest index on ties. Q selects row 6 in both paths. R selects row 9 in the
reference and row 7 in the kernel. The final metric is applied to the gathered
embedding, with denominator floor 1e-12 and tolerance 0.1. case_52 solo computes the
last norm in FP32, but equal output/reference vectors make its zero error exact.
case_53 solo converts the output to FP64 but leaves the reference norm in FP32, giving
a harmless approximately 6.9e-9 metric difference. Recomputing FP64 relative L2
from its recorded output/reference arrays gives 1.3401840023235219, matching the
frozen oracle. case_52 solo and case_53 solo recorded output vectors also reproduce the
frozen GPU output SHA256 exactly.

case_52 debate includes a CPU-only tie probe (`t13`), clearly distinguished from the
decisive actual GPU execution (`t15`). Its quantized minimum is unique; no
unobserved tie behavior is needed for the verdict.
case_53 debate similarly uses actual kernel probe `t12` for the decisive metric. Its
`t13` allocates the public input tensors on GPU but performs a host-side tie
analysis, without calling the kernel. It finds unique minima in both metrics
and correctly rebuts the tie-specific claim. The diagnostic field named
`tie_changes_selection` merely compares the two route indices and is true even
without a tie; the final claim interpretation does not repeat that misleading
field name as a causal conclusion.

## Errors and recovery

- All reviewed runs initially attempted at least one `record_claim` call without
  `scope_rationale`. The ledger rejected those calls; subsequent corrected calls
  succeeded. Final decisive claims are closed with the appropriate confirmed or
  rebutted status and refer to successful runtime evidence.
- case_51 solo prints plain text rather than JSON. case_52 solo prints a Python dictionary
  rather than JSON. Their JSON extraction warnings do not mean the probes
  failed: exit codes are 0, stdout is retained, and the agent explicitly
  interprets it when finalizing evidence.
- case_53 solo `t7` fails while formatting `sorted(...).tolist()` after execution.
  `t8` fixes the printing bug, reruns the public kernel, and supplies valid JSON
  evidence for its confirmed claim.
- case_52 debate `t12` fails by converting a CUDA tensor directly to NumPy. It does
  not supply decisive evidence. `t15` adds host copies, reruns the actual kernel,
  and correctly rebuts the failure claim with error 0.

## Non-decisive interpretation issue

case_51 debate's `t13` describes an "EXACT (fp64) alpha" ablation but explicitly casts
alpha to `np.float32` before multiplying. Its reported error 0.00409486 therefore
includes coefficient representation rounding as well as per-element product
and subtraction rounding. Calling this strictly "per-element rounding alone"
overstates what the ablation isolates. An unnecessary temporary zero array is
also overwritten before use and has no numerical effect. The final rejection
is independently justified by `t12` running the actual public GPU kernel and
measuring error 0.0604552645 against the correct reference, so this diagnostic
wording issue does not invalidate the verdict.

## Round 2

All eight `oz_low32_r2` runs are complete and reviewed.

| Case / arm | Decisive successful GPU probe | Observed error | Verdict | Complete raw API call sets |
|---|---|---:|---|---:|
| case_50 / solo | `t7` | 0.00154364948338 | trust | 5 |
| case_50 / debate | `t12` | 0.00154364948338 | trust | 9 |
| case_51 / solo | `t6` | 0.0604552645278 | reject | 4 |
| case_51 / debate | `t15` | 0.0604552645278 | reject | 10 |
| case_52 / solo | `t6` | 0 | trust | 4 |
| case_52 / debate | `t12`, `t13` | 0 | trust | 9 |
| case_53 / solo | `t7` | 1.34018399628 | reject | 5 |
| case_53 / debate | `t12` | 1.34018400232 | reject | 10 |

Every decisive probe imports the unchanged public kernel and calls its real GPU
`run`. case_50/case_51 use the correct FP64 projection and final relative-L2 contract metric.
case_52/case_53 use unquantized FP64 reference distances, lowest-index `argmin`, and the
returned embedding's relative L2. case_53 solo computes its final norms in FP32;
the resulting approximately 6.0e-9 difference from the frozen FP64 metric does
not affect its large rejection margin. Both Q runs return reference row 6, and
both R runs identify kernel row 7 versus reference row 9. All numeric results
agree with the frozen GPU labels and errors within the stated precision.

Seven runs call public input builders directly. case_51 debate reconstructs the input
generator inline with the same PCG64 seed and expressions. Independently
executing only those input assignments reproduced both frozen input SHA256
values exactly; no alternate input was used. This check did not execute the
probe or repeat the construction search.

Source SHA256 values in `trace_meta.json` and the saved artifact were checked
against the GPU freeze for all eight runs. Probe artifact bytes match every
recorded hash. All 56 raw API calls have valid request, response, and metadata
files; every metadata record is completed with `response_saved=true`, and each
response contains choices and usage. Call counts exactly match the usage-bearing
LLM entries in `run.json` history. Every request records the same GLM model,
`reasoning_effort=low`, and `max_tokens=32768`. The saved final verdicts match
`run.json`, all trace statuses are completed, and decisive claims have final
confirmed/rebutted evidence rather than unresolved probe failures.

Round-2 errors and limitations:

- case_50 solo/debate, case_51 debate, case_52 debate, and case_53 solo/debate encountered missing
  `scope_rationale` errors and successfully resubmitted the claims. case_51 solo and
  case_52 solo recorded claims successfully on their first attempt.
- case_51 debate `t13` imports a nonexistent placeholder module and fails before
  running the public kernel. `t15` removes that import and executes the actual
  public kernel successfully. Its earlier CPU diagnostic `t12` alone is not
  counted as GPU verification.
- case_53 debate's first description update has malformed tool arguments. A later
  `record_description_update` (`t8`) succeeds. Its first probe (`t10`) fails
  converting a CUDA tensor directly to NumPy; this is retained as inconclusive
  evidence. The corrected probe `t12` makes host copies, reruns the real kernel,
  and supplies the decisive confirmed evidence.
- case_50 debate `t13` computes only a CPU coefficient diagnostic even though the tool
  event requests GPU availability. Its coefficient-error threshold is not the
  public acceptance criterion. The valid final acceptance comes from actual
  kernel probe `t12` and the required output metric.
- case_51 debate again uses the FP64 coefficient rounded to FP32 in its supplemental
  residual-rounding comparison. The successful `t15` code labels that rounding
  explicitly. Interpreting the value as strictly isolated per-element rounding
  would still be too strong, but its actual-kernel rejection and separate
  coefficient-channel calculation do not depend on that interpretation.

No round-2 trace lacks the public GPU execution needed for its final verdict,
and no malformed tool call or failed probe remains the decisive evidence.
