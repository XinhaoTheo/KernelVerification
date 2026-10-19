# Success Case Retest

Current cases 36, 41, 54 and 55 were previously 109, 114, 127 and 128. Raw traces keep the earlier IDs.

October 17, 2026. Fireworks GLM-5p3, reasoning effort `low`. One new run per case and method; all 12 runs finished with a verdict. No outcome-based retries.

**The original advantage reproduced for 36 and for Solo on 55. Case 41 no longer separated the methods. Case 54 separated Debate from Solo in this repeat. Correct verdicts and valid verification evidence are assessed separately below.**

| Case | Fixed label | No tools: previous → repeat | Solo: previous → repeat | Debate: previous → repeat |
|---|---|---|---|---|
| case_36 | Reject | Accept → [Accept](../traces_glm/case_36/single_call/r2/transcript.md) | Reject → [Reject](../traces_glm/case_36/solo/r3/transcript.md) | No verdict → [Reject](../traces_glm/case_36/debate/r3/transcript.md) |
| case_41 | Reject | Accept → [Reject](../traces_glm/case_41/single_call/r2/transcript.md) | Reject → [Reject](../traces_glm/case_41/solo/r2/transcript.md) | Reject → [Reject](../traces_glm/case_41/debate/r2/transcript.md) |
| case_54 | Reject | Accept → [Accept](../traces_glm/case_54/single_call/r2/transcript.md) | Reject → [Accept](../traces_glm/case_54/solo/r2/transcript.md) | Reject → [Reject](../traces_glm/case_54/debate/r2/transcript.md) |
| case_55 | Accept | Reject → [Reject](../traces_glm/case_55/single_call/r2/transcript.md) | Accept → [Accept](../traces_glm/case_55/solo/r2/transcript.md) | Accept → [Accept](../traces_glm/case_55/debate/r2/transcript.md) |

| Method | Correct verdicts | Probe attempts | API estimate |
|---|---:|---:|---:|
| No tools | 1/4 | 0 | $0.0462 |
| Solo + tools | 3/4 | 9 | $0.7149 |
| Debate + tools | 4/4 | 9 | $1.0430 |

The 12 evaluations made **53 model calls**, with a recorded API estimate of **$1.804152**, excluding GPU charges. The separate API availability check returned HTTP 200 and `OK` in 0.9 seconds (20 tokens; estimated $0.000037). These are estimates using the stored pricing snapshot, not invoice totals.

## What repeated

**36 — RMSNorm backward.** The no-tool model again assumed that FP32 gradient buffers made the reduction sufficiently accurate. Solo found a legal `[512,100]` input with an error ratio of **8.11746**. Debate found a legal `[512,200]` input with ratio **8.12488**; calling the row path on the same inputs gave **0.00161**, isolating the failure to the block reduction. CPU checks confirmed the decisive inputs satisfy the value, RMS and reciprocal-RMS bounds. Both tool methods rejected. Debate produced a verdict this time; previously it ran out of tokens.

**41 — Paged attention.** The no-tool model now explicitly identified `0 * NaN` from the source and correctly rejected. Solo reproduced **512/512 NaN elements** on a legal expired-cache input. Debate reproduced **32/66 query-head rows containing NaN**, and confirmed that the public input validator accepted its input. The bug reproduced, but the tools-versus-no-tools difference did not. Both tool workflows first wrote flawed probes and then corrected them; the final counterexamples are the relevant evidence.

**54 — erf.** The no-tool model again reused the original formula's error guarantee. Solo also accepted, incorrectly claiming an exhaustive FP16 test. Its probe used:

```python
hf = bits.to(torch.uint16).view(torch.int16).to(torch.float16)
```

This converts signed integer values to FP16 instead of interpreting each bit pattern as FP16. The 65,536 samples contained only **12,289 distinct integer-valued inputs**. All non-integer values, including the known failing input `0.50830078125`, were absent. The false acceptance had confidence 0.99. Debate used correct bit reinterpretation and found **1,394 violations**, with maximum error about **1.29175e-6**. Its counterexample is valid, although its claim that rounding does not contribute to failures is too strong. This repeat shows a Debate-over-Solo outcome, but Debate did not review or repair Solo's script: the runs were independent.

**55 — GELU.** The no-tool model repeated its overly pessimistic error estimate and rejected. Solo correctly enumerated **63,488 finite FP16 encodings** and accepted. It rounded the FP64 reference to FP32 before comparing, so its reported **5.36442e-7** is a difference from the rounded reference, not the independent mathematical bound **5.26772e-7** from qualification. Debate also accepted, but its evidence did not reproduce the full-domain check:

```python
xi = torch.tensor(bits.numpy(), dtype=torch.float16).cuda()
```

This numerically converts integers rather than reinterpreting their bits. After filtering infinities, its 65,520 samples contained only **7,168 distinct nonnegative integer-valued inputs**. Negative, fractional and subnormal inputs were missing. Debate's verdict matched the fixed label, but its claimed exhaustive verification was invalid. It must not be counted as a full evidence replication.

## Comparison limits

The kernel and problem files matched their original hashes. All four no-tool requests were identical to the original requests. Each case retained its original output budget, rounds, claim-scope settings and debate-closeout setting. The GPU remained T4 with PyTorch 2.8.0 and Triton 3.4.0.

Cases 54 and 55 used the same verifier fingerprint as before. Cases 36 and 41 used the current verifier, which includes later fixes; 36 also now uses the isolated case-only runner. These are documented replications with runtime differences, not exact historical replays. The provider model ID was unchanged, but no immutable provider model revision was available.

All 53 request/response captures include usage, and automated source/configuration/capture checks passed. Those checks do not validate the agents' probe logic. The enumeration bugs above show why a successful tool call, a correct final verdict, and sufficient evidence must be reported separately. Two runs per selected case still do not establish a stable general advantage.

New traces are under `traces_glm/case_*/{single_call,solo,debate}/r2`, except case 36 tool runs, which use `r3` because `r1` was a historical infrastructure failure. All previous records remain unchanged. Full settings, costs and evidence audits are in the [experiment record](private_data/experiment_success_replication_20261017.json).
