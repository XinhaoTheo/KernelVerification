# Numerical extension case_50–case_61

Generated: 2026-09-30T17:50:43.168188+00:00

This report is derived only from canonical case_50–case_61 traces. CPU-only labels are provisional; GPU verification is required for the replication gate. Missing slots are pending, not successful attempts.

| Case | Family | Truth | Evidence | CPU error | GPU max error | Tolerance |
|---|---|---|---|---:|---:|---:|
| case_50 | orthogonal_projection | trust | gpu_verified | 0.00154364948476483 | 0.00154364948476483 | 0.01 |
| case_51 | orthogonal_projection | reject | gpu_verified | 0.0604552628069095 | 0.060455264512030066 | 0.01 |
| case_52 | quantized_routing | trust | gpu_verified | 0.0 | 0.0 | 0.1 |
| case_53 | quantized_routing | reject | gpu_verified | 1.3401840023235219 | 1.3401840023235219 | 0.1 |
| case_54 | quadrature | trust | gpu_verified | 0.001552519962372429 | 0.001552519962372429 | 0.035 |
| case_55 | quadrature | reject | gpu_verified | 0.11888438877505017 | 0.11888436797488026 | 0.035 |
| case_56 | spectral_filter | trust | gpu_verified | 0.07170934896281833 | 0.0717093557382163 | 0.15 |
| case_57 | spectral_filter | reject | gpu_verified | 0.230464263032766 | 0.23046426609697562 | 0.15 |
| case_58 | logdet | trust | gpu_verified | 2.0235509963488177e-07 | 2.0235508512601344e-07 | 0.0001 |
| case_59 | logdet | reject | gpu_verified | 0.0007119199951681563 | 0.0007119199947707914 | 0.0001 |
| case_60 | distance | trust | gpu_verified | 0.0001423215199684476 | 0.00014222208321505462 | 0.05 |
| case_61 | distance | reject | gpu_verified | 0.2924063609705521 | 0.2924063609705521 | 0.05 |

## Per-trial results

Correct, explicit wrong, and no-result counts are separate. No result includes abstention, token exhaustion, or missing verdict; pending and not-started slots are shown separately. Default means the reasoning_effort field was omitted from the captured API request.

Trace trial names are local rN identifiers; experiment batches below use preserved original_trial metadata.

| Experiment batch | Arm | Provider / model | Reasoning / token cap / rounds | Attempts | Correct | Explicit wrong | No result | Pending | Not started | API estimate | Raw coverage |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| oz_default64_r1 | single_call | fireworks / accounts/fireworks/models/glm-5p3 | default / 65536 / — | 12 | 4 | 2 | 6 | 0 | 0 | $0.167357 | {"complete": 6, "partial": 6} |
| oz_low32_r1 | debate | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / 4 | 12 | 12 | 0 | 0 | 0 | 0 | $0.691501 | {"complete": 12} |
| oz_low32_r1 | single_call | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / — | 12 | 3 | 8 | 1 | 0 | 0 | $0.027576 | {"complete": 12} |
| oz_low32_r1 | solo | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / 10 | 12 | 12 | 0 | 0 | 0 | 0 | $0.206290 | {"complete": 12} |
| oz_low32_r2 | debate | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / 4 | 12 | 12 | 0 | 0 | 0 | 0 | $0.721764 | {"complete": 12} |
| oz_low32_r2 | single_call | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / — | 12 | 6 | 5 | 1 | 0 | 0 | $0.023800 | {"complete": 12} |
| oz_low32_r2 | solo | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / 10 | 12 | 12 | 0 | 0 | 0 | 0 | $0.189598 | {"complete": 12} |
| oz_low32_r3 | single_call | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / — | 12 | 4 | 6 | 2 | 0 | 0 | $0.026199 | {"complete": 12} |

Recorded API estimate across all attempts: $2.054085; unpriced attempts: 6; partial-cost attempts: 6. Raw calls: 384; saved responses: 378; responses with usage: 378.

Project profile API estimates, not invoices. Unknown or unrecorded charges and Modal GPU charges are excluded. Group accuracy is never pooled across different trial configurations.

## Fixed replication window

Overall gate: PASSED; 5 qualifying families (target: at least 3).

For each GPU-verified case, the fixed low32 r1/r2/r3 source slots must all complete under the protocol and contain at least two explicit wrong verdicts; the fixed low32 r1/r2 slots must both be correct for each tool arm. Failed, missing, exhausted, or abstaining slots are never replaced.

Exploratory replication of selected fixed workloads; not a generalization or statistical-significance claim. Default64 is reported separately and cannot qualify the low32 gate.

| Case | Status | Source explicit errors / 3 | Solo correct / 2 | Debate correct / 2 |
|---|---|---:|---:|---:|
| case_50 | passed | 3 | 2 | 2 |
| case_51 | passed | 2 | 2 | 2 |
| case_52 | passed | 3 | 2 | 2 |
| case_53 | failed | 0 | 2 | 2 |
| case_54 | failed | 1 | 2 | 2 |
| case_55 | passed | 3 | 2 | 2 |
| case_56 | failed | 1 | 2 | 2 |
| case_57 | failed | 0 | 2 | 2 |
| case_58 | failed | 0 | 2 | 2 |
| case_59 | passed | 3 | 2 | 2 |
| case_60 | failed | 1 | 2 | 2 |
| case_61 | passed | 2 | 2 | 2 |

Matched tool comparisons: 24; valid both-correct pairs: 24; debate corrects an explicit solo error: 0; solo corrects an explicit debate error: 0.

## Every recorded attempt

| Case | Trial | Arm | Status | Verdict | Outcome | API estimate | Raw coverage | Trace / protocol issues |
|---|---|---|---|---|---|---:|---|---|
| case_50 | r1 | debate | completed | trust | correct | $0.057753 | complete | [trace](../traces_glm/case_50/debate/r1/transcript.md)  |
| case_50 | r2 | debate | completed | trust | correct | $0.062025 | complete | [trace](../traces_glm/case_50/debate/r2/transcript.md)  |
| case_50 | r1 | single_call | completed | reject | wrong_verdict | $0.003510 | complete | [trace](../traces_glm/case_50/single_call/r1/transcript.md)  |
| case_50 | r2 | single_call | error | — | no_verdict | unknown | partial | [trace](../traces_glm/case_50/single_call/r2/trace_meta.json)  |
| case_50 | r3 | single_call | completed | reject | wrong_verdict | $0.003352 | complete | [trace](../traces_glm/case_50/single_call/r3/transcript.md)  |
| case_50 | r4 | single_call | completed | reject | wrong_verdict | $0.001849 | complete | [trace](../traces_glm/case_50/single_call/r4/transcript.md)  |
| case_50 | r1 | solo | completed | trust | correct | $0.016082 | complete | [trace](../traces_glm/case_50/solo/r1/transcript.md)  |
| case_50 | r2 | solo | completed | trust | correct | $0.015709 | complete | [trace](../traces_glm/case_50/solo/r2/transcript.md)  |
| case_51 | r1 | debate | completed | reject | correct | $0.060088 | complete | [trace](../traces_glm/case_51/debate/r1/transcript.md)  |
| case_51 | r2 | debate | completed | reject | correct | $0.069894 | complete | [trace](../traces_glm/case_51/debate/r2/transcript.md)  |
| case_51 | r1 | single_call | completed | reject | correct | $0.000817 | complete | [trace](../traces_glm/case_51/single_call/r1/transcript.md)  |
| case_51 | r2 | single_call | completed | reject | correct | $0.030176 | complete | [trace](../traces_glm/case_51/single_call/r2/transcript.md)  |
| case_51 | r3 | single_call | completed | trust | wrong_verdict | $0.000384 | complete | [trace](../traces_glm/case_51/single_call/r3/transcript.md)  |
| case_51 | r4 | single_call | completed | trust | wrong_verdict | $0.003499 | complete | [trace](../traces_glm/case_51/single_call/r4/transcript.md)  |
| case_51 | r1 | solo | completed | reject | correct | $0.014905 | complete | [trace](../traces_glm/case_51/solo/r1/transcript.md)  |
| case_51 | r2 | solo | completed | reject | correct | $0.012707 | complete | [trace](../traces_glm/case_51/solo/r2/transcript.md)  |
| case_52 | r1 | debate | completed | trust | correct | $0.067835 | complete | [trace](../traces_glm/case_52/debate/r1/transcript.md)  |
| case_52 | r2 | debate | completed | trust | correct | $0.051041 | complete | [trace](../traces_glm/case_52/debate/r2/transcript.md)  |
| case_52 | r1 | single_call | completed | reject | wrong_verdict | $0.001518 | complete | [trace](../traces_glm/case_52/single_call/r1/transcript.md)  |
| case_52 | r2 | single_call | completed | reject | wrong_verdict | $0.008849 | complete | [trace](../traces_glm/case_52/single_call/r2/transcript.md)  |
| case_52 | r3 | single_call | completed | reject | wrong_verdict | $0.001096 | complete | [trace](../traces_glm/case_52/single_call/r3/transcript.md)  |
| case_52 | r4 | single_call | completed | reject | wrong_verdict | $0.000953 | complete | [trace](../traces_glm/case_52/single_call/r4/transcript.md)  |
| case_52 | r1 | solo | completed | trust | correct | $0.016828 | complete | [trace](../traces_glm/case_52/solo/r1/transcript.md)  |
| case_52 | r2 | solo | completed | trust | correct | $0.014520 | complete | [trace](../traces_glm/case_52/solo/r2/transcript.md)  |
| case_53 | r1 | debate | completed | reject | correct | $0.056074 | complete | [trace](../traces_glm/case_53/debate/r1/transcript.md)  |
| case_53 | r2 | debate | completed | reject | correct | $0.054168 | complete | [trace](../traces_glm/case_53/debate/r2/transcript.md)  |
| case_53 | r1 | single_call | completed | reject | correct | $0.001471 | complete | [trace](../traces_glm/case_53/single_call/r1/transcript.md)  |
| case_53 | r2 | single_call | completed | reject | correct | $0.009904 | complete | [trace](../traces_glm/case_53/single_call/r2/transcript.md)  |
| case_53 | r3 | single_call | completed | reject | correct | $0.001243 | complete | [trace](../traces_glm/case_53/single_call/r3/transcript.md)  |
| case_53 | r4 | single_call | completed | reject | correct | $0.000862 | complete | [trace](../traces_glm/case_53/single_call/r4/transcript.md)  |
| case_53 | r1 | solo | completed | reject | correct | $0.022590 | complete | [trace](../traces_glm/case_53/solo/r1/transcript.md)  |
| case_53 | r2 | solo | completed | reject | correct | $0.016685 | complete | [trace](../traces_glm/case_53/solo/r2/transcript.md)  |
| case_54 | r1 | debate | completed | trust | correct | $0.036779 | complete | [trace](../traces_glm/case_54/debate/r1/transcript.md)  |
| case_54 | r2 | debate | completed | trust | correct | $0.075678 | complete | [trace](../traces_glm/case_54/debate/r2/transcript.md)  |
| case_54 | r1 | single_call | error | — | no_verdict | unknown | partial | [trace](../traces_glm/case_54/single_call/r1/trace_meta.json)  |
| case_54 | r2 | single_call | completed | reject | wrong_verdict | $0.002591 | complete | [trace](../traces_glm/case_54/single_call/r2/transcript.md)  |
| case_54 | r3 | single_call | completed | trust | correct | $0.000616 | complete | [trace](../traces_glm/case_54/single_call/r3/transcript.md)  |
| case_54 | r4 | single_call | completed | trust | correct | $0.003016 | complete | [trace](../traces_glm/case_54/single_call/r4/transcript.md)  |
| case_54 | r1 | solo | completed | trust | correct | $0.017070 | complete | [trace](../traces_glm/case_54/solo/r1/transcript.md)  |
| case_54 | r2 | solo | completed | trust | correct | $0.014114 | complete | [trace](../traces_glm/case_54/solo/r2/transcript.md)  |
| case_55 | r1 | debate | completed | reject | correct | $0.079228 | complete | [trace](../traces_glm/case_55/debate/r1/transcript.md)  |
| case_55 | r2 | debate | completed | reject | correct | $0.074027 | complete | [trace](../traces_glm/case_55/debate/r2/transcript.md)  |
| case_55 | r1 | single_call | error | — | no_verdict | unknown | partial | [trace](../traces_glm/case_55/single_call/r1/trace_meta.json)  |
| case_55 | r2 | single_call | completed | trust | wrong_verdict | $0.007621 | complete | [trace](../traces_glm/case_55/single_call/r2/transcript.md)  |
| case_55 | r3 | single_call | completed | trust | wrong_verdict | $0.007056 | complete | [trace](../traces_glm/case_55/single_call/r3/transcript.md)  |
| case_55 | r4 | single_call | completed | trust | wrong_verdict | $0.004390 | complete | [trace](../traces_glm/case_55/single_call/r4/transcript.md)  |
| case_55 | r1 | solo | completed | reject | correct | $0.020792 | complete | [trace](../traces_glm/case_55/solo/r1/transcript.md)  |
| case_55 | r2 | solo | completed | reject | correct | $0.016258 | complete | [trace](../traces_glm/case_55/solo/r2/transcript.md)  |
| case_56 | r1 | debate | completed | trust | correct | $0.046540 | complete | [trace](../traces_glm/case_56/debate/r1/transcript.md)  |
| case_56 | r2 | debate | completed | trust | correct | $0.047179 | complete | [trace](../traces_glm/case_56/debate/r2/transcript.md)  |
| case_56 | r1 | single_call | completed | reject | wrong_verdict | $0.001007 | complete | [trace](../traces_glm/case_56/single_call/r1/transcript.md)  |
| case_56 | r2 | single_call | completed | trust | correct | $0.042444 | complete | [trace](../traces_glm/case_56/single_call/r2/transcript.md)  |
| case_56 | r3 | single_call | completed | trust | correct | $0.002022 | complete | [trace](../traces_glm/case_56/single_call/r3/transcript.md)  |
| case_56 | r4 | single_call | completed | needs_more_evidence | abstention | $0.002739 | complete | [trace](../traces_glm/case_56/single_call/r4/transcript.md)  |
| case_56 | r1 | solo | completed | trust | correct | $0.016159 | complete | [trace](../traces_glm/case_56/solo/r1/transcript.md)  |
| case_56 | r2 | solo | completed | trust | correct | $0.020247 | complete | [trace](../traces_glm/case_56/solo/r2/transcript.md)  |
| case_57 | r1 | debate | completed | reject | correct | $0.032250 | complete | [trace](../traces_glm/case_57/debate/r1/transcript.md)  |
| case_57 | r2 | debate | completed | reject | correct | $0.048097 | complete | [trace](../traces_glm/case_57/debate/r2/transcript.md)  |
| case_57 | r1 | single_call | completed | needs_more_evidence | abstention | $0.002487 | complete | [trace](../traces_glm/case_57/single_call/r1/transcript.md)  |
| case_57 | r2 | single_call | error | — | no_verdict | unknown | partial | [trace](../traces_glm/case_57/single_call/r2/trace_meta.json)  |
| case_57 | r3 | single_call | completed | needs_more_evidence | abstention | $0.003260 | complete | [trace](../traces_glm/case_57/single_call/r3/transcript.md)  |
| case_57 | r4 | single_call | completed | needs_more_evidence | abstention | $0.005023 | complete | [trace](../traces_glm/case_57/single_call/r4/transcript.md)  |
| case_57 | r1 | solo | completed | reject | correct | $0.012442 | complete | [trace](../traces_glm/case_57/solo/r1/transcript.md)  |
| case_57 | r2 | solo | completed | reject | correct | $0.015817 | complete | [trace](../traces_glm/case_57/solo/r2/transcript.md)  |
| case_58 | r1 | debate | completed | trust | correct | $0.068535 | complete | [trace](../traces_glm/case_58/debate/r1/transcript.md)  |
| case_58 | r2 | debate | completed | trust | correct | $0.064695 | complete | [trace](../traces_glm/case_58/debate/r2/transcript.md)  |
| case_58 | r1 | single_call | completed | trust | correct | $0.000431 | complete | [trace](../traces_glm/case_58/single_call/r1/transcript.md)  |
| case_58 | r2 | single_call | completed | trust | correct | $0.000870 | complete | [trace](../traces_glm/case_58/single_call/r2/transcript.md)  |
| case_58 | r3 | single_call | completed | trust | correct | $0.000732 | complete | [trace](../traces_glm/case_58/single_call/r3/transcript.md)  |
| case_58 | r4 | single_call | error | — | no_verdict | unknown | partial | [trace](../traces_glm/case_58/single_call/r4/trace_meta.json)  |
| case_58 | r1 | solo | completed | trust | correct | $0.015561 | complete | [trace](../traces_glm/case_58/solo/r1/transcript.md)  |
| case_58 | r2 | solo | completed | trust | correct | $0.015572 | complete | [trace](../traces_glm/case_58/solo/r2/transcript.md)  |
| case_59 | r1 | debate | completed | reject | correct | $0.060670 | complete | [trace](../traces_glm/case_59/debate/r1/transcript.md)  |
| case_59 | r2 | debate | completed | reject | correct | $0.063870 | complete | [trace](../traces_glm/case_59/debate/r2/transcript.md)  |
| case_59 | r1 | single_call | completed | trust | wrong_verdict | $0.001377 | complete | [trace](../traces_glm/case_59/single_call/r1/transcript.md)  |
| case_59 | r2 | single_call | completed | trust | wrong_verdict | $0.001344 | complete | [trace](../traces_glm/case_59/single_call/r2/transcript.md)  |
| case_59 | r3 | single_call | completed | trust | wrong_verdict | $0.000743 | complete | [trace](../traces_glm/case_59/single_call/r3/transcript.md)  |
| case_59 | r4 | single_call | error | — | no_verdict | unknown | partial | [trace](../traces_glm/case_59/single_call/r4/trace_meta.json)  |
| case_59 | r1 | solo | completed | reject | correct | $0.015591 | complete | [trace](../traces_glm/case_59/solo/r1/transcript.md)  |
| case_59 | r2 | solo | completed | reject | correct | $0.015441 | complete | [trace](../traces_glm/case_59/solo/r2/transcript.md)  |
| case_60 | r1 | debate | completed | trust | correct | $0.066816 | complete | [trace](../traces_glm/case_60/debate/r1/transcript.md)  |
| case_60 | r2 | debate | completed | trust | correct | $0.072841 | complete | [trace](../traces_glm/case_60/debate/r2/transcript.md)  |
| case_60 | r1 | single_call | completed | reject | wrong_verdict | $0.003406 | complete | [trace](../traces_glm/case_60/single_call/r1/transcript.md)  |
| case_60 | r2 | single_call | completed | trust | correct | $0.000564 | complete | [trace](../traces_glm/case_60/single_call/r2/transcript.md)  |
| case_60 | r3 | single_call | completed | trust | correct | $0.000745 | complete | [trace](../traces_glm/case_60/single_call/r3/transcript.md)  |
| case_60 | r4 | single_call | completed | trust | correct | $0.029318 | complete | [trace](../traces_glm/case_60/single_call/r4/transcript.md)  |
| case_60 | r1 | solo | completed | trust | correct | $0.015597 | complete | [trace](../traces_glm/case_60/solo/r1/transcript.md)  |
| case_60 | r2 | solo | completed | trust | correct | $0.016247 | complete | [trace](../traces_glm/case_60/solo/r2/transcript.md)  |
| case_61 | r1 | debate | completed | reject | correct | $0.058933 | complete | [trace](../traces_glm/case_61/debate/r1/transcript.md)  |
| case_61 | r2 | debate | completed | reject | correct | $0.038249 | complete | [trace](../traces_glm/case_61/debate/r2/transcript.md)  |
| case_61 | r1 | single_call | completed | trust | wrong_verdict | $0.001340 | complete | [trace](../traces_glm/case_61/single_call/r1/transcript.md)  |
| case_61 | r2 | single_call | completed | reject | correct | $0.001993 | complete | [trace](../traces_glm/case_61/single_call/r2/transcript.md)  |
| case_61 | r3 | single_call | completed | trust | wrong_verdict | $0.001648 | complete | [trace](../traces_glm/case_61/single_call/r3/transcript.md)  |
| case_61 | r4 | single_call | completed | trust | wrong_verdict | $0.046666 | complete | [trace](../traces_glm/case_61/single_call/r4/transcript.md)  |
| case_61 | r1 | solo | completed | reject | correct | $0.022673 | complete | [trace](../traces_glm/case_61/solo/r1/transcript.md)  |
| case_61 | r2 | solo | completed | reject | correct | $0.016281 | complete | [trace](../traces_glm/case_61/solo/r2/transcript.md)  |
