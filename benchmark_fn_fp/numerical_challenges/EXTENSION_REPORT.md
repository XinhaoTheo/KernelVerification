# Numerical extension case_44–case_49

Generated: 2026-09-30T17:50:42.313485+00:00

This report is derived only from canonical case_44–case_49 traces. CPU-only labels are provisional; GPU verification is required for the replication gate. Missing slots are pending, not successful attempts.

| Case | Family | Truth | Evidence | CPU error | GPU max error | Tolerance |
|---|---|---|---|---:|---:|---:|
| case_44 | variance | trust | gpu_verified | 0.00024263532316837978 | 0.00024261613799325 | 0.02 |
| case_45 | variance | reject | gpu_verified | 0.144819518713021 | 0.14481964583989307 | 0.02 |
| case_46 | iterative_solve | trust | gpu_verified | 0.023636882701893187 | 0.023636854556236476 | 0.08 |
| case_47 | iterative_solve | reject | gpu_verified | 0.12440403980323361 | 0.12440399903958174 | 0.08 |
| case_48 | polynomial | trust | gpu_verified | 3.3728673707072835e-05 | 3.3728673707072835e-05 | 0.0002 |
| case_49 | polynomial | reject | gpu_verified | 0.0004729252165482065 | 0.0004729252165482065 | 0.0002 |

## Per-trial results

Correct, explicit wrong, and no-result counts are separate. No result includes abstention, token exhaustion, or missing verdict; pending and not-started slots are shown separately. Default means the reasoning_effort field was omitted from the captured API request.

Trace trial names are local rN identifiers; experiment batches below use preserved original_trial metadata.

| Experiment batch | Arm | Provider / model | Reasoning / token cap / rounds | Attempts | Correct | Explicit wrong | No result | Pending | Not started | API estimate | Raw coverage |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| extension_default64_r1 | single_call | fireworks / accounts/fireworks/models/glm-5p3 | default / 65536 / — | 6 | 2 | 2 | 2 | 0 | 0 | $0.142321 | {"complete": 4, "partial": 2} |
| extension_low32_r1 | debate | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / 4 | 6 | 6 | 0 | 0 | 0 | 0 | $0.357550 | {"complete": 6} |
| extension_low32_r1 | single_call | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / — | 6 | 3 | 3 | 0 | 0 | 0 | $0.014530 | {"complete": 6} |
| extension_low32_r1 | solo | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / 10 | 6 | 6 | 0 | 0 | 0 | 0 | $0.090318 | {"complete": 6} |
| extension_low32_r2 | debate | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / 4 | 6 | 6 | 0 | 0 | 0 | 0 | $0.320481 | {"complete": 6} |
| extension_low32_r2 | single_call | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / — | 6 | 3 | 3 | 0 | 0 | 0 | $0.016824 | {"complete": 6} |
| extension_low32_r2 | solo | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / 10 | 6 | 6 | 0 | 0 | 0 | 0 | $0.098714 | {"complete": 6} |
| extension_low32_r3 | single_call | fireworks / accounts/fireworks/models/glm-5p3 | low / 32768 / — | 6 | 5 | 1 | 0 | 0 | 0 | $0.014874 | {"complete": 6} |

Recorded API estimate across all attempts: $1.055612; unpriced attempts: 2; partial-cost attempts: 2. Raw calls: 184; saved responses: 182; responses with usage: 182.

Project profile API estimates, not invoices. Unknown or unrecorded charges and Modal GPU charges are excluded. Group accuracy is never pooled across different trial configurations.

## Fixed replication window

Overall gate: PASSED; 2 qualifying families (target: at least 2).

For each GPU-verified case, the fixed low32 r1/r2/r3 source slots must all complete under the protocol and contain at least two explicit wrong verdicts; the fixed low32 r1/r2 slots must both be correct for each tool arm. Failed, missing, exhausted, or abstaining slots are never replaced.

Exploratory replication of selected fixed workloads; not a generalization or statistical-significance claim. Default64 is reported separately and cannot qualify the low32 gate.

| Case | Status | Source explicit errors / 3 | Solo correct / 2 | Debate correct / 2 |
|---|---|---:|---:|---:|
| case_44 | passed | 3 | 2 | 2 |
| case_45 | failed | 0 | 2 | 2 |
| case_46 | failed | 1 | 2 | 2 |
| case_47 | failed | 0 | 2 | 2 |
| case_48 | passed | 2 | 2 | 2 |
| case_49 | failed | 1 | 2 | 2 |

Matched tool comparisons: 12; valid both-correct pairs: 12; debate corrects an explicit solo error: 0; solo corrects an explicit debate error: 0.

## Every recorded attempt

| Case | Trial | Arm | Status | Verdict | Outcome | API estimate | Raw coverage | Trace / protocol issues |
|---|---|---|---|---|---|---:|---|---|
| case_44 | r1 | debate | completed | trust | correct | $0.060771 | complete | [trace](../traces_glm/case_44/debate/r1/transcript.md)  |
| case_44 | r2 | debate | completed | trust | correct | $0.042331 | complete | [trace](../traces_glm/case_44/debate/r2/transcript.md)  |
| case_44 | r1 | single_call | completed | reject | wrong_verdict | $0.000583 | complete | [trace](../traces_glm/case_44/single_call/r1/transcript.md)  |
| case_44 | r2 | single_call | completed | reject | wrong_verdict | $0.002322 | complete | [trace](../traces_glm/case_44/single_call/r2/transcript.md)  |
| case_44 | r3 | single_call | completed | reject | wrong_verdict | $0.028056 | complete | [trace](../traces_glm/case_44/single_call/r3/transcript.md)  |
| case_44 | r4 | single_call | completed | reject | wrong_verdict | $0.002860 | complete | [trace](../traces_glm/case_44/single_call/r4/transcript.md)  |
| case_44 | r1 | solo | completed | trust | correct | $0.015025 | complete | [trace](../traces_glm/case_44/solo/r1/transcript.md)  |
| case_44 | r2 | solo | completed | trust | correct | $0.014727 | complete | [trace](../traces_glm/case_44/solo/r2/transcript.md)  |
| case_45 | r1 | debate | completed | reject | correct | $0.059954 | complete | [trace](../traces_glm/case_45/debate/r1/transcript.md)  |
| case_45 | r2 | debate | completed | reject | correct | $0.034313 | complete | [trace](../traces_glm/case_45/debate/r2/transcript.md)  |
| case_45 | r1 | single_call | completed | reject | correct | $0.001958 | complete | [trace](../traces_glm/case_45/single_call/r1/transcript.md)  |
| case_45 | r2 | single_call | completed | reject | correct | $0.002179 | complete | [trace](../traces_glm/case_45/single_call/r2/transcript.md)  |
| case_45 | r3 | single_call | completed | reject | correct | $0.048121 | complete | [trace](../traces_glm/case_45/single_call/r3/transcript.md)  |
| case_45 | r4 | single_call | completed | reject | correct | $0.001448 | complete | [trace](../traces_glm/case_45/single_call/r4/transcript.md)  |
| case_45 | r1 | solo | completed | reject | correct | $0.015745 | complete | [trace](../traces_glm/case_45/solo/r1/transcript.md)  |
| case_45 | r2 | solo | completed | reject | correct | $0.016387 | complete | [trace](../traces_glm/case_45/solo/r2/transcript.md)  |
| case_46 | r1 | debate | completed | trust | correct | $0.064723 | complete | [trace](../traces_glm/case_46/debate/r1/transcript.md)  |
| case_46 | r2 | debate | completed | trust | correct | $0.054726 | complete | [trace](../traces_glm/case_46/debate/r2/transcript.md)  |
| case_46 | r1 | single_call | completed | reject | wrong_verdict | $0.003186 | complete | [trace](../traces_glm/case_46/single_call/r1/transcript.md)  |
| case_46 | r2 | single_call | completed | reject | wrong_verdict | $0.036318 | complete | [trace](../traces_glm/case_46/single_call/r2/transcript.md)  |
| case_46 | r3 | single_call | completed | trust | correct | $0.001400 | complete | [trace](../traces_glm/case_46/single_call/r3/transcript.md)  |
| case_46 | r4 | single_call | completed | trust | correct | $0.000887 | complete | [trace](../traces_glm/case_46/single_call/r4/transcript.md)  |
| case_46 | r1 | solo | completed | trust | correct | $0.012300 | complete | [trace](../traces_glm/case_46/solo/r1/transcript.md)  |
| case_46 | r2 | solo | completed | trust | correct | $0.016405 | complete | [trace](../traces_glm/case_46/solo/r2/transcript.md)  |
| case_47 | r1 | debate | completed | reject | correct | $0.063264 | complete | [trace](../traces_glm/case_47/debate/r1/transcript.md)  |
| case_47 | r2 | debate | completed | reject | correct | $0.058952 | complete | [trace](../traces_glm/case_47/debate/r2/transcript.md)  |
| case_47 | r1 | single_call | completed | reject | correct | $0.001924 | complete | [trace](../traces_glm/case_47/single_call/r1/transcript.md)  |
| case_47 | r2 | single_call | completed | reject | correct | $0.029826 | complete | [trace](../traces_glm/case_47/single_call/r2/transcript.md)  |
| case_47 | r3 | single_call | completed | reject | correct | $0.004703 | complete | [trace](../traces_glm/case_47/single_call/r3/transcript.md)  |
| case_47 | r4 | single_call | completed | reject | correct | $0.001653 | complete | [trace](../traces_glm/case_47/single_call/r4/transcript.md)  |
| case_47 | r1 | solo | completed | reject | correct | $0.016235 | complete | [trace](../traces_glm/case_47/solo/r1/transcript.md)  |
| case_47 | r2 | solo | completed | reject | correct | $0.016795 | complete | [trace](../traces_glm/case_47/solo/r2/transcript.md)  |
| case_48 | r1 | debate | completed | trust | correct | $0.071507 | complete | [trace](../traces_glm/case_48/debate/r1/transcript.md)  |
| case_48 | r2 | debate | completed | trust | correct | $0.048843 | complete | [trace](../traces_glm/case_48/debate/r2/transcript.md)  |
| case_48 | r1 | single_call | completed | reject | wrong_verdict | $0.002791 | complete | [trace](../traces_glm/case_48/single_call/r1/transcript.md)  |
| case_48 | r2 | single_call | error | — | no_verdict | unknown | partial | [trace](../traces_glm/case_48/single_call/r2/trace_meta.json)  |
| case_48 | r3 | single_call | completed | reject | wrong_verdict | $0.004500 | complete | [trace](../traces_glm/case_48/single_call/r3/transcript.md)  |
| case_48 | r4 | single_call | completed | trust | correct | $0.002693 | complete | [trace](../traces_glm/case_48/single_call/r4/transcript.md)  |
| case_48 | r1 | solo | completed | trust | correct | $0.014356 | complete | [trace](../traces_glm/case_48/solo/r1/transcript.md)  |
| case_48 | r2 | solo | completed | trust | correct | $0.017396 | complete | [trace](../traces_glm/case_48/solo/r2/transcript.md)  |
| case_49 | r1 | debate | completed | reject | correct | $0.037331 | complete | [trace](../traces_glm/case_49/debate/r1/transcript.md)  |
| case_49 | r2 | debate | completed | reject | correct | $0.081316 | complete | [trace](../traces_glm/case_49/debate/r2/transcript.md)  |
| case_49 | r1 | single_call | completed | reject | correct | $0.004088 | complete | [trace](../traces_glm/case_49/single_call/r1/transcript.md)  |
| case_49 | r2 | single_call | error | — | no_verdict | unknown | partial | [trace](../traces_glm/case_49/single_call/r2/trace_meta.json)  |
| case_49 | r3 | single_call | completed | trust | wrong_verdict | $0.001720 | complete | [trace](../traces_glm/case_49/single_call/r3/transcript.md)  |
| case_49 | r4 | single_call | completed | reject | correct | $0.005333 | complete | [trace](../traces_glm/case_49/single_call/r4/transcript.md)  |
| case_49 | r1 | solo | completed | reject | correct | $0.016657 | complete | [trace](../traces_glm/case_49/solo/r1/transcript.md)  |
| case_49 | r2 | solo | completed | reject | correct | $0.017004 | complete | [trace](../traces_glm/case_49/solo/r2/transcript.md)  |
