# Other-methods pilot v3: case_72–case_75

Generated: 2026-09-30T17:50:47.281156+00:00

See [prospective protocol](METHODS_V3_PROTOCOL.md). All three arms share a 32768 total output-token allowance; input tokens, dollars and GPU time differ.

Trace trial names are local rN identifiers; experiment batches below use preserved original_trial metadata.

| Experiment batch | Arm | Attempts | Outcomes | Recorded API estimate |
|---|---|---:|---|---:|
| ea_methods_v3_r1 | debate | 4 | {"correct": 3, "abstention": 1} | $0.273171 |
| ea_methods_v3_r2 | debate | 4 | {"correct": 3, "abstention": 1} | $0.439262 |
| ea_methods_v3_r1 | single_call | 4 | {"wrong_verdict": 3, "correct": 1} | $0.019717 |
| ea_methods_v3_r2 | single_call | 4 | {"correct": 1, "abstention": 1, "wrong_verdict": 2} | $0.028178 |
| ea_methods_v3_r1 | solo | 4 | {"correct": 3, "wrong_verdict": 1} | $0.118328 |
| ea_methods_v3_r2 | solo | 4 | {"correct": 3, "wrong_verdict": 1} | $0.084322 |

Paired repeat gate (a positive result also requires evidence audit):
The gate counts explicit wrong-verdict corrections and reverse explicit errors. Abstention is reported separately; a positive net gate is not necessarily a gain in total accuracy.
```json
{
  "r1_activates_r2": true,
  "replicated_cases": [
    "case_74"
  ],
  "net_paired_improvements": 2,
  "all_two_round_slots_terminal": true,
  "expansion_numerical_gate": true,
  "additional_requirement": "Audit actual independent GPU evidence before expansion."
}
```

| Case | Trial | Arm | Truth | Verdict | Outcome | Evidence | Issues |
|---|---|---|---|---|---|---|---|
| case_72 | r1 | debate | trust | trust | correct | [trace](../traces_glm/case_72/debate/r1/transcript.md) |  |
| case_72 | r2 | debate | trust | trust | correct | [trace](../traces_glm/case_72/debate/r2/transcript.md) |  |
| case_72 | r1 | single_call | trust | reject | wrong_verdict | [trace](../traces_glm/case_72/single_call/r1/transcript.md) |  |
| case_72 | r2 | single_call | trust | trust | correct | [trace](../traces_glm/case_72/single_call/r2/transcript.md) |  |
| case_72 | r1 | solo | trust | trust | correct | [trace](../traces_glm/case_72/solo/r1/transcript.md) |  |
| case_72 | r2 | solo | trust | trust | correct | [trace](../traces_glm/case_72/solo/r2/transcript.md) |  |
| case_73 | r1 | debate | reject | reject | correct | [trace](../traces_glm/case_73/debate/r1/transcript.md) |  |
| case_73 | r2 | debate | reject | reject | correct | [trace](../traces_glm/case_73/debate/r2/transcript.md) |  |
| case_73 | r1 | single_call | reject | reject | correct | [trace](../traces_glm/case_73/single_call/r1/transcript.md) |  |
| case_73 | r2 | single_call | reject | needs_more_evidence | abstention | [trace](../traces_glm/case_73/single_call/r2/transcript.md) |  |
| case_73 | r1 | solo | reject | reject | correct | [trace](../traces_glm/case_73/solo/r1/transcript.md) |  |
| case_73 | r2 | solo | reject | reject | correct | [trace](../traces_glm/case_73/solo/r2/transcript.md) |  |
| case_74 | r1 | debate | trust | trust | correct | [trace](../traces_glm/case_74/debate/r1/transcript.md) |  |
| case_74 | r2 | debate | trust | trust | correct | [trace](../traces_glm/case_74/debate/r2/transcript.md) |  |
| case_74 | r1 | single_call | trust | reject | wrong_verdict | [trace](../traces_glm/case_74/single_call/r1/transcript.md) |  |
| case_74 | r2 | single_call | trust | reject | wrong_verdict | [trace](../traces_glm/case_74/single_call/r2/transcript.md) |  |
| case_74 | r1 | solo | trust | reject | wrong_verdict | [trace](../traces_glm/case_74/solo/r1/transcript.md) |  |
| case_74 | r2 | solo | trust | reject | wrong_verdict | [trace](../traces_glm/case_74/solo/r2/transcript.md) |  |
| case_75 | r1 | debate | reject | needs_more_evidence | abstention | [trace](../traces_glm/case_75/debate/r1/transcript.md) |  |
| case_75 | r2 | debate | reject | needs_more_evidence | abstention | [trace](../traces_glm/case_75/debate/r2/transcript.md) |  |
| case_75 | r1 | single_call | reject | trust | wrong_verdict | [trace](../traces_glm/case_75/single_call/r1/transcript.md) |  |
| case_75 | r2 | single_call | reject | trust | wrong_verdict | [trace](../traces_glm/case_75/single_call/r2/transcript.md) |  |
| case_75 | r1 | solo | reject | reject | correct | [trace](../traces_glm/case_75/solo/r1/transcript.md) |  |
| case_75 | r2 | solo | reject | reject | correct | [trace](../traces_glm/case_75/solo/r2/transcript.md) |  |

Recorded API estimate: $0.962978; unknown-cost attempts: 0; partial-cost attempts: 0.
Excluded: Modal GPU and unreported failed-call charges. Explicit mistakes, abstentions, exhaustion and failures are distinct outcomes.
