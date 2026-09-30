# Other-methods pilot v2: case_66–case_71

Generated: 2026-09-30T17:50:45.051987+00:00

See [prospective protocol](METHODS_V2_PROTOCOL.md). All three arms share a 32768 total output-token allowance; input tokens, dollars and GPU time differ.

Trace trial names are local rN identifiers; experiment batches below use preserved original_trial metadata.

| Experiment batch | Arm | Attempts | Outcomes | Recorded API estimate |
|---|---|---:|---|---:|
| ea_methods_v2_r1 | debate | 6 | {"correct": 6} | $0.455780 |
| ea_methods_v2_r1 | single_call | 6 | {"abstention": 1, "wrong_verdict": 2, "correct": 3} | $0.010024 |
| ea_methods_v2_r1 | solo | 6 | {"correct": 6} | $0.127509 |

Paired repeat gate (a positive result also requires evidence audit):
The gate counts explicit wrong-verdict corrections and reverse explicit errors. Abstention is reported separately; a positive net gate is not necessarily a gain in total accuracy.
```json
{
  "r1_activates_r2": false,
  "replicated_cases": [],
  "net_paired_improvements": 0,
  "all_two_round_slots_terminal": false,
  "expansion_numerical_gate": false,
  "additional_requirement": "Audit actual independent GPU evidence before expansion."
}
```

| Case | Trial | Arm | Truth | Verdict | Outcome | Evidence | Issues |
|---|---|---|---|---|---|---|---|
| case_66 | r1 | debate | trust | trust | correct | [trace](../traces_glm/case_66/debate/r1/transcript.md) |  |
| case_66 | r1 | single_call | trust | needs_more_evidence | abstention | [trace](../traces_glm/case_66/single_call/r1/transcript.md) |  |
| case_66 | r1 | solo | trust | trust | correct | [trace](../traces_glm/case_66/solo/r1/transcript.md) |  |
| case_67 | r1 | debate | reject | reject | correct | [trace](../traces_glm/case_67/debate/r1/transcript.md) |  |
| case_67 | r1 | single_call | reject | trust | wrong_verdict | [trace](../traces_glm/case_67/single_call/r1/transcript.md) |  |
| case_67 | r1 | solo | reject | reject | correct | [trace](../traces_glm/case_67/solo/r1/transcript.md) |  |
| case_68 | r1 | debate | trust | trust | correct | [trace](../traces_glm/case_68/debate/r1/transcript.md) |  |
| case_68 | r1 | single_call | trust | trust | correct | [trace](../traces_glm/case_68/single_call/r1/transcript.md) |  |
| case_68 | r1 | solo | trust | trust | correct | [trace](../traces_glm/case_68/solo/r1/transcript.md) |  |
| case_69 | r1 | debate | reject | reject | correct | [trace](../traces_glm/case_69/debate/r1/transcript.md) |  |
| case_69 | r1 | single_call | reject | reject | correct | [trace](../traces_glm/case_69/single_call/r1/transcript.md) |  |
| case_69 | r1 | solo | reject | reject | correct | [trace](../traces_glm/case_69/solo/r1/transcript.md) |  |
| case_70 | r1 | debate | trust | trust | correct | [trace](../traces_glm/case_70/debate/r1/transcript.md) |  |
| case_70 | r1 | single_call | trust | reject | wrong_verdict | [trace](../traces_glm/case_70/single_call/r1/transcript.md) |  |
| case_70 | r1 | solo | trust | trust | correct | [trace](../traces_glm/case_70/solo/r1/transcript.md) |  |
| case_71 | r1 | debate | reject | reject | correct | [trace](../traces_glm/case_71/debate/r1/transcript.md) |  |
| case_71 | r1 | single_call | reject | reject | correct | [trace](../traces_glm/case_71/single_call/r1/transcript.md) |  |
| case_71 | r1 | solo | reject | reject | correct | [trace](../traces_glm/case_71/solo/r1/transcript.md) |  |

Recorded API estimate: $0.593313; unknown-cost attempts: 0; partial-cost attempts: 0.
Excluded: Modal GPU and unreported failed-call charges. Explicit mistakes, abstentions, exhaustion and failures are distinct outcomes.
