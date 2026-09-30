# Other-methods pilot v2: case_66–case_71

Generated: 2026-09-24T07:08:40.399944+00:00

See [prospective protocol](METHODS_V2_PROTOCOL.md). All three arms share a 32768 total output-token allowance; input tokens, dollars and GPU time differ.

| Trial | Arm | Attempts | Outcomes | Recorded API estimate |
|---|---|---:|---|---:|
| ea_methods_v2_r1 | single_call | 6 | {"abstention": 1, "wrong_verdict": 2, "correct": 3} | $0.010024 |
| ea_methods_v2_r1 | solo | 6 | {"correct": 6} | $0.127509 |
| ea_methods_v2_r1 | debate | 6 | {"correct": 6} | $0.455780 |

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
| case_66 | ea_methods_v2_r1 | debate | trust | trust | correct | [trace](../traces_glm/case_66/debate/ea_methods_v2_r1/transcript.md) |  |
| case_66 | ea_methods_v2_r1 | single_call | trust | needs_more_evidence | abstention | [trace](../traces_glm/case_66/single_call/ea_methods_v2_r1/transcript.md) |  |
| case_66 | ea_methods_v2_r1 | solo | trust | trust | correct | [trace](../traces_glm/case_66/solo/ea_methods_v2_r1/transcript.md) |  |
| case_67 | ea_methods_v2_r1 | debate | reject | reject | correct | [trace](../traces_glm/case_67/debate/ea_methods_v2_r1/transcript.md) |  |
| case_67 | ea_methods_v2_r1 | single_call | reject | trust | wrong_verdict | [trace](../traces_glm/case_67/single_call/ea_methods_v2_r1/transcript.md) |  |
| case_67 | ea_methods_v2_r1 | solo | reject | reject | correct | [trace](../traces_glm/case_67/solo/ea_methods_v2_r1/transcript.md) |  |
| case_68 | ea_methods_v2_r1 | debate | trust | trust | correct | [trace](../traces_glm/case_68/debate/ea_methods_v2_r1/transcript.md) |  |
| case_68 | ea_methods_v2_r1 | single_call | trust | trust | correct | [trace](../traces_glm/case_68/single_call/ea_methods_v2_r1/transcript.md) |  |
| case_68 | ea_methods_v2_r1 | solo | trust | trust | correct | [trace](../traces_glm/case_68/solo/ea_methods_v2_r1/transcript.md) |  |
| case_69 | ea_methods_v2_r1 | debate | reject | reject | correct | [trace](../traces_glm/case_69/debate/ea_methods_v2_r1/transcript.md) |  |
| case_69 | ea_methods_v2_r1 | single_call | reject | reject | correct | [trace](../traces_glm/case_69/single_call/ea_methods_v2_r1/transcript.md) |  |
| case_69 | ea_methods_v2_r1 | solo | reject | reject | correct | [trace](../traces_glm/case_69/solo/ea_methods_v2_r1/transcript.md) |  |
| case_70 | ea_methods_v2_r1 | debate | trust | trust | correct | [trace](../traces_glm/case_70/debate/ea_methods_v2_r1/transcript.md) |  |
| case_70 | ea_methods_v2_r1 | single_call | trust | reject | wrong_verdict | [trace](../traces_glm/case_70/single_call/ea_methods_v2_r1/transcript.md) |  |
| case_70 | ea_methods_v2_r1 | solo | trust | trust | correct | [trace](../traces_glm/case_70/solo/ea_methods_v2_r1/transcript.md) |  |
| case_71 | ea_methods_v2_r1 | debate | reject | reject | correct | [trace](../traces_glm/case_71/debate/ea_methods_v2_r1/transcript.md) |  |
| case_71 | ea_methods_v2_r1 | single_call | reject | reject | correct | [trace](../traces_glm/case_71/single_call/ea_methods_v2_r1/transcript.md) |  |
| case_71 | ea_methods_v2_r1 | solo | reject | reject | correct | [trace](../traces_glm/case_71/solo/ea_methods_v2_r1/transcript.md) |  |

Recorded API estimate: $0.593313; unknown-cost attempts: 0; partial-cost attempts: 0.
Excluded: Modal GPU and unreported failed-call charges. Explicit mistakes, abstentions, exhaustion and failures are distinct outcomes.
