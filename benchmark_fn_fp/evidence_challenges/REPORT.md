# Evidence-audit pilot results

Generated: 2026-09-24T06:14:54.670763+00:00

Same GLM low setting and 32768 total output-token ceiling; total input tokens, dollars and GPU time are not matched.
Pilot and conditional expansion follow [PROTOCOL.md](PROTOCOL.md). All recorded attempts remain visible.

| Trial | Arm | Attempts | Outcomes | Recorded API estimate |
|---|---|---:|---|---:|
| ea_pilot_r1 | single_call | 4 | {"correct": 3, "wrong_verdict": 1} | $0.011507 |
| ea_pilot_r1 | solo | 4 | {"correct": 4} | $0.079168 |
| ea_pilot_r1 | debate | 4 | {"correct": 4} | $0.264906 |

Gate (requires evidence audit in addition to numerical results):
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

| Case | Trial | Arm | Truth | Verdict | Outcome | Trace | Issues |
|---|---|---|---|---|---|---|---|
| case_62 | ea_pilot_r1 | debate | trust | trust | correct | [trace](../traces_glm/case_62/debate/ea_pilot_r1/transcript.md) |  |
| case_62 | ea_pilot_r1 | single_call | trust | trust | correct | [trace](../traces_glm/case_62/single_call/ea_pilot_r1/transcript.md) |  |
| case_62 | ea_pilot_r1 | solo | trust | trust | correct | [trace](../traces_glm/case_62/solo/ea_pilot_r1/transcript.md) |  |
| case_63 | ea_pilot_r1 | debate | reject | reject | correct | [trace](../traces_glm/case_63/debate/ea_pilot_r1/transcript.md) |  |
| case_63 | ea_pilot_r1 | single_call | reject | reject | correct | [trace](../traces_glm/case_63/single_call/ea_pilot_r1/transcript.md) |  |
| case_63 | ea_pilot_r1 | solo | reject | reject | correct | [trace](../traces_glm/case_63/solo/ea_pilot_r1/transcript.md) |  |
| case_64 | ea_pilot_r1 | debate | trust | trust | correct | [trace](../traces_glm/case_64/debate/ea_pilot_r1/transcript.md) |  |
| case_64 | ea_pilot_r1 | single_call | trust | reject | wrong_verdict | [trace](../traces_glm/case_64/single_call/ea_pilot_r1/transcript.md) |  |
| case_64 | ea_pilot_r1 | solo | trust | trust | correct | [trace](../traces_glm/case_64/solo/ea_pilot_r1/transcript.md) |  |
| case_65 | ea_pilot_r1 | debate | reject | reject | correct | [trace](../traces_glm/case_65/debate/ea_pilot_r1/transcript.md) |  |
| case_65 | ea_pilot_r1 | single_call | reject | reject | correct | [trace](../traces_glm/case_65/single_call/ea_pilot_r1/transcript.md) |  |
| case_65 | ea_pilot_r1 | solo | reject | reject | correct | [trace](../traces_glm/case_65/solo/ea_pilot_r1/transcript.md) |  |

Recorded API estimate: $0.355581; unknown-cost attempts: 0; partial-cost attempts: 0.
This excludes Modal GPU charges and unreported failed-call usage. Budget exhaustion, abstention and transport failures are not explicit mistakes.
