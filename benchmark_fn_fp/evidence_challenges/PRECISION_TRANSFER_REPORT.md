# Precision-reference transfer: case_76–case_81

Generated: 2026-09-30T17:50:49.300233+00:00

Follow the [prospective protocol](PRECISION_TRANSFER_PROTOCOL.md). This is a conditional fresh-seed confirmation within a mechanism selected on development cases.

All arms share GLM low and a 32768 cumulative output-token ceiling. Input tokens, actual spend and GPU time are not matched. Every attempted slot remains visible; extra trials cannot replace designated slots.

Trace trial names are local rN identifiers; experiment batches below use preserved original_trial metadata.

| Experiment batch | Arm | Attempts | Correct labels | Outcomes | Recorded API estimate |
|---|---|---:|---:|---|---:|
| ea_precision_transfer_r1 | debate | 6 | 6/6 | {"correct": 6} | $0.474371 |
| ea_precision_transfer_r1 | single_call | 6 | 2/6 | {"wrong_verdict": 4, "correct": 2} | $0.015287 |
| ea_precision_transfer_r1 | solo | 6 | 5/6 | {"correct": 5, "wrong_verdict": 1} | $0.128306 |

The first round activates the full second round only after all 18 slots terminate, with at least two additional correct debate labels and at least one valid explicit solo-error correction. Replication requires positive accuracy gain in both rounds and the same-case explicit correction in both.

Abstentions and failures remain in the fixed accuracy denominator and are not explicit reasoning mistakes. Numerical gates additionally require independent evidence review; this report schedules no further batch.

```json
{
  "rounds": [
    {
      "trial": "ea_precision_transfer_r1",
      "expected_slots": 18,
      "observed_slots": 18,
      "all_slots_terminal": true,
      "missing_slots": [],
      "pending_slots": [],
      "correct_labels": {
        "single_call": 2,
        "solo": 5,
        "debate": 6
      },
      "accuracy_denominator_per_arm": 6,
      "outcomes": {
        "single_call": {
          "wrong_verdict": 4,
          "correct": 2
        },
        "solo": {
          "correct": 5,
          "wrong_verdict": 1
        },
        "debate": {
          "correct": 6
        }
      },
      "debate_minus_solo_correct": 1,
      "unauditable_label_slots": [],
      "accuracy_labels_auditable": true
    },
    {
      "trial": "ea_precision_transfer_r2",
      "expected_slots": 18,
      "observed_slots": 0,
      "all_slots_terminal": false,
      "missing_slots": [
        {
          "case": "case_76",
          "arm": "single_call"
        },
        {
          "case": "case_76",
          "arm": "solo"
        },
        {
          "case": "case_76",
          "arm": "debate"
        },
        {
          "case": "case_77",
          "arm": "single_call"
        },
        {
          "case": "case_77",
          "arm": "solo"
        },
        {
          "case": "case_77",
          "arm": "debate"
        },
        {
          "case": "case_78",
          "arm": "single_call"
        },
        {
          "case": "case_78",
          "arm": "solo"
        },
        {
          "case": "case_78",
          "arm": "debate"
        },
        {
          "case": "case_79",
          "arm": "single_call"
        },
        {
          "case": "case_79",
          "arm": "solo"
        },
        {
          "case": "case_79",
          "arm": "debate"
        },
        {
          "case": "case_80",
          "arm": "single_call"
        },
        {
          "case": "case_80",
          "arm": "solo"
        },
        {
          "case": "case_80",
          "arm": "debate"
        },
        {
          "case": "case_81",
          "arm": "single_call"
        },
        {
          "case": "case_81",
          "arm": "solo"
        },
        {
          "case": "case_81",
          "arm": "debate"
        }
      ],
      "pending_slots": [],
      "correct_labels": {
        "single_call": 0,
        "solo": 0,
        "debate": 0
      },
      "accuracy_denominator_per_arm": 6,
      "outcomes": {
        "single_call": {},
        "solo": {},
        "debate": {}
      },
      "debate_minus_solo_correct": 0,
      "unauditable_label_slots": [],
      "accuracy_labels_auditable": true
    }
  ],
  "r1_activates_r2": false,
  "r1_explicit_correction_cases": [
    "case_78"
  ],
  "replicated_cases": [],
  "all_two_round_slots_terminal": false,
  "both_rounds_positive_accuracy_gain": false,
  "replicated_advantage_numerical_gate": false,
  "additional_requirement": "Independently audit the decisive runtime evidence before claiming a replicated fresh-seed advantage.",
  "scope": "Fresh seeds within an adaptively selected mechanism; no general superiority claim and no further batch is scheduled."
}
```

| Case | Trial | Arm | Truth | Verdict | Outcome | Evidence | Audit issues |
|---|---|---|---|---|---|---|---|
| case_76 | r1 | debate | trust | trust | correct | [trace](../traces_glm/case_76/debate/r1/transcript.md) |  |
| case_76 | r1 | single_call | trust | reject | wrong_verdict | [trace](../traces_glm/case_76/single_call/r1/transcript.md) |  |
| case_76 | r1 | solo | trust | trust | correct | [trace](../traces_glm/case_76/solo/r1/transcript.md) |  |
| case_77 | r1 | debate | reject | reject | correct | [trace](../traces_glm/case_77/debate/r1/transcript.md) |  |
| case_77 | r1 | single_call | reject | trust | wrong_verdict | [trace](../traces_glm/case_77/single_call/r1/transcript.md) |  |
| case_77 | r1 | solo | reject | reject | correct | [trace](../traces_glm/case_77/solo/r1/transcript.md) |  |
| case_78 | r1 | debate | trust | trust | correct | [trace](../traces_glm/case_78/debate/r1/transcript.md) |  |
| case_78 | r1 | single_call | trust | trust | correct | [trace](../traces_glm/case_78/single_call/r1/transcript.md) |  |
| case_78 | r1 | solo | trust | reject | wrong_verdict | [trace](../traces_glm/case_78/solo/r1/transcript.md) |  |
| case_79 | r1 | debate | reject | reject | correct | [trace](../traces_glm/case_79/debate/r1/transcript.md) |  |
| case_79 | r1 | single_call | reject | trust | wrong_verdict | [trace](../traces_glm/case_79/single_call/r1/transcript.md) |  |
| case_79 | r1 | solo | reject | reject | correct | [trace](../traces_glm/case_79/solo/r1/transcript.md) |  |
| case_80 | r1 | debate | trust | trust | correct | [trace](../traces_glm/case_80/debate/r1/transcript.md) |  |
| case_80 | r1 | single_call | trust | trust | correct | [trace](../traces_glm/case_80/single_call/r1/transcript.md) |  |
| case_80 | r1 | solo | trust | trust | correct | [trace](../traces_glm/case_80/solo/r1/transcript.md) |  |
| case_81 | r1 | debate | reject | reject | correct | [trace](../traces_glm/case_81/debate/r1/transcript.md) |  |
| case_81 | r1 | single_call | reject | trust | wrong_verdict | [trace](../traces_glm/case_81/single_call/r1/transcript.md) |  |
| case_81 | r1 | solo | reject | reject | correct | [trace](../traces_glm/case_81/solo/r1/transcript.md) |  |

Recorded API estimate: $0.617964; unknown-cost attempts: 0; partial-cost attempts: 0.
Excluded: Modal GPU charges and usage not returned by failed provider calls. Provider errors, budget exhaustion, abstention and explicit mistakes are separate outcomes.
