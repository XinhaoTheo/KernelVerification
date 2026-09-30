# GLM numerical challenges

Case labels and links use the current global numbering. Verbatim historical model explanations below retain their original wording and may mention previous IDs; see the global case map.

Generated: 2026-09-23T21:44:35.632297+00:00

Replication gate: **NOT YET MET**; 0/2 qualifying mechanisms (none).

The fixed gate uses the earliest three source-only attempts and earliest two attempts per tool arm. Failures, abstentions and token exhaustion retain their slots; later successes never replace them. All attempts, including additional trials and counterexamples, appear below. See [protocol](PROTOCOL.md).

## Scope and frozen truth

Validated candidate cases: **6**; cases with recorded attempts: **6**. Initial reservations: 18 (17 finished); confirmation reservations: 24 (24 finished); additional reservations: 25 (24 finished).

Reservations include queued/running attempts and do not imply completed model calls or charges. Finished counts include failed runs. Initial exploration covers every candidate; confirmation cases are selected adaptively, so confirmation accuracy is not a held-out benchmark estimate.

| Case | Mechanism | Frozen label | Max GPU error | Budget | Kernel version |
| --- | --- | --- | ---: | ---: | --- |
| case_38 | softmax | trust | 0.008098160619 | 0.02 | `32cd0774e134` |
| case_39 | softmax | reject | 0.03652388581 | 0.02 | `ad5345994014` |
| case_40 | recurrence | trust | 0.0007690293622 | 0.002 | `f57d059d2321` |
| case_41 | recurrence | reject | 0.004907839402 | 0.002 | `8f84da257fac` |
| case_42 | reduction | reject | 0.9001987201 | 0.1 | `94b1e21a9464` |
| case_43 | reduction | trust | 0 | 0.1 | `5154e0353db1` |

Each frozen case was run ten times on T4 and checked against CPU labels, input hashes and independent FP64 references. The problem contract and inputs are public to all arms; oracle files are absent from the agent image. Source and problem hashes are retained per attempt.

### CPU construction records

| Family | Recorded CPU candidates | Selected-row records | Log |
| --- | ---: | ---: | --- |
| recurrence | 512 | — | [log](private_data/search_log_recurrence.json) |
| reduction | — | 3 | [log](private_data/search_log_reduction.json) |
| softmax | 117 | — | [log](private_data/search_log_softmax.json) |

CPU candidate counts are separate from paid model attempts. A selected-row log does not establish the number of every construction candidate explored.

## All-attempt arm totals

| Arm | Reserved | Finished | Pending | Correct | Explicit wrong | Abstention | Token cap | No verdict | API estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_call | 31 | 29 | 2 | 8 | 10 | 0 | 4 | 7 | $0.301447 |
| solo | 18 | 18 | 0 | 13 | 0 | 0 | 0 | 5 | $0.216434 |
| debate | 18 | 18 | 0 | 12 | 0 | 0 | 0 | 6 | $0.586186 |

Correct/wrong columns describe final-label outcomes; operational failures are shown separately per trial and do not satisfy the replication gate. Pending reservations are separate from terminal no-verdict outcomes. Abstention, token caps and transport failures are not explicit wrong judgments.

## Fixed-window replication gate

| Case | Family | Source wrong / first 3 | Solo correct / first 2 | Debate correct / first 2 | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| case_38 | softmax | 0/3 | 1/2 | 0/2 | failed |
| case_39 | softmax | 0/3 | 0/2 | 0/2 | failed |
| case_40 | recurrence | 0/3 | 0/2 | 0/2 | failed |
| case_41 | recurrence | 1/3 | 0/2 | 0/2 | failed |
| case_42 | reduction | 0/3 | 0/2 | 0/2 | failed |
| case_43 | reduction | 1/3 | 0/2 | 0/2 | pending |

A source window must contain three successfully finished, protocol-matching calls, at least two with explicit wrong verdicts. Both tool arms must be correct in their first two successfully finished attempts without replacing any earlier failed slot. At least two distinct mechanisms must qualify.

## Every recorded trial

| Case | Arm | Trial / phase | Status | Verdict | Outcome | In / out tokens | Coverage | API estimate | Trace |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | --- |
| case_38 | single_call | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_38/single_call/numerical_r1/trace_meta.json) |
| case_38 | single_call | numerical_r3 / confirmation | error | — | no_verdict | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_38/single_call/numerical_r3/trace_meta.json) |
| case_38 | single_call | numerical_r4 / confirmation | completed | — | token_limit | 1287 / 32768 | complete | $0.036405 | [trace](../traces_glm/case_38/single_call/numerical_r4/transcript.md) |
| case_38 | single_call | numerical_smoke8k / additional | completed | — | token_limit | 1287 / 8192 | complete | $0.009372 | [trace](../traces_glm/case_38/single_call/numerical_smoke8k/transcript.md) |
| case_38 | single_call | numerical_probe32768 / additional | completed | reject | wrong_verdict | 1287 / 14071 | complete | $0.015838 | [trace](../traces_glm/case_38/single_call/numerical_probe32768/transcript.md) |
| case_38 | single_call | numerical_low_probe / additional | completed | reject | wrong_verdict | 1287 / 1207 | complete | $0.001688 | [trace](../traces_glm/case_38/single_call/numerical_low_probe/transcript.md) |
| case_38 | single_call | numerical_low_r1 / additional | completed | trust | correct | 1287 / 1293 | complete | $0.001783 | [trace](../traces_glm/case_38/single_call/numerical_low_r1/transcript.md) |
| case_38 | single_call | numerical_low_r2 / additional | completed | reject | wrong_verdict | 1287 / 734 | complete | $0.001168 | [trace](../traces_glm/case_38/single_call/numerical_low_r2/transcript.md) |
| case_38 | solo | numerical_r1 / initial | completed | trust | correct | 54753 / 15494 | complete | $0.032374 | [trace](../traces_glm/case_38/solo/numerical_r1/transcript.md) |
| case_38 | solo | numerical_low_r1 / confirmation | completed | trust | correct | 43335 / 1096 | complete | $0.013339 | [trace](../traces_glm/case_38/solo/numerical_low_r1/transcript.md) |
| case_38 | solo | numerical_low_r2 / additional | completed | trust | correct | 52453 / 1172 | complete | $0.015976 | [trace](../traces_glm/case_38/solo/numerical_low_r2/transcript.md) |
| case_38 | debate | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_38/debate/numerical_r1/trace_meta.json) |
| case_38 | debate | numerical_low_r1 / confirmation | completed | trust | correct | 113873 / 3120 | complete | $0.035316 | [trace](../traces_glm/case_38/debate/numerical_low_r1/transcript.md) |
| case_38 | debate | numerical_low_r2 / additional | completed | trust | correct | 92768 / 2575 | complete | $0.028808 | [trace](../traces_glm/case_38/debate/numerical_low_r2/transcript.md) |
| case_39 | single_call | numerical_r1 / initial | completed | reject | correct | 1287 / 36930 | complete | $0.040983 | [trace](../traces_glm/case_39/single_call/numerical_r1/transcript.md) |
| case_39 | single_call | numerical_r3 / confirmation | error | — | no_verdict | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_39/single_call/numerical_r3/trace_meta.json) |
| case_39 | single_call | numerical_r4 / confirmation | completed | — | token_limit | 1287 / 32768 | complete | $0.036405 | [trace](../traces_glm/case_39/single_call/numerical_r4/transcript.md) |
| case_39 | single_call | numerical_low_r1 / additional | completed | reject | correct | 1287 / 3178 | complete | $0.003856 | [trace](../traces_glm/case_39/single_call/numerical_low_r1/transcript.md) |
| case_39 | single_call | numerical_low_r2 / additional | completed | trust | wrong_verdict | 1287 / 868 | complete | $0.001315 | [trace](../traces_glm/case_39/single_call/numerical_low_r2/transcript.md) |
| case_39 | solo | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_39/solo/numerical_r1/trace_meta.json) |
| case_39 | solo | numerical_low_r1 / confirmation | completed | reject | correct | 42336 / 1078 | complete | $0.013040 | [trace](../traces_glm/case_39/solo/numerical_low_r1/transcript.md) |
| case_39 | solo | numerical_low_r2 / additional | completed | reject | correct | 53409 / 1314 | complete | $0.016400 | [trace](../traces_glm/case_39/solo/numerical_low_r2/transcript.md) |
| case_39 | debate | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_39/debate/numerical_r1/trace_meta.json) |
| case_39 | debate | numerical_low_r1 / confirmation | completed | reject | correct | 210497 / 5876 | complete | $0.065403 | [trace](../traces_glm/case_39/debate/numerical_low_r1/transcript.md) |
| case_39 | debate | numerical_low_r2 / additional | completed | reject | correct | 186904 / 5256 | complete | $0.058115 | [trace](../traces_glm/case_39/debate/numerical_low_r2/transcript.md) |
| case_40 | single_call | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_40/single_call/numerical_r1/trace_meta.json) |
| case_40 | single_call | numerical_r3 / confirmation | error | — | no_verdict | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_40/single_call/numerical_r3/trace_meta.json) |
| case_40 | single_call | numerical_low_r1 / confirmation | completed | trust | correct | 1058 / 531 | complete | $0.000880 | [trace](../traces_glm/case_40/single_call/numerical_low_r1/transcript.md) |
| case_40 | single_call | numerical_r4 / additional | completed | — | token_limit | 1058 / 32768 | complete | $0.036341 | [trace](../traces_glm/case_40/single_call/numerical_r4/transcript.md) |
| case_40 | single_call | numerical_low_r2 / additional | completed | trust | correct | 1058 / 1763 | complete | $0.002236 | [trace](../traces_glm/case_40/single_call/numerical_low_r2/transcript.md) |
| case_40 | solo | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_40/solo/numerical_r1/trace_meta.json) |
| case_40 | solo | numerical_low_r1 / confirmation | completed | trust | correct | 52135 / 1460 | complete | $0.016204 | [trace](../traces_glm/case_40/solo/numerical_low_r1/transcript.md) |
| case_40 | solo | numerical_low_r2 / additional | completed | trust | correct | 52129 / 1271 | complete | $0.015994 | [trace](../traces_glm/case_40/solo/numerical_low_r2/transcript.md) |
| case_40 | debate | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_40/debate/numerical_r1/trace_meta.json) |
| case_40 | debate | numerical_low_r1 / confirmation | completed | trust | correct | 76577 / 2787 | complete | $0.024507 | [trace](../traces_glm/case_40/debate/numerical_low_r1/transcript.md) |
| case_40 | debate | numerical_low_r2 / additional | completed | trust | correct | 106262 / 3421 | complete | $0.033516 | [trace](../traces_glm/case_40/debate/numerical_low_r2/transcript.md) |
| case_41 | single_call | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_41/single_call/numerical_r1/trace_meta.json) |
| case_41 | single_call | numerical_r3 / confirmation | completed | trust | wrong_verdict | 1057 / 37584 | complete | $0.041638 | [trace](../traces_glm/case_41/single_call/numerical_r3/transcript.md) |
| case_41 | single_call | numerical_low_r1 / confirmation | completed | trust | wrong_verdict | 1057 / 3853 | complete | $0.004534 | [trace](../traces_glm/case_41/single_call/numerical_low_r1/transcript.md) |
| case_41 | single_call | numerical_low_r2 / additional | completed | trust | wrong_verdict | 1057 / 3207 | complete | $0.003824 | [trace](../traces_glm/case_41/single_call/numerical_low_r2/transcript.md) |
| case_41 | single_call | numerical_r4 / additional | running | — | pending | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_41/single_call/numerical_r4/trace_meta.json) |
| case_41 | solo | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_41/solo/numerical_r1/trace_meta.json) |
| case_41 | solo | numerical_low_r1 / confirmation | completed | reject | correct | 41337 / 1063 | complete | $0.012744 | [trace](../traces_glm/case_41/solo/numerical_low_r1/transcript.md) |
| case_41 | solo | numerical_low_r2 / additional | completed | reject | correct | 53028 / 1518 | complete | $0.016518 | [trace](../traces_glm/case_41/solo/numerical_low_r2/transcript.md) |
| case_41 | debate | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_41/debate/numerical_r1/trace_meta.json) |
| case_41 | debate | numerical_low_r1 / confirmation | completed | reject | correct | 242054 / 6299 | complete | $0.074704 | [trace](../traces_glm/case_41/debate/numerical_low_r1/transcript.md) |
| case_41 | debate | numerical_low_r2 / additional | completed | reject | correct | 190743 / 5892 | complete | $0.059889 | [trace](../traces_glm/case_41/debate/numerical_low_r2/transcript.md) |
| case_42 | single_call | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_42/single_call/numerical_r1/trace_meta.json) |
| case_42 | single_call | numerical_r3 / confirmation | completed | reject | correct | 1202 / 22650 | complete | $0.025252 | [trace](../traces_glm/case_42/single_call/numerical_r3/transcript.md) |
| case_42 | single_call | numerical_low_r1 / confirmation | completed | reject | correct | 1202 / 2340 | complete | $0.002911 | [trace](../traces_glm/case_42/single_call/numerical_low_r1/transcript.md) |
| case_42 | single_call | numerical_low_r2 / additional | completed | reject | correct | 1202 / 2104 | complete | $0.002651 | [trace](../traces_glm/case_42/single_call/numerical_low_r2/transcript.md) |
| case_42 | solo | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_42/solo/numerical_r1/trace_meta.json) |
| case_42 | solo | numerical_low_r1 / confirmation | completed | reject | correct | 54040 / 1544 | complete | $0.016830 | [trace](../traces_glm/case_42/solo/numerical_low_r1/transcript.md) |
| case_42 | solo | numerical_low_r2 / additional | completed | reject | correct | 54271 / 1594 | complete | $0.016949 | [trace](../traces_glm/case_42/solo/numerical_low_r2/transcript.md) |
| case_42 | debate | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_42/debate/numerical_r1/trace_meta.json) |
| case_42 | debate | numerical_low_r1 / confirmation | completed | reject | correct | 90823 / 2829 | complete | $0.028542 | [trace](../traces_glm/case_42/debate/numerical_low_r1/transcript.md) |
| case_42 | debate | numerical_low_r2 / additional | completed | reject | correct | 195651 / 5601 | complete | $0.060943 | [trace](../traces_glm/case_42/debate/numerical_low_r2/transcript.md) |
| case_43 | single_call | numerical_r1 / initial | running | — | pending | 0 / 0 | partial | unknown (partial) | [trace](../traces_glm/case_43/single_call/numerical_r1/trace_meta.json) |
| case_43 | single_call | numerical_r3 / confirmation | completed | reject | wrong_verdict | 1202 / 26413 | complete | $0.029391 | [trace](../traces_glm/case_43/single_call/numerical_r3/transcript.md) |
| case_43 | single_call | numerical_low_r1 / confirmation | completed | reject | wrong_verdict | 1202 / 1436 | complete | $0.001916 | [trace](../traces_glm/case_43/single_call/numerical_low_r1/transcript.md) |
| case_43 | single_call | numerical_low_r2 / additional | completed | reject | wrong_verdict | 1202 / 658 | complete | $0.001060 | [trace](../traces_glm/case_43/single_call/numerical_low_r2/transcript.md) |
| case_43 | solo | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_43/solo/numerical_r1/trace_meta.json) |
| case_43 | solo | numerical_low_r1 / confirmation | completed | trust | correct | 43020 / 1560 | complete | $0.013762 | [trace](../traces_glm/case_43/solo/numerical_low_r1/transcript.md) |
| case_43 | solo | numerical_low_r2 / additional | completed | trust | correct | 52629 / 1425 | complete | $0.016304 | [trace](../traces_glm/case_43/solo/numerical_low_r2/transcript.md) |
| case_43 | debate | numerical_r1 / initial | error | — | no_verdict | 0 / 0 | unknown | unknown | [trace](../traces_glm/case_43/debate/numerical_r1/trace_meta.json) |
| case_43 | debate | numerical_low_r1 / confirmation | completed | trust | correct | 183016 / 5756 | complete | $0.057576 | [trace](../traces_glm/case_43/debate/numerical_low_r1/transcript.md) |
| case_43 | debate | numerical_low_r2 / additional | completed | trust | correct | 186393 / 6070 | complete | $0.058867 | [trace](../traces_glm/case_43/debate/numerical_low_r2/transcript.md) |

## Capture and cost limits

Recorded API estimate: **$1.104067**. Complete captured usage contributes $1.104067; partially captured usage contributes $0.000000.

These are project-profile token estimates, not billing invoices. Missing responses may have incurred unknown charges; Modal GPU charges are not included. Legacy/unknown coverage cannot certify a complete bill.

- Unknown cost: case_38/single_call/numerical_r1, case_38/single_call/numerical_r3, case_38/debate/numerical_r1, case_39/single_call/numerical_r3, case_39/solo/numerical_r1, case_39/debate/numerical_r1, case_40/single_call/numerical_r1, case_40/single_call/numerical_r3, case_40/solo/numerical_r1, case_40/debate/numerical_r1, case_41/single_call/numerical_r1, case_41/single_call/numerical_r4, case_41/solo/numerical_r1, case_41/debate/numerical_r1, case_42/single_call/numerical_r1, case_42/solo/numerical_r1, case_42/debate/numerical_r1, case_43/single_call/numerical_r1, case_43/solo/numerical_r1, case_43/debate/numerical_r1.
- Partial API usage: case_38/single_call/numerical_r1, case_38/single_call/numerical_r3, case_39/single_call/numerical_r3, case_40/single_call/numerical_r1, case_40/single_call/numerical_r3, case_41/single_call/numerical_r1, case_41/single_call/numerical_r4, case_42/single_call/numerical_r1, case_43/single_call/numerical_r1.
- Unknown capture coverage: case_38/debate/numerical_r1, case_39/solo/numerical_r1, case_39/debate/numerical_r1, case_40/solo/numerical_r1, case_40/debate/numerical_r1, case_41/solo/numerical_r1, case_41/debate/numerical_r1, case_42/solo/numerical_r1, case_42/debate/numerical_r1, case_43/solo/numerical_r1, case_43/debate/numerical_r1.

## Tool-arm comparison

Matched case/trial/model comparisons: 18. Both correct: 0; debate correct where solo explicitly wrong: 0; solo correct where debate explicitly wrong: 0.

Both tool arms succeeding supports tool benefit relative to source-only mistakes; it does not establish an extra correctness benefit from debate. More explanation or more probes do not count as accuracy gains. Compute and total-token budgets are not equalized, so even an observed difference does not isolate debate itself.

## Per-trial notes

### case_38/single_call/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: APIConnectionError: see llm_calls/*/error.json for sanitized details

Captured provider failure (01790190586704550000_7c5a69c56e4b429bba44bf094a99a91a): Connection error.

Requests without captured responses: 01790190586704550000_7c5a69c56e4b429bba44bf094a99a91a

### case_38/single_call/numerical_r3

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: InternalServerError: see llm_calls/*/error.json for sanitized details

Captured provider failure (01790197715561185000_ff65a2c723124d1487d6013398034f36): Error code: 504 - {'error': {'message': 'Gateway timeout', 'param': None, 'code': 'GATEWAY_TIMEOUT', 'type': 'error'}, 'request_id': 'chatcmpl-33446bed0fc24560984f7a42c951b373'}

Requests without captured responses: 01790197715561185000_ff65a2c723124d1487d6013398034f36

### case_38/single_call/numerical_r4

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 32768; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

No final explanation recorded.

Protocol issues: max_tokens differs from 65536

API finish reasons: length.

### case_38/single_call/numerical_smoke8k

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

No final explanation recorded.

Protocol issues: max_tokens differs from 65536

API finish reasons: length.

### case_38/single_call/numerical_probe32768

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 32768; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel gratuitously quantizes the logits to a coarse 15-level grid (step = max|logits|/7 ≈ 0.45–0.5, per-logit rounding error up to ~±0.23), which perturbs softmax weights by factors up to e^0.23 ≈ 1.26; for this workload (128 logits ~ N(0, 1.2²), values ~ N(1,1), ‖ref‖ ≈ 4) the resulting relative L2 error is expected around 2.5%, exceeding the 0.02 tolerance, and the contract explicitly defines the reference on the original unquantized logits.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_38/single_call/numerical_low_probe

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel quantizes each logit to ~15 levels (scale = max|logits|/7), introducing per-logit rounding errors of up to ~0.2 in log-space, whereas the contract requires softmax on the original logits without quantization. For this workload (logits std 1.2, max|logit| ≈ 3) the induced perturbation of p[i] is on the order of e^0.2 ≈ 20% per element, giving an estimated relative L2 error around or above the 0.02 tolerance.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_38/single_call/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel computes softmax over the returned logits (with symmetric uniform quantization of step max|logits|/7, giving per-logit rounding error ≤ max|logits|/14 ≈ 0.25) applied to the returned values tensor, matching the reference contract on the supplied inputs. The random rounding perturbations largely cancel in the probability-weighted sum, yielding an estimated relative L2 error well under the 0.02 tolerance.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_38/single_call/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel quantizes each logit to a grid of step max|logits|/7 (~0.48 for this workload), introducing per-logit errors up to ~0.24 before the softmax; such perturbations change attention weights by factors of e^±0.24 (~±27%), so the output's relative L2 error will far exceed the 0.02 tolerance required against the unquantized float64 reference.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_38/solo/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 10; captured API calls: 4; responses with usage: 4; probes: 1.

Probe t6 ran the real kernel on the exact in-scope make_inputs() workload (logits (128,) fp32, values (128,16) fp32 contiguous, GPU, exit 0, no stderr/timeout) and computed the contract's float64 unquantized-softmax reference. Results: output is a finite float32 vector of shape (16,); relative L2 error ||output - reference||_2 / max(||reference||_2, 0.001*sqrt(16)) = 0.0080982 (denominator 4.23137), well within the required <= 0.02. Source inspection confirms the kernel quantizes logits to a 15-level grid (scale = max|logits|/7, codes clamped to [-7,7]) before the softmax, but the probe shows it implements that math faithfully (agreement with an independent float64 evaluation of the same quantization: 3.6e-8 rel L2), so its entire deviation from the contract reference is the quantization-induced error, which measures 0.0081 on this fixed workload — inside the tolerance the contract itself declares. The contract imposes no other requirements (fixed workload, output shape (16,), finiteness, error bound), all directly satisfied. In-scope claim c1 (that the error exceeds 0.02) is rebutted by this evidence; no confirmed in-scope defect exists, so the kernel satisfies the contract.

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls.

### case_38/solo/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 4; responses with usage: 4; probes: 1.

Ran the kernel on the exact make_inputs() workload (probe t6): output shape (16,), all finite, relative L2 error vs the float64 unquantized reference = 0.008098, comfortably within the contract's 0.02 threshold (max abs err 0.0176, ref norm 4.231). The only deviation from the reference — logit quantization to a 7-step grid — stays within tolerance on the fixed in-scope workload, and the contract explicitly permits this level of error. No other failure modes apply to a fixed single-workload contract.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls.

### case_38/solo/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 5; responses with usage: 5; probes: 1.

Ran the fixed make_inputs() workload on GPU (probe t7): kernel output is finite, shape (16,), and relative L2 error vs the float64 unquantized reference is 0.0081, within the contract's 0.02 tolerance. The only hypothesized defect (7-level logit quantization breaking accuracy, claim c1) was rebutted by runtime evidence. No other contract requirements (shape, finiteness, tolerance) are violated.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_38/debate/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 4; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: Function call was cancelled by user or a failure.

### case_38/debate/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 7; responses with usage: 7; probes: 2.

Contract (problem.txt) requires only that the output on the fixed make_inputs() (PCG64 seed 918233) workload stay within 0.02 relative L2 of the unquantized float64 softmax reference and be a finite (16,) vector. The decisive runtime probe (tool event t8, evidence c1.e1) ran the kernel on exactly those inputs and measured kernel_rel_l2 = 0.0081 (tolerance 0.02), output finite, shape (16,), with the fp64 quantization-only simulation matching (0.0081), confirming the deviation stems from the intentional 15-level logit quantization but stays comfortably within the declared tolerance budget. Claim c1 (error exceeding 0.02) is rebutted. The contract's explicit 0.02 tolerance clause makes the quantization deviation acceptable; robustness to other tensors is out of scope per the contract. Skeptic reviewed the latest evidence and recorded no new claims; no open claims remain.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_38/debate/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 6; responses with usage: 6; probes: 1.

The only material in-scope claim, c1, was rebutted by a runtime probe on the exact make_inputs() workload (seed 918233): the kernel's output vs the fp64 unquantized reference has relative L2 error 0.00810, comfortably below the contract's 0.02 bound (max abs err 0.0176, ref norm 4.231), with finite output of shape (16,). The contract in problem.txt scopes correctness solely to this fixed seeded workload and tolerates the kernel's logits quantization as long as the <=0.02 relative L2 bound is met, which it is. No open claims remain and the Skeptic recorded no new claims after reviewing all evidence.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_39/single_call/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel replaces the true logits with a 15-level (codes clamped to ±7) symmetric quantization at scale max|logits|/7 (≈0.45 for these N(0,1.2) logits, i.e. up to ~0.23 rounding error per logit) before the softmax, while the contract compares against the un-quantized reference. This perturbs the softmax weights by ~10-15% relative, giving an expected output relative-L2 error of roughly 2.2-2.5% for this workload — above the 0.02 tolerance — so the int4-range quantization shortcut is a real numerical defect rather than an equivalent implementation.

API finish reasons: stop.

### case_39/single_call/numerical_r3

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: InternalServerError: see llm_calls/*/error.json for sanitized details

Captured provider failure (01790197715561255000_26ece4732abd4d09849886b9e550e4cb): Error code: 504 - {'error': {'message': 'Gateway timeout', 'param': None, 'code': 'GATEWAY_TIMEOUT', 'type': 'error'}, 'request_id': 'chatcmpl-83658db6d6df496c8a3e4c03d81b8198'}

Requests without captured responses: 01790197715561255000_26ece4732abd4d09849886b9e550e4cb

### case_39/single_call/numerical_r4

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 32768; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

No final explanation recorded.

Protocol issues: max_tokens differs from 65536

API finish reasons: length.

### case_39/single_call/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel quantizes logits to ~4-bit codes (step = max|logits|/7, rounding error std ≈ 0.14–0.15 per logit) before the softmax, while the reference uses unquantized logits. With 128 logits of std 1.2, this perturbs the output by an expected relative L2 error of roughly 0.025–0.03 (≈ rounding-std × sqrt(Σp²) ≈ 0.15 × 0.18), which exceeds the required 0.02 tolerance for this workload; the rest of the kernel (loads, softmax, weighted sum, store) is correct, but the quantization is the defect.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_39/single_call/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel quantizes logits to a symmetric 15-level grid with step max|logits|/7 (~0.47), giving per-logit rounding error up to ~0.24 but typically ~0.14 std; these errors are roughly independent across the 128 rows and the output is a probability-weighted average over 16 output dims, so the expected relative L2 error is well under the 2% tolerance, and the softmax normalization plus fp32 math is otherwise exact.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_39/solo/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 10; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: Deadline exceeded

### case_39/solo/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 4; responses with usage: 4; probes: 1.

Ran the kernel on the exact fixed make_inputs() workload (probe t6) and computed the contract's metric: ||output - float64 unquantized reference||_2 / max(||ref||_2, 0.04) = 0.03652, exceeding the required tolerance of 0.02. The output was finite and correctly shaped, but the kernel quantizes logits to a 15-level grid (scale = max|logits|/7) before softmax, while the contract requires the reference computed from the original unquantized logits. This confirmed, in-scope violation of the explicit numerical contract (claim c1) justifies rejection.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls.

### case_39/solo/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 5; responses with usage: 5; probes: 1.

Probe t7 ran the kernel on the exact make_inputs() workload from kernel.py and compared its output to the contract's float64 unquantized softmax reference. The kernel quantizes logits to 15 levels (codes in [-7,7], scale = max|logits|/7) before softmax, while the contract requires matching the unquantized reference within relative L2 error <= 0.02. Measured relative L2 error was 0.03652 (ref_norm 4.081, out_norm 3.948), nearly double the tolerance. Output shape and finiteness are correct, but the numeric contract is violated. Claim c1 is confirmed, in_scope, and backed by contract scope evidence.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_39/debate/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 4; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: Function call was cancelled by user or a failure.

### case_39/debate/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 9; responses with usage: 9; probes: 2.

problem.txt (the operative contract; no test.py/get_inputs narrowing beyond it) requires the kernel's (16,) output to be within relative L2 error <= 0.02 of the unquantized float64 softmax reference on the exact tensors from make_inputs(). Probe t12 ran the actual Triton kernel on that exact fixed seeded workload and measured relative L2 error 0.03652 — 1.83x over the budget — with finite, correctly shaped output, so this is not a reference-undefined or metric artifact. Probe t13 attributes the full error to the sole deviation surface: the kernel quantizes logits to 14 distinct symmetric codes (scale = max|logits|/7 ≈ 0.439) before softmax, collapsing the top four logits (2.716, 2.609, 2.564, 2.562) to one code 2.634 with differential rounding (top rounds down, runners-up round up), redistributing probability mass (L1 shift 0.0877); an fp64 simulation of the quantized path alone reproduces the entire 0.03652 error, so fp32-vs-fp64 rounding and the values permutation are not factors. The contract's clause "The reference uses the original input logits, without quantization" makes this deviation a defect rather than an acceptable declared lossy format. No unresolved claims remain; Skeptic reviewed the final evidence and recorded no new concerns.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_39/debate/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 9; responses with usage: 9; probes: 2.

The contract (problem.txt, the operative contract for this artifact) requires softmax attention on the original unquantized logits, evaluated against an fp64 reference with relative L2 error <= 0.02 on the single fixed make_inputs() workload (PCG64 seed 918233, fixed value-row permutation). The kernel instead quantizes logits to a ±7-code grid (scale = max|logits|/7, kernel.py lines 12-14) before softmax. Probe t12 reproduced the exact fixed workload and measured the kernel output's relative L2 error against the fp64 unquantized reference as 0.0365 — exceeding the 0.02 tolerance — with finite output, quantization step 0.439, and max probability shift 0.0077. This is a confirmed, in-scope violation of an explicitly required numeric contract, not a mere accumulation-order or tolerance artifact: probe t13 isolated fp32 accumulation, showing it contributes only ~1.8e-7 relative L2, so the entire error budget is consumed by the quantization deviation alone (c2 rebutted as immaterial). The problem.txt's tolerance clause (<= 0.02 relative L2 vs the unquantized fp64 reference) makes this deviation a defect despite the otherwise clean softmax implementation. Skeptic reviewed the final evidence (t16) and recorded no new claims; no open claims remain. Verdict applies only to the single fixed input, as the contract states.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_40/single_call/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: APIConnectionError: see llm_calls/*/error.json for sanitized details

Captured provider failure (01790191063420125000_2c573919d0694d818096c66199e8fd21): Connection error.

Requests without captured responses: 01790191063420125000_2c573919d0694d818096c66199e8fd21

### case_40/single_call/numerical_r3

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: InternalServerError: see llm_calls/*/error.json for sanitized details

Captured provider failure (01790198315862764000_911059312e494a9586975ea3f13d6dd4): Error code: 504 - {'error': {'message': 'Gateway timeout', 'param': None, 'code': 'GATEWAY_TIMEOUT', 'type': 'error'}, 'request_id': 'chatcmpl-21d5712957f84e05a587daa00f231564'}

Requests without captured responses: 01790198315862764000_911059312e494a9586975ea3f13d6dd4

### case_40/single_call/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel computes the recurrence in fp32 but rounds the state to fp16 each step. The matrix eigenvalues are ~0.875–0.97, so past rounding errors decay geometrically; the accumulated fp16 quantization error is roughly 2^-12 * sqrt(1/(1-λ²)) ≈ a few times 1e-4 relative, comfortably within the 0.002 tolerance.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_40/single_call/numerical_r4

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 32768; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

No final explanation recorded.

Protocol issues: max_tokens differs from 65536

API finish reasons: length.

### case_40/single_call/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The only approximation is rounding the state to fp16 each step, but the matrix is an orthogonal conjugation of an upper-triangular matrix with spectral radius ≤ 0.96875, so quantization errors (relative ~2.4e-4) decay geometrically and accumulate to roughly ~1e-3 relative L2 error, within the 0.002 tolerance on this workload; the fp32 matvec and indexing are otherwise correct.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_40/solo/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 10; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: Function call was cancelled by user or a failure.

### case_40/solo/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 5; responses with usage: 5; probes: 1.

Ran the actual kernel via run(*make_inputs()) on GPU and compared to the float64 recurrence reference defined in problem.txt (probe event t7, evidence c1.e1). Output: finite, shape (16,), relative L2 error 0.000769 (tolerance 0.002), max elementwise error 0.300 on a reference norm of 511. The fp16-intermediate hypothesis was rebutted; the kernel satisfies the contract's sole numeric requirement on the fixed in-scope workload.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_40/solo/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 5; responses with usage: 5; probes: 1.

Ran run(*make_inputs()) on the real GPU against a float64 reference of the 64-step recurrence (probe event t7). Output was finite, shape (16,), with relative L2 error 0.000769 versus tolerance 0.002 (reference norm 511.05, abs error 0.393). The only risk identified — fp16 rounding of intermediate states at line 16 accumulating over 64 steps — was rebutted: the contract explicitly permits approximating intermediate states, and the measured final error stays comfortably within the required 0.002 bound. The kernel satisfies the contract on the specified workload.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_40/debate/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 4; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: function fu-jSjt2RbkNJ2fQmCc9XYcal is stopped

### case_40/debate/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 6; responses with usage: 6; probes: 1.

The contract scopes judgment to run(*make_inputs()) on the fixed seeded workload with relative L2 error <= 0.002 vs the float64 reference. The sole in-scope claim c1 — that per-step fp16 state quantization accumulated over 64 steps exceeds the 0.002 tolerance — was directly rebutted by a GPU runtime probe on the exact contract inputs: output is finite, shape (16,), with relative L2 error 7.69e-4 (roughly 2.6x margin below tolerance). The contract explicitly allows intermediate approximation, so per-step fp16 rounding is acceptable by design; only the final error matters. No open claims remain and the Skeptic signed off with no new concerns after reviewing the probe evidence.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_40/debate/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 6; responses with usage: 6; probes: 2.

The contract (problem.txt) requires only that run(*make_inputs()) produce a finite float32 vector of shape (16,) with relative L2 error <= 0.002 versus the float64 64-step reference on the single fixed workload. The two raised in-scope claims (fp16 per-step quantization accumulation, and fp32 accumulation-order error compounding with fp16) were both directly measured on that exact workload: measured relative L2 error 0.000769, tolerance 0.002, output finite, shape (16,), float32. A decomposition probe reproduces the kernel's arithmetic in numpy (identical error value) and shows the fp32-only path contributes ~4.7e-7, confirming the error is dominated by the permitted intermediate fp16 approximation and stays within the contract's budget. The contract explicitly permits intermediate approximation, so this is acceptable under that clause. No open claims remain, and the Skeptic reviewed the latest evidence and recorded no new claims.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_41/single_call/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: APIConnectionError: see llm_calls/*/error.json for sanitized details

Captured provider failure (01790191990479792000_de4da952445e48b2b52a37b55331b53c): Connection error.

Requests without captured responses: 01790191990479792000_de4da952445e48b2b52a37b55331b53c

### case_41/single_call/numerical_r3

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The only real concern is the per-step float16 rounding of the state (relative error up to ~2^-11 per step), but the workload's matrix is an orthogonal conjugation of an upper-triangular matrix with spectrum in [0.875, 0.96875], so the recurrence is a contraction and rounding errors accumulate incoherently rather than compounding: a steady-state error analysis (errors amplified by A^k with variance-weighted time constants 1/(1-λ²), λ≤0.96875) gives an expected relative L2 error of roughly 7e-4 to 1.3e-3 even allowing for non-normal transient growth, comfortably under the 0.002 tolerance; indexing, dtypes, and output shape are all correct, and the contract explicitly permits approximating intermediate states.

API finish reasons: stop.

### case_41/single_call/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel computes the recurrence correctly in fp32 and only quantizes the state to fp16 each step (unit roundoff ~4.9e-4). Since the reference itself permits intermediate approximation and the contraction factors (0.875–0.969) limit error accumulation, the expected relative L2 error is roughly (2^-11/√3)/√(1-g²) ≈ 1e-3, below the 0.002 tolerance; values stay well within fp16 normal range.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_41/single_call/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel implements the recurrence correctly (fp32 matvec with drive, correct indexing), and the only approximation is rounding the state to fp16 each step. With fp16 half-ulp relative error ~2.8e-4 RMS per step and the contraction of the matrix (spectral radius ~0.97, modest non-normal transient growth), the accumulated relative L2 error should stay around or below 1e-3, within the 0.002 tolerance.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_41/single_call/numerical_r4

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 32768; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Protocol issues: max_tokens differs from 65536

Requests without captured responses: 01790199874133582000_6f1a26aa56a148c1aac531ea2b7449b1

### case_41/solo/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 10; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: function fu-jSjt2RbkNJ2fQmCc9XYcal is stopped

### case_41/solo/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 4; responses with usage: 4; probes: 1.

Probe t6 ran run(*make_inputs()) on the actual GPU workload and compared to the float64 reference defined in problem.txt. Output was finite with shape (16,), but relative L2 error = 0.004908 (num 0.7971, den 162.414), exceeding the contract's bound of 0.002. This is caused by the kernel's per-step float16 quantization of the intermediate state (kernel.py line 16) over 64 recurrence steps, accumulating error beyond what the contract tolerates on this fixed workload. Claim c1 is in_scope and confirmed; the kernel violates the numerical contract.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls.

### case_41/solo/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 5; responses with usage: 5; probes: 1.

Probe t7 executed run(*make_inputs()) on the real GPU and compared to the float64 reference computed from the same float32 inputs. Output was finite, shape (16,), norm_out=161.92 vs norm_ref=162.41, max_abs_err=0.632, relative L2 error = 0.004908, which exceeds the contract's explicit tolerance of 0.002 for this exact fixed workload (problem.txt). The kernel rounds each intermediate state to fp16 (kernel.py line 16) over 64 recurrence steps, and this accumulated rounding is the cause. Contract violation is in_scope and confirmed; reject.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_41/debate/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 4; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: function fu-jSjt2RbkNJ2fQmCc9XYcal is stopped

### case_41/debate/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 11; responses with usage: 11; probes: 2.

Claim c2 is confirmed, in_scope, and backed by scope evidence tied to the contract's own input domain: problem.txt fixes the exact seeded workload from make_inputs() and requires run(*make_inputs()) to produce a finite (16,) float32 output with relative L2 error at most 0.002 versus the float64 64-step recurrence reference. Probes t8 and t13 (GPU, exact seeded workload) both measured the actual kernel at relative L2 0.0049078 — ~2.45x the tolerance — while the identical variant without the fp16 intermediate cast achieves 1.11e-06 (claim c1, confirmed), and a float64 simulation rounding each intermediate state to fp16 reproduces the kernel error exactly. This isolates the fp16 quantization at kernel.py line 16 (.to(tl.float16).to(tl.float32)) as the sole and sufficient error source. Note the contract explicitly permits intermediate-state approximation but binds the FINAL output error to <= 0.002; since the final output violates that explicit numeric requirement on the contract's only in-scope workload, this is a defect, not an acceptable approximation. The only caveat (probe hardware not verified to be exactly a T4) is immaterial given the ~2.5x margin and deterministic fp16 round-to-nearest behavior. Skeptic reviewed the latest evidence and recorded no new claims; all claims have evidence.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_41/debate/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 9; responses with usage: 9; probes: 2.

The sole in-scope input is run(*make_inputs()) on the fixed seeded workload, and the contract explicitly requires final relative L2 error <= 0.002 versus the float64 64-step reference (denominator max(||ref||_2, 0.004)). Runtime probe t12 measured the kernel output on exactly those inputs: relative L2 error 0.004908 (2.45x the tolerance), output finite, shape (16,), denominator ||ref||_2 = 162.41 (floor not active, per probe t13, which rebutted c2's floor-denominator alternative). The float64 fp16-round-trip emulation reproduces the kernel error exactly (0.004908), isolating the per-step fp16 cast at kernel.py line 16 as the cause. The contract permits approximate intermediate states, but only if the final tolerance is met; here it is exceeded by a wide margin, so this is a confirmed in-scope contract violation. Skeptic reviewed the latest evidence and recorded no new claims.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_42/single_call/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: APIConnectionError: see llm_calls/*/error.json for sanitized details

Captured provider failure (01790191990479623000_732513362c6642078f650b04b4128385): Connection error.

Requests without captured responses: 01790191990479623000_732513362c6642078f650b04b4128385

### case_42/single_call/numerical_r3

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

Each row's exact sum is just the 64 small values (0.25–0.75, total ≈16–48) since the ±w entries (multiples of 2^20 up to 2^25) cancel exactly, but the kernel's naive sequential float32 accumulation passes through partial sums of magnitude ~2^26–2^28 where ulp is 8–32, so small addends are rounded away whenever the partial sum exceeds ~2^23 (an absorbing state: later 0.25–0.75 adds can't move a grid-aligned accumulator). The output per row is only the few smalls absorbed during rare low-magnitude windows, versus a reference of ~32, giving relative error E on the order of 0.5–1.0, far above the required 0.1.

API finish reasons: stop.

### case_42/single_call/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The workload is a cancellation trap: rows contain exact ±(1..32)*2^20 pairs plus tiny 0.25-granularity values, so the true row sums are ~8-32 while intermediate float32 accumulators reach ~2^25-2^29. The kernel does a naive sequential float32 sum, so each 0.25/0.5/0.75 addition when the accumulator is large gets rounded at ulp up to 4-32, producing per-row errors of the same order as the reference itself, likely violating E <= 0.1.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_42/single_call/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel performs a naive sequential float32 accumulation over an adversarially permuted input where large ±2^20-scale terms cancel exactly, leaving a tiny ~0.25-granular residue. While the big terms stay exact in fp32, adding the small 0.25-multiples while the accumulator is still ~2^24-2^26 (ulp 2-8) irrevocably rounds away low bits, producing per-row errors of several ulps against a reference of magnitude ~32; the expected relative error E exceeds the 0.1 tolerance. A float64 or compensated accumulator would be needed.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_42/solo/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 10; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: function fu-jSjt2RbkNJ2fQmCc9XYcal is stopped

### case_42/solo/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 5; responses with usage: 5; probes: 1.

Ran the supplied Triton kernel on the exact specified workload (kernel.make_inputs, seed 730119, (64,128) fp32, public permutation) on the GPU and compared to a float64 reference (tool event t7, evidence c1.e1). Result: all outputs finite but E = ||out-ref||_2 / max(||ref||_2, 0.008) = 0.9002, far above the contract limit of 0.1 (max abs error 35.75 against reference row values ~28.5–35.75). The contract explicitly requires E <= 0.1 on this fully specified input; the kernel's sequential float32 accumulation of large cancelling terms (~±33.5e6) destroys accuracy. Reject.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_42/solo/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 5; responses with usage: 5; probes: 1.

Ran the kernel on the exact fully specified workload (PCG64 seed 730119, published column permutations, contiguous float32 (64,128) input) and compared against a float64 reference (tool event t7). All outputs are finite, but E = ||out-ref||_2 / max(||ref||_2, 0.008) = 0.9002, far above the contract limit of 0.1. The kernel sums rows sequentially in float32; the ±w terms (~1e7-3e7) cancel exactly in real arithmetic leaving small sums (~30), but during the sequential sweep the float32 accumulator reaches ~1e9 where the 0.25-0.75 addends fall below the ULP and are lost (e.g., output 6.0 vs reference 30.5, max abs error 35.75, 100% relative per-row error). The contract explicitly requires E <= 0.1 on this specified input and does not accept float32 error beyond that, so this is an in-scope violation. Verdict: reject.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_42/debate/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 4; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: function fu-jSjt2RbkNJ2fQmCc9XYcal is stopped

### case_42/debate/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 6; responses with usage: 6; probes: 1.

Claim c1 is confirmed and in_scope: a direct runtime probe on the exact specified workload (run(make_inputs('cuda')), PCG64 seed 730119, public column permutation) compared the kernel's output against the contractually mandated float64 per-row reference sum. The contract in problem.txt explicitly requires E = ||out-ref||_2 / max(||ref||_2, 0.008) <= 0.1 with finite outputs; the measured E = 0.9002 (max per-row absolute error 35.75 vs reference row sums of 28.5-35.75), a ~9x violation. Outputs are finite, so finiteness is met, but the explicit numeric clause is not. The mechanism matches the source: the kernel does strictly sequential left-to-right fp32 accumulation (enable_fp_fusion=False, fp32 accumulator), and once the running sum is on the order of tens of millions its fp32 ulp is 2-4, so the 0.25-granularity small terms are rounded away entirely. The contract permits fp32 arithmetic and any summation order only "if this numerical requirement is met" — it is not, so this is a defect, not a rounding-artifact allowance. No unresolved claims remain; the Skeptic reviewed the latest evidence and recorded no new claims.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_42/debate/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 9; responses with usage: 9; probes: 2.

Confirmed in-scope claim c1: a runtime probe ran the actual Triton kernel on the exact specified make_inputs() workload and computed the contract metric E = ||output - float64 per-row reference||_2 / max(||reference||_2, 0.008) = 0.9002, far exceeding the contract bound E <= 0.1 (problem.txt explicitly requires this on exactly this workload; only the final result on the specified input is in scope, so the scope evidence from problem.txt plus kernel.py's public generator is fully binding). Outputs are finite, so finiteness holds, but the numeric tolerance is violated by 9x with max abs row error 35.75. The mechanism claim c2 is also confirmed: an fp32 simulation of the identical ascending-order sequential accumulation reproduces the kernel's error exactly (sim E 0.9002, same max row err 35.75), showing 3298/4096 small entries absorbed when partial sums reach 2.26e8 (ulp ~16 >> 0.75). The contract explicitly permits float32 arithmetic and any summation order only "if this numerical requirement is met"; it is not met, so this is a genuine defect, not an acceptable rounding/order deviation. No open claims remain; the Skeptic reviewed the latest evidence and recorded no new claims.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_43/single_call/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 0; probes: 0.

No final explanation recorded.

Requests without captured responses: 01790195697781950000_96103ece57374c15a90438ea977fcc0a

### case_43/single_call/numerical_r3

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The kernel does a strictly sequential float32 accumulation over the (randomly shuffled) columns. Each row's true sum (~16–48, made entirely of the 0.25–0.75 'small' terms) is dwarfed by intermediate partial sums of the ±c·2^20 terms, whose running imbalance typically reaches ~2^25–2^26 mid-row; once |accumulator| ≥ 2^24 the ULP is ≥ 2, so each 0.25–0.75 addition rounds away completely (absorbed). With the 64 w-terms spread over the 128 positions (last one near position ~126) and only the handful of smalls at the very start/end surviving, each row loses ~25–30 of its ~32 true sum, giving ||out−ref||₂ ≈ 200–250 vs the allowed ~25.6 (||ref||₂ ≈ 256), i.e. E ≈ 0.7–1.0 ≫ 0.1 — a structural ~10× violation, not a borderline effect, so the sequential-fp32 kernel fails the stated contract on this exact workload.

API finish reasons: stop.

### case_43/single_call/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The workload is adversarially constructed for catastrophic cancellation: each row contains exact ±large values (up to 2^25) plus small 0.25-step values whose sum (~32/row) constitutes the entire reference, all interleaved by a fixed permutation. The kernel does a plain sequential float32 accumulation with no fp fusion; once the running accumulator reaches ~1e8 (after a few large terms, ulp ≈ 8-64), the small terms are lost entirely, and partial sums of the large integers themselves round since they exceed 2^24. The resulting error per row is on the order of the reference magnitude, so E ≫ 0.1.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_43/single_call/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: —; captured API calls: 1; responses with usage: 1; probes: 0.

The input is deliberately adversarial: each row contains pairs of ±large values (up to ~3.5e7) that cancel exactly, leaving a true sum of only ~32 from the small terms. The kernel does a naive sequential float32 accumulation, whose intermediate partial sums reach ~1e9 (ulp ~64), so accumulated rounding error per row can be on the order of hundreds—far exceeding the allowed relative error E ≤ 0.1 against the reference norm of ~256.

Protocol issues: max_tokens differs from 65536

API finish reasons: stop.

### case_43/solo/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 10; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: function fu-jSjt2RbkNJ2fQmCc9XYcal is stopped

### case_43/solo/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 4; responses with usage: 4; probes: 1.

Ran the kernel on the exact fully specified workload (kernel.py make_inputs, seed 730119, (64,128) float32 matrix) and compared against a float64 reference (tool event t6). Output was finite for all 64 rows, max absolute error 0.0, E = ||out-ref||_2 / max(||ref||_2, 0.008) = 0.0, far below the contract's E <= 0.1 threshold. The large +w/-w terms cancel exactly (they are exact float32 opposites), so the sequential float32 sum is exact on this data. The hypothesis that catastrophic cancellation would lose the small terms was rebutted. Contract satisfied on the specified input, which is the entire scope.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls.

### case_43/solo/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 10; captured API calls: 5; responses with usage: 5; probes: 1.

Ran the fully specified workload (make_inputs with seed 730119 and the published column permutation) through the kernel on GPU and compared against a float64 row-sum reference (probe t7). Result: all outputs finite, E = ||out-ref||_2 / max(||ref||_2, 0.008) = 0.0, max abs error 0.0, ref norm 257.13 — the fp32 sequential accumulation is exact on this input because every value is an exactly representable multiple of 0.25 or 2^20·integer and the large ±1048576 pairs cancel losslessly at fp32 precision. The contract explicitly allows float32 arithmetic and any summation order provided E <= 0.1, which is satisfied with zero error. Claim c1 was raised in_scope against the E<=0.1/finiteness requirement and is rebutted by direct runtime evidence.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_43/debate/numerical_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 65536; max rounds: 4; captured API calls: 0; responses with usage: 0; probes: 0.

No final explanation recorded.

Runner failure: function fu-jSjt2RbkNJ2fQmCc9XYcal is stopped

### case_43/debate/numerical_low_r1

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 9; responses with usage: 9; probes: 2.

The contract covers only the single specified seed-730119 workload with E = ||out-ref||_2/max(||ref||_2,0.008) <= 0.1 and all-finite outputs. Probe t12 shows the kernel's exact left-to-right fp32 sequential summation on the exact double-permuted input matrix matches the float64 (Kahan cross-checked) reference exactly (E = 0.0, max_abs_err = 0.0, zero rows exceeding tolerance), rebutting the catastrophic-cancellation concern (c1) — all large ±w terms are integer multiples of 2^20 and cancel exactly. Probe t13 confirms (c2) the actual GPU run of run(*make_inputs()) returned a finite (64,) float32 output bit-identical to the fp32 sequential emulation with zero deviation, so the contract's finiteness and accuracy clauses are both satisfied on the in-scope input. No open claims remain; skeptic reviewed the latest evidence and found no new claims.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

### case_43/debate/numerical_low_r2

Model: `accounts/fireworks/models/glm-5p3`; provider: fireworks; max tokens: 8192; max rounds: 4; captured API calls: 9; responses with usage: 9; probes: 1.

The operative contract (problem.txt) requires only that run(*make_inputs()) on the fixed seeded (64,128) workload produce finite outputs with E = ||out - float64 ref||_2 / max(||ref||_2, 0.008) <= 0.1, and explicitly permits fp32 arithmetic and any summation order. The sole in-scope failure hypothesis (c1: sequential fp32 accumulation absorbing the 0.25-0.75 small terms into large 2^20-scale partial sums, driving E > 0.1) was tested by a direct GPU probe on the exact seeded input (tool event t10): kernel output is bit-identical to the float64 row-sum reference, E = 0.0, max/mean per-row absolute error 0.0, all outputs finite, ||ref||_2 = 257.13, and the GPU result matches a CPU sequential-fp32 simulation of the same column order (no reassociation/fusion). The contract's explicit allowance of fp32 arithmetic and arbitrary summation order makes any benign rounding acceptable; in fact none was observed. No open claims remain, and the Skeptic signed off after reviewing the latest evidence with no new claims. Verdict: trust.

Protocol issues: max_tokens differs from 65536

API finish reasons: tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls, tool_calls.

## Interpretation limits

This selected synthetic dataset tests finite-workload numerical compliance. Replication uses the same frozen cases and measures repeatability, not performance on independent unseen cases. The gate is exploratory and does not establish statistical significance or broad generalization.

Later tool counterexamples after the fixed gate windows: case_38/solo/numerical_low_r2, case_38/debate/numerical_low_r2, case_39/solo/numerical_low_r2, case_39/debate/numerical_low_r2, case_40/solo/numerical_low_r2, case_40/debate/numerical_low_r2, case_41/solo/numerical_low_r2, case_41/debate/numerical_low_r2, case_42/solo/numerical_low_r2, case_42/debate/numerical_low_r2, case_43/solo/numerical_low_r2, case_43/debate/numerical_low_r2.
