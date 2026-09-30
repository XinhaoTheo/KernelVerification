# Evidence-audit pilot: prospective protocol

Storage update (2026-09-30): Batch labels and repeat shorthand below retain their original experimental meaning; original labels are saved in `trace_meta.json.original_trial`. Canonical links use per-case/arm `rN` folders; see [trace naming and migration](../TRACES.md).

Recorded 2026-09-24 before any model evaluation of these cases. This is a small
development experiment, followed by conditional expansion, not a claim that
debate is superior. Existing case_36–case_61 cases and their protocols remain unchanged.

## Scope and construction

Four initial cases: case_62/case_63 audit a common-mode numerical reference; case_64/case_65
audit finite state-operation coverage. Each family contains a compliant and a
noncompliant workload. Public kernel.py embeds an executable initial_probe.
The contract states the actual fixed or finite domain and does not grant this
initial tester authority to redefine that domain. The real initial-probe output
is appended to problem.txt after GPU preview and before the final source freeze.
All three arms see the same code, contract, and initial measurements. Output
numbers are never fabricated. Private construction oracles and answer keys are
not mounted into the agent image. The private oracle independently checks the
reference and complete declared domain and repeats decisive GPU measurements.
All CPU candidates and failed construction/validation attempts are retained.

## Three arms and resource settings

- single_call: original source-only prompt, no execution tools, one request.
- solo: existing solo agent with GPU tools, up to ten rounds.
- debate: existing describer/skeptic/experimenter/judge workflow, four rounds.

Use accounts/fireworks/models/glm-5p3, reasoning_effort=low, per-call cap 32768.
For this new pilot, solo and debate share a per-run cumulative output-token cap
of 32768; the source-only cap is also 32768. Actual provider output usage is
deducted across every role and request. Captured request limits can therefore
decrease as the budget is spent. No automatic SDK retries when this cap is used.

This matches the allocated output-token ceiling, NOT total input tokens, dollars,
GPU time, or call count. No separate final-verdict token reserve is introduced.
Budget exhaustion is a distinct non-verdict outcome, not an explicit mistake.
Any apparent advantage remains exploratory and needs a resource and trace audit.
Prompts and role behavior remain unchanged, isolating the new task mechanism
from a simultaneous debate redesign. No fourth arm in this user-requested pilot.

## Stages fixed before model outcomes

1. ea_pilot_r1: all four cases, all three arms, once each (12 runs).
2. Only if r1 contains an explicit solo error with a correct debate verdict,
   run ea_pilot_r2 for ALL four cases and ALL three arms (12 more runs).
   Abstention, token exhaustion and provider failures do not activate this gate.
3. Expand only if at least one of the same cases repeats the solo-wrong /
   debate-correct result in both designated rounds, with valid independent GPU
   evidence and complete traces, and debate has more such improvements than
   reverse errors across the pilot. This is a development signal, not a
   significance claim.
4. If that gate passes, freeze a construction rule and build six further cases
   without model feedback, balanced three compliant/three noncompliant. This
   gives about ten cases including the four pilot cases. Run all six held-out
   cases in all three arms twice under ea_expansion_r1/r2. Report pilot and
   expansion separately, including all counterexamples to the initial signal.

If the gate fails, stop paid expansion and report the failure instead of
manufacturing ten supposedly distinguishing cases. A later construction or
workflow change requires a new explicitly versioned protocol and fresh cases.
No case changes after its first model call, and no failed designated slot is
silently replaced by a retry. There is no separate high-token control this time.

## Recording and scoring

Canonical path: ../traces_glm/<case>/<arm>/rN/. Save raw requests before
submission, all returned responses/reasoning/usage, failure diagnostics, tool
events, probe code and outputs, contract/source hashes and cumulative budget.
Separate correct verdicts, explicit mistakes, abstentions, token exhaustion,
transport failures and missing results. Report API estimates separately from
GPU charges and unknown failed-call usage.

A proposed debate correction must have actual trace support: identify the
incorrect/insufficient initial test, construct a valid new measurement, and
use it to reach the correct verdict. Repeating an expected label, speculative
causal prose, or extra turns alone do not establish a successful correction.
