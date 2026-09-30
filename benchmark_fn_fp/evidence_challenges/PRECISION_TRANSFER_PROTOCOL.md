# Conditional fresh-seed confirmation of the precision-reference mechanism

2026-09-24. Recorded before constructing or evaluating transfer cases.
This stage is conditional on the v3 predeclared repeated-case gate and an
independent evidence audit. It does not change or replace any case_72–case_75 result.

If activated, construct case_76–case_81 with an entirely new fixed 512-seed interval,
recorded before its CPU sweep. Keep the same 4×12 float32 generator, Neumaier
candidate, mathematical row-sum contract and 1e-5 tolerance. Select, in ascending
seed order, three passing workloads with at least a factor-two tolerance margin,
two failing workloads with nonzero partial-sum outputs, and one failing workload
with zero outputs. Failures must exceed twice the tolerance. This avoids using
only zero output as the distinguishing feature of failing transfer cases.
Retain every candidate and all construction failures; select without model
feedback. Use independent exact dyadic-integer summation and math.fsum, followed
by real T4 preview and ten-repeat GPU validation before source freeze.

Every arm sees identical public code, contract and actual initial-probe results.
Keep the existing model, prompts, roles, tool access, total 32768 output-token
cap and three arms. No transfer-time prompt edits or selective retries.

Run `ea_precision_transfer_r1` on all six cases and all three arms (18 trials).
Complete the entire cohort before interpreting outcomes. Report correct labels,
explicit mistakes, abstentions, execution failures, actual usage and estimated
API cost separately. This is fresh-seed validation within a mechanism chosen
adaptively on the development cohort, not proof of general debate superiority.

If debate's correct-label count exceeds solo's by at least two and at least one
valid explicit solo-wrong/debate-correct pair exists, run a second fixed full
cohort as `ea_precision_transfer_r2` (all six × all three). Otherwise stop paid
expansion and preserve the negative or inconclusive outcome. Claim a replicated
fresh-seed advantage only if debate has higher correct-label count in BOTH
rounds, with at least one same-case explicit correction repeated and supporting
evidence independently audited. An abstention counts as not correct in this
accuracy comparison. Resource/provider failures remain separately visible and
cannot substitute for a reasoning mistake.

This protocol schedules no further batch. It is a bounded confirmation step;
any later experimental design must retain these results and must not imply
that the desired result has already been obtained.
