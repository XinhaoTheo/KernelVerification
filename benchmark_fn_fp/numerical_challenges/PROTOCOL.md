# GLM numerical challenge experiments

Started 2026-09-23 at the user's request. This is an exploratory synthetic
dataset, separate from the original FN/FP benchmark and the earlier case_36/case_37 pair.
Its purpose is to test additional mechanisms whose concrete numerical outcome
is difficult to infer from source without executing the public workload.

## Initial batch and frozen truth

Six cases form three balanced pairs: attention sensitivity to logit quantization
(case_38/case_39), rounding propagation in a recurrence (case_40/case_41), and summation-order loss
(case_42/case_43). Each pair has one numerically compliant and one noncompliant workload.
The agent sees the complete input generator, seed, constants, kernel and final
error contract. Private construction/oracle files are never mounted in its GPU
container. This is finite-workload compliance, not arbitrary-input correctness.

Before any model call, independent CPU references are compared, then the actual
Triton kernel is run ten times on T4. CPU/GPU labels and input/source hashes must
agree. `private_data/validation_gpu.json` freezes the hashes. Do not revise an evaluated case;
new adaptive attempts receive new case IDs and are retained, including failures.
Candidate search logs stay alongside the private answer keys.

## Three arms

All arms use `accounts/fireworks/models/glm-5p3` through the existing Fireworks
credential. They share the same frozen case and 65536 maximum output tokens per
API call, including provider reasoning. Existing baseline prompts are unchanged.

1. `single_call`: one source-only call, no tools, no earlier case results.
2. `solo`: one agent with local GPU tools, at most ten rounds.
3. `debate`: describer/skeptic/experimenter/judge with GPU tools, at most four
   debate rounds under the existing orchestrator.

There is no equal-total-token or equal-cost constraint between the agentic arms;
report API usage, probes and estimated cost alongside accuracy. A difference
between these arms would not by itself isolate the effect of debate from compute.
No application-level retry is made merely because a verdict is wrong.

First run all six cases once in each arm. Run one tool-enabled smoke trial and
inspect its complete trace before the remaining tool batch. Up to two source
calls and four GPU tool runs may be active concurrently. Repeats get fresh trial
IDs and independent conversations. Credential material must never be logged.

## Success and follow-up

The practical target is two additional mechanisms with a case that gives an
explicitly wrong source-only verdict in at least two of three source-only
attempts, while each tool arm gives the correct verdict in both initial and
confirmation trials. This is a small exploratory replication gate, not a
statistical claim of generalization. Report all cases and all attempts; never
report only the selected cases as aggregate benchmark accuracy.

Separately inspect whether debate corrects an error made by solo. If both tool
arms agree and succeed, report tool benefit only. An abstention, missing final
answer, timeout, transport failure or token-budget exhaustion is not an explicit
wrong verdict. Extra explanation alone is not a correct-label improvement.

## Records

Every paid run reserves `../traces_glm/<case>/<arm>/rN/` before calling the
model. Preserve full API requests/responses, provider reasoning, finish reasons,
reported usage, readable transcript and final answer. Tool runs also preserve
state, claims, tool events and every probe's source/stdout/stderr/result. Failed
calls remain recorded. Current remote archives are returned when a task ends;
container hard termination can still prevent recovery of unreturned files.

`eval_scripts/scoreboard.json` and `traces_glm/INDEX.md` are derived from these records.
The dataset report must separate exploratory selection from repeat evidence,
include cumulative model cost estimates and exclude unknown charges from any
claim of a complete bill. Modal GPU cost is separate from token estimates.
