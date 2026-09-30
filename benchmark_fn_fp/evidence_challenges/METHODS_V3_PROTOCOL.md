# Other-methods pilot v3: scope certificates and reference precision

2026-09-24. Prospective before any case_72–case_75 model evaluation. v2 completed
with both tool arms correct on all six workloads. The user asked for further
methods, so this is a new adaptive development cohort, not a held-out test of
the preceding constructions. All earlier negative results remain unchanged.

## Mechanisms

case_72/case_73: a pruned residual ReLU block must meet an absolute error bound over
an explicitly declared continuous input box, with fixed public coefficients.
Initial measurements cover only a small sample. Ground truth uses a certificate
for the whole domain: exhaustive vertices of a convex residual, cross-checked
by an independent active-subset/support-function maximization. Candidate base
rounding must be exactly absent or rigorously bounded, so the certificate is
about the actual candidate, not merely an ideal arithmetic surrogate. A passing
case must not be rejected solely because a loose sum of per-neuron maxima
exceeds the tolerance; a failure requires an admitted counterexample.

case_74/case_75: a compensated FP32 reduction is compared with a naive FP64 reduction
over a cancellation-heavy workload. A larger dtype is not by itself a valid
reference certificate. Exact summation of stored dyadic inputs and an independent
high-accuracy summation reference establish the label. The FP32 compensation can
itself lose precision, so the pair includes both a compliant and a noncompliant
workload. The initial comparison may falsely reject a compliant candidate.

Selection uses numerical/structural CPU calculations only, retaining complete
candidate and calibration records. No seed selection from model responses.
Actual T4 preview, ten repeated decisive measurements and independent references
precede the source freeze. Real initial-probe results are appended to the public
contract before final freeze. Agent images receive only public artifacts. Every
arm receives the same contract, input generator, candidate and initial evidence.
New public source and problem files remain below the existing prompt limits.

## Arms, budget, stages

Exactly single_call/no tools, solo/tools and existing four-role debate/tools.
GLM accounts/fireworks/models/glm-5p3 through Fireworks, low reasoning, 32768
per-call ceiling and 32768 cumulative output tokens per run across all roles.
Ten solo rounds and four debate rounds. No prompt or role-policy change. Input
tokens and GPU usage are not equalized; report actual API estimate separately.
No automatic retries, no substitution of failed trials and no case edits after
the first model call. Keep complete raw calls and tool artifacts.

ea_methods_v3_r1: all four cases, all three arms (12 trials).
If there is an explicit valid solo error and a correct debate verdict, run
ea_methods_v3_r2 on ALL four cases and ALL three arms (12 further trials).
The same-case correction must repeat and outweigh reverse paired errors, with
independently audited evidence, before the mechanism qualifies for expansion.
Abstention, resource exhaustion and provider failure are not explicit mistakes.
The fixed slots cannot be replaced by successful retries. All outcomes remain
in the report even when the desired difference is absent.

Save METHODS_V3_REPORT.md / private_data/reports/methods_v3_scoreboard.json separately. A positive
signal in this development search still needs fresh cases to establish transfer.
A negative signal does not justify weakened solo permissions, withheld facts,
different scoring rules or an increased debate-only budget.

报告默认只写 Markdown。`private_data/reports/` 中的 JSON 是可选缓存，需运行对应报告脚本并传入 `--json` 才会生成；清理时删除这些缓存不会删除实验记录。
