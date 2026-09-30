# Other-methods pilot v2

2026-09-24, recorded before the first case_66–case_71 model call. The user requested
additional small experiments after case_62–case_65 failed to separate solo and debate.
This is a new development cohort; the original protocol and negative results
remain unchanged. No case is selected using model responses.

## Three new mechanisms, six cases

- case_66/case_67: higher-order joint behavior in a finite dropout seed distribution.
  Supplied evidence can verify single-channel and pairwise behavior without
  determining the required four-channel law. Labels use exact finite-domain
  computation and an independently implemented structural reference.
- case_68/case_69: multiple latent feature views subject to a shared channel symmetry.
  A tester that permits a separate alignment per view may be insufficient for
  a contract requiring one global alignment. Both admissible transforms and
  the numerical metric must be public, explicit and independently checkable.
- case_70/case_71: mutable optimizer state and the lifetime of returned outputs.
  The contract observes retained outputs after the full update sequence.
  Copying values immediately inside a test can inadvertently change the
  observation being tested. Ground truth must preserve the actual objects.

Each pair contains one compliant and one noncompliant workload, ideally changing
only an input seed. The public contract defines all inputs and observation
semantics. Every arm gets identical public code, contract and real initial-probe
output. The initial report is data to inspect, never a privileged truth label.
Independent oracles, full candidate search records and frozen answers remain
outside model and tool images. Small selected synthetic examples do not establish
generalization. These mechanisms may also be solved by solo; that is a result.

Validate CPU references and actual T4 execution, retain failed construction or
GPU attempts, append real initial-probe outputs to the public problem, then
freeze public hashes before the first model call. Do not modify evaluated cases.
Use ten repeated decisive GPU workloads; enumerate the declared finite domain
where required. Verify permissible input mutation and output alias semantics
instead of imposing the old nonmutation assumption on every new operator.

## Exactly three arms

Existing single_call (no tools), solo (GPU tools), and four-role debate (GPU
tools). No fourth arm and no simultaneous role-prompt changes. Model:
accounts/fireworks/models/glm-5p3, Fireworks, reasoning_effort=low. Per-call
max_tokens=32768; cumulative output allowance=32768 per run across all roles.
Solo permits ten rounds, debate four. Input tokens, dollars and GPU time are
recorded but not equalized; equal output allowances are not equal total cost.
Budget exhaustion and failures are separate from explicit mistakes. SDK retries
remain disabled, and every attempt keeps raw requests/responses and probe files.

## Prospective stages

1. ea_methods_v2_r1: all six cases × all three arms (18 trials).
2. If an explicit, fully recorded solo error has a correct debate result, run
   ea_methods_v2_r2 on ALL six cases and ALL three arms (18 trials). A source-only
   error, abstention or transport failure does not activate this repeat.
3. A positive development signal requires the same solo-wrong/debate-correct
   case in both fixed rounds, more paired debate improvements than reverse
   errors, and an audit that the debate conclusion has valid runtime evidence.
4. If there is a signal, repeat that mechanism on fresh balanced cases selected
   without model feedback before calling it a reusable construction. Keep the
   original paired failures and all nonqualifying cases visible.

If all tool arms remain correct, this cohort is a negative result, not a reason
to drop trials or lower solo's budget. Other construction or workflow ideas can
be tested as a separately versioned pilot; do not relabel them as this cohort.
No unrecorded retries and no replacement of a failed designated slot.

Traces: ../traces_glm/<case>/<arm>/<trial>/. Report this cohort separately in
METHODS_V2_REPORT.md and private_data/reports/methods_v2_scoreboard.json, preserving case_62–case_65 REPORT.md
and private_data/reports/scoreboard.json. Costs are profile estimates excluding Modal GPU and any
unreported failed-call usage. Accuracy and the quality of its evidence are
audited separately.

报告默认只写 Markdown。`private_data/reports/` 中的 JSON 是可选缓存，需运行对应报告脚本并传入 `--json` 才会生成；清理时删除这些缓存不会删除实验记录。
