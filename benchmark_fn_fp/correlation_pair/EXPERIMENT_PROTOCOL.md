# Candidate evaluation protocol (2026-09-22)

The two cases were frozen before any model call. CPU witnesses are already
validated. Do not change their contracts or sources after reading model answers;
new constructions must have a new version and preserve every trial.

1. Run both actual Triton kernels on T4 ten times, cross-check the FP64 reference
   with scalar math.fsum, and match every input hash to the CPU construction.
2. Use the same source-only prompt and JSON schema as the existing Opus baseline.
   Both source and complete input construction are exposed. Use a 32768-token
   output ceiling; token exhaustion is not a wrong answer. Save every request,
   response, stop reason and usage record, including unsuccessful attempts.
3. If the source-only baseline misclassifies a case, repeat the frozen case in
   independent requests to distinguish an isolated error from a repeatable one.
   If it abstains, label the outcome as abstention, not incorrect reasoning.
4. Evaluate solo-with-tools and debate on the same input artifacts and model.
   Inspect the probes and decisive evidence, not just verdict accuracy. Tools
   must not see answer keys, construction-side search logs, or measurements.
5. Report all trials, costs using the project's price assumptions, source-only
   answer coverage, incorrect judgments, abstentions and tool failures separately.

The target is an independently verified, reproducible computational-evidence
separation. A source-only error alone is insufficient: tool-enabled resolution
must be demonstrated. Success does not establish debate superiority to a solo
agent with tools, nor a population-level result from a selected synthetic pair.

## Execution state

The initial Modal launch could not connect from the sandbox. Its requested
network escalation was rejected by automatic approval review: exporting private
verifier and benchmark source to Modal was considered insufficiently authorized.
The user subsequently instructed the assistant to proceed with verification
after this disclosure. Automatic approval accepted the renewed request and
Modal GPU validation was started. This history is retained for audit.

Oracle validation and agent evaluation now have separate image definitions:
the oracle needs only the new answer-free cases and numeric dependencies; only
the agent evaluator needs the verifier and API credentials. Answer keys and
search logs are not mounted into either image.

Neutral meta.json files (passed=null, status=unverified) are supplied because the
existing tool-enabled dataset loader requires them. They contain no labels or
measurements. Source-only prompts still use the unchanged original prompt and
expose all kernel/input-generator code and the contract.

## Completed measurements

GPU validation passed on both cases. Two independent original-prompt calls per
case all gave incorrect verdicts. A post-observation wording ablation removed
only a nonbinding explanatory sentence: A then became correct, B remained
incorrect. Both debate runs reached correct verdicts with direct GPU evidence;
the B solo-with-tools control also reached the correct verdict. These are saved
as nine complete trial traces; no output was token-capped. See FINDINGS.md for
the interpretation, including A's wording sensitivity and its Judge's erroneous
independence explanation despite a correctly measured final error.
