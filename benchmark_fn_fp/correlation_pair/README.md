# Two-path quantization: correlation witness

[case_36](../triton_eval_cases/case_36/problem.txt) and
[case_37](../triton_eval_cases/case_37/problem.txt) test `y = A @ x + B @ x`
with per-row low-bit quantization. Their public code differs only in B's row
permutation: individual branch errors retain the same norms but reinforce or
cancel when added. Every arm sees the complete generator, seed and permutation.
The reference evaluates original FP32 inputs in FP64, with a 10% relative L2
budget for these fixed inputs.

Source-only judgments failed in several recorded attempts; both solo and debate
verified the relevant inputs with execution tools. The experiments establish no
additional accuracy benefit from four-role debate. The distinct Opus, GLM 32K
and later GLM 64K cohorts remain separate below.

<a id="results"></a>

<!-- BEGIN GENERATED RESULTS -->
## 实验结果（自动生成）

Generated: 2026-10-01T03:20:21.111276+00:00

All **17 attempts** from `traces_glm` and `traces_opus5` are included once. Local rN names identify stored attempts; original experiment batches remain separate.

See [protocol](#protocol), [evidence audit](#evidence-audit) and [64K follow-up](#glm-64k-follow-up) for the manually maintained experimental context.

### Frozen GPU labels

| Case | Relative L2 error | Budget | Truth |
|---|---:|---:|---|
| case_36 | 0.2241158064 | 0.1 | reject |
| case_37 | 0.0336471468 | 0.1 | trust |

Frozen environment: {"gpu": "Tesla T4", "torch": "2.8.0+cu128", "triton": "3.4.0", "numpy": "1.26.4"}

### Experiment groups

Token caps are per request unless a shared output budget is recorded. Missing historical settings remain unknown rather than inferred from current defaults.

| Model / provider | Original batch | Arm | Token cap / shared cap / variant | Attempts | Correct | Wrong | Abstain | Token limit | No verdict | Pending | API estimate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-opus-5 / anthropic | neutral1 | single_call | 32768 / unknown / neutral_contract | 2 | 1 | 1 | 0 | 0 | 0 | 0 | $0.152685 |
| claude-opus-5 / anthropic | r1 | debate | unknown / unknown / original | 2 | 2 | 0 | 0 | 0 | 0 | 0 | $2.658329 |
| claude-opus-5 / anthropic | r1 | single_call | 32768 / unknown / original | 2 | 0 | 2 | 0 | 0 | 0 | 0 | $0.3027 |
| claude-opus-5 / anthropic | r1 | solo | unknown / unknown / original | 1 | 1 | 0 | 0 | 0 | 0 | 0 | $0.500216 |
| claude-opus-5 / anthropic | r2 | single_call | 32768 / unknown / original | 2 | 0 | 2 | 0 | 0 | 0 | 0 | $0.289825 |
| accounts/fireworks/models/glm-5p3 / fireworks | r1 | debate | 32768 / unknown / original | 2 | 2 | 0 | 0 | 0 | 0 | 0 | $0.322377 |
| accounts/fireworks/models/glm-5p3 / fireworks | r1 | single_call | 32768 / unknown / original | 2 | 1 | 0 | 0 | 1 | 0 | 0 | $0.0504042 |
| accounts/fireworks/models/glm-5p3 / fireworks | r1 | solo | 32768 / unknown / original | 2 | 2 | 0 | 0 | 0 | 0 | 0 | $0.09537 |
| accounts/fireworks/models/glm-5p3 / fireworks | r2_64k | single_call | 65536 / unknown / original | 2 | 1 | 1 | 0 | 0 | 0 | 0 | $0.053053 |

### Every attempt

| Case | Model / provider | Arm | Trial | Original batch | Cap | Variant | Status | Verdict | Outcome | In / out tokens | API estimate / basis | Capture | Trace |
|---|---|---|---|---|---:|---|---|---|---|---:|---|---|---|
| case_36 | accounts/fireworks/models/glm-5p3 / fireworks | single_call | r1 | r1 | 32768 | original | historical | reject | correct | 1210 / 12438 | $0.0140206 / saved usage.estimated_usd | complete_api | [open](../traces_glm/case_36/single_call/r1/transcript.md) |
| case_36 | accounts/fireworks/models/glm-5p3 / fireworks | single_call | r2 | r2_64k | 65536 | original | historical | reject | correct | 1210 / 25492 | $0.02838 / saved usage.estimated_usd | complete_api | [open](../traces_glm/case_36/single_call/r2/transcript.md) |
| case_36 | accounts/fireworks/models/glm-5p3 / fireworks | solo | r1 | r1 | 32768 | original | historical | reject | correct | 55989 / 21631 | $0.039471 / historical project model profile | history_only | [open](../traces_glm/case_36/solo/r1/transcript.md) |
| case_36 | accounts/fireworks/models/glm-5p3 / fireworks | debate | r1 | r1 | 32768 | original | historical | reject | correct | 193910 / 66670 | $0.127632 / historical project model profile | history_only | [open](../traces_glm/case_36/debate/r1/transcript.md) |
| case_37 | accounts/fireworks/models/glm-5p3 / fireworks | single_call | r1 | r1 | 32768 | original | historical | unknown | token_limit | 1210 / 32768 | $0.0363836 / saved usage.estimated_usd | complete_api | [open](../traces_glm/case_37/single_call/r1/transcript.md) |
| case_37 | accounts/fireworks/models/glm-5p3 / fireworks | single_call | r2 | r2_64k | 65536 | original | historical | reject | wrong_verdict | 1210 / 22122 | $0.024673 / saved usage.estimated_usd | complete_api | [open](../traces_glm/case_37/single_call/r2/transcript.md) |
| case_37 | accounts/fireworks/models/glm-5p3 / fireworks | solo | r1 | r1 | 32768 | original | historical | trust | correct | 60838 / 35331 | $0.055899 / historical project model profile | history_only | [open](../traces_glm/case_37/solo/r1/transcript.md) |
| case_37 | accounts/fireworks/models/glm-5p3 / fireworks | debate | r1 | r1 | 32768 | original | historical | trust | correct | 271768 / 107864 | $0.194745 / historical project model profile | history_only | [open](../traces_glm/case_37/debate/r1/transcript.md) |
| case_36 | claude-opus-5 / anthropic | single_call | r1 | r1 | 32768 | original | historical | trust | wrong_verdict | 1920 / 7695 | $0.201975 / saved usage.estimated_usd | text_and_usage_only | [open](../traces_opus5/case_36/single_call/r1/user_prompt.txt) |
| case_36 | claude-opus-5 / anthropic | single_call | r2 | r2 | 32768 | original | historical | trust | wrong_verdict | 1920 / 8103 | $0.212175 / saved usage.estimated_usd | text_and_usage_only | [open](../traces_opus5/case_36/single_call/r2/user_prompt.txt) |
| case_36 | claude-opus-5 / anthropic | single_call | r3 | neutral1 | 32768 | neutral_contract | historical | reject | correct | 1896 / 2482 | $0.07153 / saved usage.estimated_usd | text_and_usage_only | [open](../traces_opus5/case_36/single_call/r3/user_prompt.txt) |
| case_36 | claude-opus-5 / anthropic | debate | r1 | r1 | unknown | original | historical | reject | correct | 195101 / 13547 | $1.534235 / historical project model profile | history_only | [open](../traces_opus5/case_36/debate/r1/transcript.md) |
| case_37 | claude-opus-5 / anthropic | single_call | r1 | r1 | 32768 | original | historical | reject | wrong_verdict | 1920 / 3645 | $0.100725 / saved usage.estimated_usd | text_and_usage_only | [open](../traces_opus5/case_37/single_call/r1/user_prompt.txt) |
| case_37 | claude-opus-5 / anthropic | single_call | r2 | r2 | 32768 | original | historical | reject | wrong_verdict | 1920 / 2722 | $0.07765 / saved usage.estimated_usd | text_and_usage_only | [open](../traces_opus5/case_37/single_call/r2/user_prompt.txt) |
| case_37 | claude-opus-5 / anthropic | single_call | r3 | neutral1 | 32768 | neutral_contract | historical | reject | wrong_verdict | 1896 / 2867 | $0.081155 / saved usage.estimated_usd | text_and_usage_only | [open](../traces_opus5/case_37/single_call/r3/user_prompt.txt) |
| case_37 | claude-opus-5 / anthropic | solo | r1 | r1 | unknown | original | historical | trust | correct | 49286 / 6781 | $0.500216 / historical project model profile | history_only | [open](../traces_opus5/case_37/solo/r1/transcript.md) |
| case_37 | claude-opus-5 / anthropic | debate | r1 | r1 | unknown | original | historical | trust | correct | 143550 / 15579 | $1.124094 / historical project model profile | history_only | [open](../traces_opus5/case_37/debate/r1/transcript.md) |

Total recorded API estimate: **$4.4249592**; unknown-cost attempts: 0; partial-cost attempts: 0.
Outcomes: {"correct": 10, "wrong_verdict": 6, "abstention": 0, "token_limit": 1, "no_verdict": 0, "pending": 0}. Input / output tokens: 986754 / 387737.

Single-call costs prefer the original saved estimate. Other costs use saved pricing snapshots or historical project profiles, not today's list prices. These estimates exclude GPU charges and are not invoices.

Capture: {"complete_api": 4, "history_only": 7, "text_and_usage_only": 6}. `complete_api` requires original request/response payloads; `text_and_usage_only` has prompt/final text and usage without the API envelope; `history_only` has agent/tool transcripts and recorded usage without raw API calls.

Public hashes are checked against frozen labels. The neutral-contract ablation permits only removal of the original explanatory branch-error sentence, with source and modified problem hashes checked separately.

### Automatic trace-audit findings

- [accounts/fireworks/models/glm-5p3 / case_37 / debate / r1](../traces_glm/case_37/debate/r1/transcript.md): tool error: finalize_probe_evidence -- tool finalize_probe_evidence missing required arg: supports
<!-- END GENERATED RESULTS -->

## Files and commands

This directory contains this README and `private_data/` (CPU answer key,
candidate search, exact NPZ inputs, CPU/GPU validation and code provenance).
Public cases are stored once in `../triton_eval_cases/case_36/` and `case_37/`.
The logical dataset remains `correlation_pair`; `--dataset correlation_pair --all`
selects these two registered cases, not every folder in that shared tree.

Programs live in `../eval_scripts/correlation_pair/`. From the repository root:

```sh
python benchmark_fn_fp/eval_scripts/correlation_pair/validate_cpu.py
modal run benchmark_fn_fp/eval_scripts/correlation_pair/validate_gpu.py
python benchmark_fn_fp/eval_scripts/correlation_pair/report.py
```

CPU validation and reporting make no model calls; GPU validation runs the T4
oracle and incurs Modal charges. The report command replaces only the generated
results block above. `--json` optionally exports structured results under
`private_data/reports/`. `build.py` retains the CPU construction recipe and
refuses to overwrite evaluated artifacts; use a fresh copy for reconstruction.

New paid GLM evaluations use the shared runners:

```sh
python benchmark_fn_fp/eval_scripts/run_single_fireworks.py --dataset correlation_pair --cases case_36,case_37 --max-tokens 32768
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --dataset correlation_pair --arm both --cases case_36,case_37 --provider fireworks --max-tokens 32768
```

Omitting `--trial` reserves an unused `rN`. These commands use current defaults;
they do not automatically reproduce historical prompts, SDK retry policy or
total budgets. Traces reside in `../traces_opus5/` and `../traces_glm/`, under
case/arm/rN. Original batches and prompt variants remain in metadata. A matching
number alone does not establish equal settings. See [the trace guide](../TRACES.md)
and [migration record](../traces_opus5/migration_20260930_correlation_cleanup.json).

## Protocol

The original Opus and Fireworks protocols were recorded before their
2026-09-22 batches. Both cases were frozen before model evaluation. Inputs,
source, contracts, thresholds and labels were never adjusted to model answers.
New constructions must use new versions and preserve prior attempts.

The common plan was to validate each actual Triton kernel on T4 ten times,
cross-check FP64 references with scalar `math.fsum`, and match public-generator
input hashes to CPU construction. Every evaluation arm received the complete
public source and contract; neither oracle nor agent images mounted answer
keys or search logs. Tool loaders received only neutral `meta.json` files
(`passed=null`, `status=unverified`). Probes, usage, failures and all verdicts
were retained. Abstention, malformed answers, token exhaustion, infrastructure
failure and submitted wrong verdicts were kept separate. Historical capture
gaps remain unknown rather than reconstructed.

### Opus settings and follow-ups

Model: `claude-opus-5`. The protocol declared 32768 output tokens per API call;
some historical tool metadata did not save the cap, so the generated table
correctly shows it as unknown. Source-only calls used the original baseline
system/user prompt and JSON schema, with no tools. The conditional plan was to
repeat a misclassified frozen case independently, then test tool-enabled
resolution with the same model.

Each case received two original-prompt calls, followed by one post-observation
wording ablation removing only the nonbinding sentence that a single branch's
error may exceed 0.1. The numerical contract, source, inputs and threshold
remained unchanged. This was not a prespecified replication of the original
condition. Its original label `neutral1` is preserved; its current path is
`single_call/r3`. Two debate runs and the case_37 solo control complete the nine
Opus attempts. All six source-only calls ended with `end_turn`, without token
exhaustion or `needs_more_evidence`.

### GLM settings and follow-ups

Fireworks model `accounts/fireworks/models/glm-5p3` used `FIREWORKS_API_KEY`.
The original 2026-09-22–23 comparison ran exactly six cells: both cases, each
with source-only, solo and debate, once each. All used the original contract
wording, with no semantic retry or selection of successful answers.

Source-only used the baseline prompts, an explicitly appended verdict schema
and JSON-object format, saving the full provider response and supplied
reasoning. It had no SDK retries. Tool arms used the existing verifier prompts,
T4 tools and then-existing SDK retry behavior. Debate roles were describer,
skeptic, experimenter and judge. All arms had a 32768-token cap per API call
and a 1800-second timeout; solo allowed ten rounds, debate four. Total calls,
tokens and costs were not matched. Original six-cell records use `r1`.

After observing that comparison, the user requested one further independent
source-only call per case. Only `max_tokens` changed to 65536; requests were
checked field by field and source/contract hashes matched the frozen oracle.
No tools, answer feedback or semantic retries were used. The calls ran
concurrently. Original label `r2_64k` is preserved; current paths are
`single_call/r2`. This budget follow-up also changes the independent model
sample, so it is a separate cohort. The pre-run estimate if both calls filled
their caps was about $0.145; observed usage cost $0.053053.

Historical GLM pricing was the project's $0.28/$1.10 per million input/output
tokens, not a verified invoice or a current rate. Estimates exclude Modal and
unreported usage from failed/retried HTTP requests. Summed SDK wall time includes
provider/network waiting and retries, not pure generation or parallel duration.

## Construction and oracle

Both inputs share x and A; permuting B preserves its elements, row multiset,
norm, extrema and singular values. Each branch's scalar error multiset and
norm remain unchanged, while the final reference norms are nearly equal.
Construction paired measured branch errors in the same or opposite order,
using a predeclared 10% threshold and 25% clearance. The first candidate seed
qualified, before any model response was observed.

Branch error L2 norms are approximately 0.849963 and 0.961688. Their cosine is
+0.95282 in case_36 and -0.96345 in case_37. Independent-noise combination would
predict `sqrt(0.849963**2 + 0.961688**2) / 7.988 = 0.161` for both cases, wrongly
rejecting case_37. Independent random generation does not establish independence
after the specified error-dependent permutation.

CPU construction used NumPy 1.26.4. FP32 emulation was cross-checked with
sequential FP32 reduction; FP64 sums with scalar `math.fsum`. Public-generator
arrays matched NPZ bytes and input hashes; checks confirmed B's exact row
multiset and branch error norms. The reduction-order sensitivity bound assumed
identical products and quantization codes and did not validate GPU lowering,
division or compilation. Subsequent actual T4 validation gave ten identical
outputs per case and matching input hashes. Frozen evidence is in
[GPU validation](private_data/validation_gpu.json) and
[CPU validation](private_data/validation_cpu.json).

## Evidence audit

### Opus: wording sensitivity and measured cancellation

case_37's three source-only answers estimated independent errors of about 17%
and rejected with confidence 0.84, 0.83 and 0.86. Its
[debate trace](../traces_opus5/case_37/debate/r1/transcript.md) measured
E=0.03364714675 in t8 using the actual kernel and FP64 reference from original
FP32 inputs. t9 found actual combined error norm about 0.269 versus 1.283 under
independence, with Pearson correlation about -0.964. t12 repeated execution five
times with bitwise-identical outputs; the threshold is about 2.97 times the
observed error. The [solo control](../traces_opus5/case_37/solo/r1/transcript.md)
also independently measured the same error, returned trust and repeated five
times. It did not require another agent to identify the cancellation.

case_36's original-prompt answers guessed cancellation from the explanatory
sentence, then correctly rejected after its removal. This is wording
sensitivity, not a stable hard case independent of prompts. Its
[debate trace](../traces_opus5/case_36/debate/r1/transcript.md) directly measured
E=0.22411580645 and reproduced it with quantization emulation, yet the Judge
incorrectly said errors were independent and combined in quadrature: that would
give norm 1.283, not the observed 1.790. All three tool workflows passed the
automatic audit; this explanation error required manual review. The nine Opus
attempts cost about $3.9037 under historical project estimates, excluding Modal.

### GLM 32K: direct evidence and recovered errors

The [case_36 source-only response](../traces_glm/case_36/single_call/r1/raw_response.json)
correctly rejects but estimates independent noise at about 17%; it does not
compute 22.4116%. The
[case_37 response](../traces_glm/case_37/single_call/r1/raw_response.json) has
`finish_reason=length`, exactly 32768 output tokens and empty `message.content`.
Its unfinished reasoning includes noise estimates and guesses about the intended
answer. None is a submitted verdict: this is a token limit, not an abstention
or a wrong answer.

All four tool runs matched frozen source/contract hashes and actual T4 truth:

- [case_36 solo](../traces_glm/case_36/solo/r1/transcript.md): t6 uses the actual
  kernel, `make_inputs()` and FP64 reference, measures E=0.22411580644590948,
  checks finite outputs, independent ideal-arithmetic emulation and repeat
  equality. Four API calls, one probe.
- [case_37 solo](../traces_glm/case_37/solo/r1/transcript.md): t6 measures
  E=0.033647146752712576. t8 computes references before invoking the kernel,
  compares Torch and NumPy FP64 (difference 4.44e-16), checks input nonmutation
  and determinism, and finds cosine -0.9634516751. Four calls, two probes.
- [case_36 debate](../traces_glm/case_36/debate/r1/transcript.md): t8 measures the
  correct metric; t9 checks ideal emulation, round-half-up versus half-even and
  zero FP32/FP64 code flips. Claim c1 mistypes the denominator floor as 0.08
  instead of 0.008; probe and verdict use 0.008, and reference norm 7.988
  dominates both. Six calls, two probes. Measurement supports rejection;
  the earlier independence assumption does not.
- [case_37 debate](../traces_glm/case_37/debate/r1/transcript.md): t8 measures
  E=0.033647146752712576; t9 finds Pearson -0.96402917 and cosine -0.96345166.
  Final trust has confidence 0.95. Seven calls, three probes, 107864 output tokens.

case_37 debate's [t13](../traces_glm/case_37/debate/r1/probes/t13_probe.py)
independently reconstructs bitwise-matching inputs and an anti-sort permutation
matching 64/64 rows. Identity pairing gives E≈0.16717; 200 random pairings
average E≈0.16095, with none passing. These outside-contract controls explain
cancellation without invalidating the actual compliant workload. They add
explanation beyond solo, not an additional correct verdict or repair of a solo
error. Different total calls, tokens and role prompts prevent attributing that
explanatory depth to role structure alone.

That run recovered from t11 omitting `finalize_probe_evidence.supports`; t12
supplied it, marked c2 rebutted and retained t9's data. The preceding Experimenter
call used exactly 32768 tokens, but its raw finish reason was not saved;
truncation is suggestive, not confirmed. The run continued to a valid verdict.
The auditor's one recovered-tool-error flag remains; the other three tool runs
had no workflow flags. Neither outcome certifies all natural-language reasoning.

t13 remains in tool events and the verdict but is not a separate ledger evidence
item. `decisive_claims` contains descriptions starting c1/c2 rather than bare
IDs; manual review resolves them, and automated readers must not assume bare-ID
formatting. A backup request raced with normal container completion; the full
returned archive contains all three probes and outputs, with no lost evidence
or model retry.

The original six cells cost about $0.4682. Debate cost 3.38× solo while both were
2/2 correct. Summed SDK wall time was about 10.26 minutes source-only, 10.57 solo
and 33.82 debate, including service/network waiting and retries.

### GLM 64K follow-up

Both [case_36/r2](../traces_glm/case_36/single_call/r2/) and
[case_37/r2](../traces_glm/case_37/single_call/r2/) ended with `finish_reason=stop`.
case_36 correctly rejected (confidence 0.90, 25,492 output tokens, 305 seconds,
$0.028380). case_37 wrongly rejected (0.85, 22,122 tokens, 314 seconds, $0.024673).
Concurrent waiting was about 5 minutes 14 seconds; there were no GPU tasks.

case_37 again estimated independent combined errors of 16%–17%, missing the
permutation's cancellation and actual 3.3647% error. This is a submitted wrong
verdict; the earlier 32K attempt remains a separate token-limit outcome.
case_36's correct label still relies on the independence estimate and calls its
64 output rows 128 rows; it does not derive 22.4116%.

Both actual outputs were below 32768 tokens. The higher cap alone cannot be
credited with completion; independent sampling can also explain the change.
One post-observation follow-up is not a stable error-rate estimate. The original
six-cell comparison and these two later calls remain separate and reproducible
from raw traces.

### Infrastructure history

The initial Modal launch could not connect from the sandbox. Automatic approval
rejected network escalation because exporting private verifier/benchmark source
was insufficiently authorized. After disclosure, the user instructed verification
to proceed and a renewed request was accepted. Oracle image build order and
remote path resolution were fixed before model evaluation. Oracle and agent
execution used separate images; neither mounted answer keys or search logs.

The first local Fireworks command failed to import `openai` before any API call;
it was rerun in an existing environment with the dependency, preserving the
startup error. Debate case_37 later launched separately while case_36 was
running. The old sequential driver refused case_37's existing directory before
another remote call; that `Trace already exists` exit was a duplicate-run guard,
not an additional paid attempt or model failure.

## Interpretation limits

These are two selected, fully public synthetic inputs, not production defects,
an all-input-domain correctness proof or an independent large-sample benchmark.
They do not show that a strong model cannot solve the workload statically, that
other prompts/models must fail, or that debate is more accurate than solo with
tools. Given the reference and metric, a fixed numerical script also succeeds;
this does not defeat every `allclose` test. Confidence was recorded, not
calibrated or used to relabel outcomes.

All nine Opus and eight GLM attempts remain, including errors, token exhaustion,
recovered tool failures and post-observation variants. They were not pooled into
the original 32-case scoreboard at experiment time (that suite now has 34 active
cases). Current reports keep this dataset and its cohorts separate. Historical
tool traces lack raw per-call provider payloads; Opus source-only records retain
prompt/final text and usage without the API envelope. Those gaps cannot be
reconstructed from a score table and remain disclosed above.
