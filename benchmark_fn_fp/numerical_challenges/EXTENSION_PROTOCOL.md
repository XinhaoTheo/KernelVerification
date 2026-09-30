# case_44–case_49 extension protocol

Recorded 2026-09-23 before any case_44–case_49 model evaluation. This extension does not
change the original case_38–case_43 protocol or its replication gate.

Three new balanced pairs: raw-moment LayerNorm (case_44/case_45), fixed-iteration SPD solve
(case_46/case_47), and FP32 Horner polynomial evaluation (case_48/case_49). Public source and contract
specify the entire finite workload. Each pair differs only in the input seed.
CPU candidate searches use numerical error, without LLM feedback; every search
log is retained. Independent references use the actual supplied FP32 inputs.
The actual kernels must pass the existing ten-repeat T4 oracle before any paid
model evaluation. Its source/input hashes freeze each evaluated case.

All three arms use accounts/fireworks/models/glm-5p3, the existing prompts,
reasoning_effort=low, and a 32768-token limit per API call. Solo has ten rounds;
debate has four rounds. The source-only arm has one call and no tools. This
does not match total calls, tokens, or cost. The tool arm receives public cases
but not private search logs, answer keys, or numerical oracle implementations.

Predeclared trials, all covering all six cases to avoid outcome-based selection:

- extension_low32_r1: single_call, solo, debate.
- extension_low32_r2: single_call, solo, debate.
- extension_low32_r3: single_call only.
- extension_default64_r1: single_call only, reasoning_effort omitted, 65536
  tokens. This tests sensitivity to the default reasoning configuration and a
  larger allowance; it is a separate control, not a matched comparison.

The exploratory replication target is a case in at least two families with
explicitly wrong source verdicts in at least two of its three designated low
trials, and correct verdicts in both designated trials of each tool arm.
Missing verdicts, token caps, abstentions, and failures retain their slots and
are reported separately, never counted as explicit wrong judgments. They do
not qualify for the gate. Later retries cannot replace these slots. Default
reasoning control results must be disclosed even if they remove the observed
gap; one control attempt is not evidence of a stable error rate.

All attempts reserve canonical traces under ../traces_glm/<case>/<arm>/<trial>
before calls. Requests, full provider responses and reasoning, usage, errors,
finish reasons, transcripts, tool events, probe sources and results are kept.
No case is revised after model evaluation. Report all six cases and all trials,
both tool arms separately, and actual captured API cost estimates (GPU cost and
unknown charges excluded). This is exploratory synthetic case construction,
not a held-out or representative benchmark, and does not establish a benefit
of debate over solo when both tool arms perform alike.
