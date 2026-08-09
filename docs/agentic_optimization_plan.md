# Agentic Verification Optimization Plan

This document tracks the parts of the current agentic verifier that are working
but not yet good enough for a robust verification framework.

## Current Status

The system can run a full agentic verification path:

1. Load artifact context.
2. Describer explains the kernel.
3. Skeptic records claims.
4. Experimenter gathers evidence through tools.
5. Judge records a verdict.
6. Run state, claims, tool events, and verdict are persisted.

This is enough for a first working agentic verifier, but several workflow and
debugging gaps remain.

## P0: Debate Convergence Is Too Shallow (Addressed)

### Problem

The current workflow can run Judge after the first debate round if all claims
have evidence. This means the system may stop after:

```text
describer -> skeptic -> experimenter coverage -> judge
```

That is a valid single-pass verification, but it is not a strong multi-round
debate. There is no forced second round where Skeptic reviews the new evidence,
challenges weak evidence, or raises follow-up claims.

### Why This Is Bad

Real verification often needs the loop:

```text
claim -> experiment -> evidence -> critique evidence -> follow-up experiment
```

If Judge runs immediately after first coverage, weak or misleading evidence can
end the process too early.

### Optimization

Added explicit debate-round controls:

```text
--max-debate-rounds
--min-debate-rounds-before-judge
```

Workflow:

```text
for debate_round in max_debate_rounds:
    run describer/skeptic agents
    run experimenter until claim coverage is satisfied

    if debate_round < min_debate_rounds_before_judge:
        continue

    run judge
```

### Acceptance Criteria

- With `--min-debate-rounds-before-judge 2`, Judge cannot run in round 1.
- Round 2 agents see round 1 evidence in their prompt state.
- Tests verify history order.
- Current status: implemented and verified with unit tests plus a real OpenAI run.

Expected history order:

```text
orchestrator
describer
skeptic
experimenter...
describer
skeptic
experimenter...
judge
```

## P0: Experimenter Evidence Attachment Is Still Too Prompt-Dependent

### Problem

The system now allows Experimenter to run a probe in one turn and consume the
probe result in the next turn. However, attaching evidence is still dependent on
the LLM choosing the correct `append_evidence` and `update_claim_status` calls.

### Why This Is Bad

The local runtime already has structured probe output. If the probe result
contains clear claim IDs and result values, the system should help turn that
into evidence instead of relying entirely on the LLM.

### Optimization

Add an evidence drafting tool:

```text
draft_evidence_from_probe
```

Inputs:

```json
{
  "tool_event_id": "t10",
  "claim_id": "c1"
}
```

Output:

```json
{
  "claim_id": "c1",
  "kind": "runtime_probe",
  "suggested_supports": "confirmed | rebutted | inconclusive",
  "summary": "...",
  "data": {...}
}
```

Experimenter still decides whether to accept or edit the draft, but the system
provides structured assistance.

### Acceptance Criteria

- A probe result with `claim_id` can be converted into an evidence draft.
- Experimenter prompt instructs it to use the draft before writing evidence.
- Tests cover probe output -> evidence draft -> append evidence.

## P0: Full LLM Transcript Is Not Persisted

### Problem

`run.json` stores agent messages, tool calls, tool outputs, claims, and verdicts.
It does not store the full LLM call transcript:

```text
system prompt
user prompt
raw model response
provider
model
timestamp
```

### Why This Is Bad

When a real run behaves badly, we cannot fully reproduce what the model saw or
why it chose certain tool calls.

### Optimization

Add transcript persistence:

```text
llm_transcript.jsonl
```

Each record:

```json
{
  "role": "experimenter",
  "round": 4,
  "provider": "openai",
  "model": "gpt-5",
  "system": "...",
  "user": "...",
  "raw_response": "...",
  "created_at": "..."
}
```

### Acceptance Criteria

- Every real LLM call appends one transcript record.
- Transcript is not required for dry deterministic tests.
- Secrets are not stored.

## P1: CLI Naming Is Confusing (Addressed)

### Problem

`--max-rounds` used to mean max debate rounds, not total agent turns.

### Why This Is Bad

Users can reasonably expect a generic rounds flag to mean one complete discussion
cycle, but the system also has claim coverage subrounds. The old name was ambiguous.

### Optimization

Added a clearer flag:

```text
--max-debate-rounds
```

`--max-rounds` remains as a backward-compatible deprecated alias temporarily.

### Acceptance Criteria

- CLI help describes debate rounds vs claim coverage rounds.
- Tests cover both old and new flags.
- Current status: `--max-debate-rounds` exists; `--max-rounds` is a deprecated alias.

## P1: Sequential Runner Name Is Misleading (Addressed)

### Problem

`run_agents()` was no longer the main verification workflow. It was a fallback for
single-agent debugging or partial chains.

### Why This Is Bad

The old name suggested it was the primary runner.

### Optimization

Renamed:

```text
run_agents -> run_agents_sequential
```

### Acceptance Criteria

- Main CLI workflow uses `run_verification_workflow`.
- Partial debug paths use `run_agents_sequential`.
- Tests reflect the new name.
- Current status: renamed in code.

## P1: Judge Needs Better Convergence Semantics (Addressed)

### Problem

Judge currently records a final verdict once allowed to run. It cannot clearly
say:

```text
Evidence is insufficient; run another debate round.
```

except by recording `needs_more_evidence`.

### Why This Is Bad

`needs_more_evidence` can mean two different things:

1. Final result is inconclusive.
2. Workflow should continue investigating.

### Optimization

Use a two-outcome Judge protocol:

```text
record_verdict(...)      -> final result; stop workflow
request_more_debate(...) -> no final result yet; run another debate round if budget remains
```

`request_more_debate` records a reason and optional focus claim ids, without an
extra nested `decision` field. If Judge is ready to finalize, it should call
`record_verdict` directly.

### Acceptance Criteria

- Judge can request another debate round without writing a final verdict.
- Orchestrator respects `request_more_debate` when debate budget remains.
- Current status: implemented with `request_more_debate`; covered by tests.

## P1: Claim Quality Control Is Mostly Prompt-Based

### Problem

Skeptic is instructed to record at most 3 high-value claims per turn, but this is
only a soft prompt rule.

### Why This Is Bad

The model can still record too many overlapping or low-value claims.

### Optimization

Add optional claim ledger validation:

```text
max_new_claims_per_turn
deduplicate_similar_claims
require_testable_condition
```

The first version can enforce only count:

```text
max_new_claims_per_turn = 3
```

### Acceptance Criteria

- If Skeptic records more than the allowed number in one turn, the extra claims
  are rejected or marked invalid.
- Tool output reports which claims were rejected and why.

## P2: Skills Are Static Markdown Only

### Problem

Skills are loaded as markdown instructions. They cannot declare required tools,
budgets, or convergence contracts in machine-readable form.

### Why This Is Bad

Important workflow rules live in prose and are hard for the orchestrator to
enforce.

### Optimization

Add optional skill metadata:

```yaml
required_tools:
  - read_claim_ledger
  - run_python_probe
  - append_evidence
convergence:
  requires_claim_coverage: true
```

### Acceptance Criteria

- Orchestrator can read skill metadata.
- Missing required tools fails early.

## P2: Artifact and Probe Result Summaries Can Become Too Large

### Problem

Tool outputs can grow large and may crowd the prompt.

### Why This Is Bad

Long histories reduce model focus and increase cost.

### Optimization

Add structured compaction:

```text
recent full tool events
older summarized tool events
claim-centered evidence summaries
```

### Acceptance Criteria

- Prompt state remains bounded.
- Claims preserve all evidence references.

## Recommended Implementation Order

1. Add full LLM transcript persistence.
2. Add `draft_evidence_from_probe`.
3. Add hard claim-count enforcement.
4. Add machine-readable skill metadata.
5. Add prompt/history compaction.

## Definition of Done

The verifier should be considered mature when:

- It can run at least two debate rounds before final judgment.
- Every probe result is either consumed as evidence or explicitly marked
  unconsumed with a stop reason.
- Every final verdict cites decisive claims and evidence IDs.
- Full LLM transcripts are persisted for debugging.
- The workflow can stop for clear reasons:

```text
verdict_recorded
tool_budget_exhausted
claim_coverage_required
probe_output_unconsumed
claim_coverage_stalled
max_debate_rounds_exhausted
```

