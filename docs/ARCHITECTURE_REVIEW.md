# Agentic Verification Architecture Review

This document records the known gaps in the current agentic verifier prototype.
It is not a replacement for `README.md` or `agentic_roadmap.md`. The README
describes how the system works, while this file tracks where the implementation
still differs from the intended architecture.

## Current Assessment

The prototype has the main end-to-end structure:

```text
artifact
  -> Describer builds an implementation model
  -> Skeptic records testable claims
  -> Experimenter runs local probes and attaches evidence
  -> Skeptic reviews the latest evidence
  -> Judge records a verdict or requests more debate
```

The runtime also persists claims, tool events, probe artifacts, transcripts, and
the final verdict. The core direction is sound, but several boundaries still
depend too heavily on prompts or are described incorrectly in the diagrams.

## Independent Re-Review (2026-08-05)

The sections below this point were written as a forward-looking plan. This section is a
fresh pass against the code as it actually stands, not against what the plan assumed. Two
findings here were not in the original plan at all and are judged more urgent than anything
below; three of the original P0 items turned out to already be implemented in the same commit
that introduced this document, which the plan text below does not reflect.

### P(-1): No End-to-End Run Has Ever Produced a Verdict — First Real Run Done, Partial

**Update (2026-08-06):** `kv-agentic-run --all --agents describer,skeptic,experimenter,judge`
was run for the first time against a real provider (Anthropic). Result: 5 of 24 entries
completed with a recorded verdict; 19 failed with `ProtocolError: agent response is not valid
JSON`. This is the first real signal this document has ever had, and it cuts both ways:

- **The 5 that completed all reached the correct verdict.** The four `_correct` negative
  controls (`_advprec_matmul_correct`, `_advprec_matmul_fp8`, `_advprec_softmax_correct`,
  `_advprec_softmax_fp8_correct`) all got `trust` with confidence 0.9-0.97, and the Judge's own
  reasoning correctly treated FP8 lossiness as shared between kernel and reference rather than
  a kernel-specific defect — the exact class-aware judgment the old rule-based precision axis
  tried to hand-code, produced here by free agent reasoning instead. `_advprec_topk_subsample`
  (a real bug: `STRIDE=2` excludes odd-indexed keys from top-k) got `reject` at confidence 0.95,
  backed by an actual runtime probe (16/16 rows missing a true top-8 index, mean value-gap
  0.66-2.17), not assertion without evidence. This is meaningful positive signal for the core
  thesis, on the (small, unfortunately) sample that got far enough to produce one.
- **`_advprec_topk_boundary`** — the specific tie-break-bug entry this document named above as
  the target acceptance case — was one of the 19 that failed to parse. It is still unverified.
- **Root cause of the 19 failures, diagnosed and fixed the same day:** `AnthropicLLMClient.call`
  had no structured-output enforcement — it asked for JSON via system-prompt text only, unlike
  `OpenAILLMClient.call` which already used `response_format={"type": "json_object"}`. 15/19
  failures were prose wrapped around otherwise-valid JSON (`Expecting value` / `Expecting ','
  delimiter` / `Expecting property name`); 4/19 were genuine `max_tokens` truncation
  (`Unterminated string starting at`) on verbose Describer turns. A contributing gap: the raw
  malformed text was never persisted anywhere, so today's specific failures can't be replayed
  for a byte-for-byte post-mortem — confirmed only by inspecting the last-successful turn's
  content in each failed run's partial transcript.

  Fixed: both `AnthropicLLMClient` and `OpenAILLMClient` now use the provider's native
  tool-calling (`tools=`/`tool_choice`) when tools are available, instead of hand-parsing JSON
  out of free text — this is exactly the migration `agentic_roadmap.md` §9 anticipated
  ("swapped to native Anthropic/OpenAI tool calling without changing tool implementations").
  The result is re-serialized into the same `{"message", "tool_calls"}` text shape inside
  `llm.py` (`_serialize_tool_response`), so `protocol.py`, `LLMAgent`, the tool registry, and
  every existing test are unchanged. `protocol.py::_extract_json_text` was also hardened
  (fenced-block-anywhere, then balanced-brace scan respecting string escaping) as a second line
  of defense for any future non-tool-calling text path. `LLMAgent.act` now attaches a
  head+tail snippet of the raw response to any `ProtocolError` it re-raises, so a future failure
  is diagnosable without re-running. All fixes verified against the real Anthropic and OpenAI
  APIs (not just the mocked test suite): a live tool-calling round-trip through the unchanged
  `parse_agent_response` succeeded for both providers.

Acceptance criteria, revised:

- Re-run the 19 failed entries with the tool-calling fix; confirm `_advprec_topk_boundary`
  specifically converges to `reject`.
- A full `--all` run with zero `ProtocolError` failures, whatever the verdicts turn out to be.

### P(-1): No Sandboxing on Agent-Executed Probe Code — Fixed

`verifier/agentic/tools/execution.py::_execute_python_probe` used to run LLM-generated code via
`subprocess.run([sys.executable, probe_path], env=env, timeout=timeout_s, ...)` with no
containment beyond the wall-clock timeout, sharing the orchestrator's OS user, filesystem, and
network. The threat model this project exists for is "the kernel under review may be
adversarial," and the probe code that inspects that kernel is itself LLM-generated, reacting to
that kernel's source, unattended, on every Experimenter turn — treating it as trusted was
inconsistent with the project's own premise.

Implemented and empirically verified on this host (not assumed):

- **CPU**: an unconditional `ulimit -t <timeout_s + 30>` wrapper (`/bin/sh -c 'ulimit -t N;
  exec "$@"'`). Verified a tight busy loop is SIGKILLed at the limit; does not interact with
  CUDA.
- **Memory**: `systemd-run --user --scope -p MemoryMax=8192M` when a systemd user session is
  available, gated behind a cached one-time availability probe. **`ulimit -v`/`RLIMIT_AS` was
  tried first and rejected** — verified empirically that even a 16 GiB `RLIMIT_AS` makes CUDA
  context creation fail (`cudaGetDeviceCount` "out of memory"), because the NVIDIA driver's
  virtual-address bookkeeping for a device context is far larger than actual usage. cgroup
  `MemoryMax` tracks resident memory instead, so a 4-6 GiB cap runs real CUDA matmuls fine while
  still SIGKILLing a genuine 20 GiB runaway allocation — verified both ways on this host. When
  no systemd user session exists, memory is honestly left uncapped (recorded as `null` in
  `sandbox.memory_limit_mb`) rather than applying a `ulimit -v` value that is either broken
  (probe's own `import torch` on a CUDA-enabled build fails below ~8 GiB just to mmap the
  shared libraries) or too loose to mean anything.
- **Network**: `unshare --net --map-root-user` when unprivileged netns unshare works. Verified
  it blocks real outbound connections (`OSError: Network is unreachable`) while leaving
  loopback usable.
- **Env hygiene**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, cloud credentials, `SSH_AUTH_SOCK`,
  and similar are stripped from every probe's environment unconditionally, independent of
  whether network isolation is available.
- **Process-group kill on timeout**: the probe now launches via `Popen(..., start_new_session=True)`
  instead of `subprocess.run(timeout=...)`, so a timeout kills the probe's entire process group
  (`os.killpg`) rather than only the immediate child — closing the gap where a probe-spawned
  grandchild could survive as an orphan after the parent was killed. Verified no orphan
  survives after a timeout.
- Every probe result now carries a `sandbox` field (`cpu_limit_s`, `memory_limit_mb`,
  `cpu_quota_pct`, `network_isolated`, `mode`) recording what was *actually* applied to that
  specific run, so the transcript is auditable rather than assuming protection was in place.
- Escape hatch: `AGENTIC_PROBE_SANDBOX=off` disables the wrapper entirely, for environments
  where it misbehaves.

This is still a blast-radius backstop, not a full sandbox — no seccomp/container/gVisor,
filesystem confinement is by convention (cwd) only, and GPU VRAM itself is not capped (cgroup
`MemoryMax` covers host RAM, not device memory). Full test suite re-run after the change: 56
passed.

### Status Correction for P0 Items Below

Verified against the current tree (commit `38f0436`), not against the plan text:

| Original P0 item | Status |
|---|---|
| Enforce Role-Specific Tool Access | **Implemented.** `ToolSpec.allowed_roles`, `ToolRegistry.list_tools(role=...)` schema filtering, and `ToolRegistry._authorize` execution-time check all exist in `verifier/agentic/tools/registry.py`. `record_claim` already derives `raised_by` from `context.current_role`, not from an argument. |
| Return Tool Errors to the Workflow | **Implemented.** `ToolRegistry.call` catches `Exception` (not `BaseException`), records one error `ToolEvent`, and returns a structured `{"ok": false, "error": {...}}` result instead of raising. `verifier/agentic_run.py::_run_one_entry` persists state via `orchestrator.persist(stop_reason="fatal_workflow_error")` before re-raising unexpected failures. |
| Close the Claim Evidence/Status Loop | **Implemented.** `ClaimLedger.update_claim_status` in `verifier/agentic/ledger.py` requires matching evidence before a non-open status is accepted. `record_verdict` in `tools/verdict.py` independently blocks `trust`/`reject` while any claim is `open`. |

The sections below are kept because the permission matrix, error-shape examples, and test
lists are still accurate reference material — just no longer a to-do list. Treat their
"Current problem" framing as historical, describing the state before this same commit, not
the state now.

### One Cleanup Found While Verifying the Above — Fixed

`verifier/agentic/agents/base.py::_claim_coverage` computed `open_claim_ids`,
`pending_open_claim_ids`, and `uncovered_open_claim_ids` as the identical list comprehension
three times, evidently left over from the "for prompt compatibility" migration the plan
mentions in the (now-implemented) claim/evidence section. Collapsed to a single
`open_claim_ids` field plus the (distinct, derived) `all_open_claims_have_evidence` boolean.
No skill file or test referenced the two dropped aliases by name — the only existing check
was for the outer `"claim_coverage"` key in the prompt text
(`tests/test_agentic_llm_agent.py:138`), which still passes. Full suite re-run: 56 passed.

### Revised Priority Order

1. Run the system end-to-end against the red-team dataset with a real provider; fix whatever
   that surfaces before anything else here. **Still the top open item.**
2. ~~Add a resource/network limit to probe subprocess execution.~~ Done.
3. ~~Collapse the `_claim_coverage` duplicate fields.~~ Done.
4. Re-audit the "Recommended Order" list further down against current code the same way this
   section did — items 4-7 there (initial description model guarantee, diagram accuracy,
   README wording, pytest collection scope) have not been re-verified here and may or may not
   still be open.

---

## P0: Enforce Role-Specific Tool Access

### Current problem

Every agent currently receives the complete tool registry. Some tools enforce a
role internally, but most rely on prompt instructions.

This means, at the protocol level:

- Skeptic can request a runtime probe.
- Experimenter can attempt to record a verdict.
- Judge can attempt to mutate claims.
- Describer can use tools outside its intended responsibility.

### Intended behavior

The Orchestrator should expose tools according to the current role:

| Role | Main write capabilities |
|---|---|
| Describer | update the description model |
| Skeptic | record claims and record evidence review |
| Experimenter | run probes, attach evidence, update claim status |
| Judge | request more debate or record the final verdict |

Read-only artifact and ledger tools may be shared where useful.

### Implementation plan

This change has two enforcement layers:

```text
tool registration
  -> declares allowed roles
  -> Agent only sees allowed tool schemas
  -> Registry checks the role again before execution
```

The schema filtering improves model behavior. The execution check is the actual
security boundary and prevents an Agent from bypassing the filter by emitting a
tool name from memory or hallucination.

#### 1. Add role metadata to every tool

Extend `ToolSpec` in `verifier/agentic/tools/registry.py`:

```python
@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, JsonValue]
    allowed_roles: frozenset[str]
    handler: ToolHandler
```

Every `registry.register(...)` call must declare `allowed_roles` explicitly.
There should be no implicit "all roles" default for write or execution tools,
because forgetting the field would silently expose a new capability.

#### 2. Filter schemas before building the Agent prompt

Change the registry API from:

```python
registry.list_tools()
```

to:

```python
registry.list_tools(role=agent.role)
```

`AgenticOrchestrator.run_agent_once` passes the active Agent role. The LLM
therefore receives only tools it is allowed to call.

The deterministic artifact-loading turn uses `Role.ORCHESTRATOR`, so dry-run and
initial context loading continue to work without exposing `load_artifact` to
LLM agents.

#### 3. Check authorization again during execution

Before invoking the handler, `ToolRegistry.call` compares
`context.current_role` with `ToolSpec.allowed_roles`.

An unauthorized call returns the structured error format defined in
`Return Tool Errors to the Workflow`:

```json
{
  "ok": false,
  "event_id": "t12",
  "error": {
    "type": "ToolPermissionError",
    "message": "judge is not allowed to call update_claim_status",
    "recoverable": true
  }
}
```

The failed attempt is recorded as one `ToolEvent` and counts toward the tool
budget. It does not invoke the handler or mutate claims, evidence, description
state, or verdict.

This work should be implemented after structured tool-error handling, so a
permission mistake becomes recoverable evidence instead of terminating the
workflow.

#### 4. Initial permission matrix

| Tool | Allowed roles |
|---|---|
| `load_artifact` | Orchestrator |
| `inspect_problem` | Describer, Skeptic, Experimenter, Judge |
| `inspect_kernel_source` | Describer, Skeptic, Experimenter, Judge |
| `list_artifact_files` | Describer, Skeptic, Experimenter, Judge |
| `read_artifact_file` | Describer, Skeptic, Experimenter, Judge |
| `read_claim_ledger` | Describer, Skeptic, Experimenter, Judge |
| `request_description` | Skeptic, Experimenter, Judge |
| `record_description_update` | Describer |
| `record_claim` | Skeptic |
| `record_no_new_claims` | Skeptic |
| `append_evidence` | Experimenter |
| `update_claim_status` | Experimenter |
| `run_python_probe` | Experimenter |
| `run_claim_probe` | Experimenter |
| `finalize_probe_evidence` | Experimenter |
| `retrieve_experiment_history` | Skeptic, Experimenter, Judge |
| `request_more_debate` | Judge |
| `record_verdict` | Judge |

Read access is deliberately broader than mutation access. Later evidence from
real runs may justify narrowing or expanding individual read tools, but write
ownership should remain strict.

#### 5. Derive ownership from runtime context

`record_claim` currently accepts a `raised_by` argument. After role enforcement,
the tool should derive ownership from `context.current_role` instead of trusting
LLM-provided identity:

```python
raised_by = context.current_role
```

The public schema should remove `raised_by`. The same rule applies to future
tools that record an actor: identity comes from the Orchestrator context, not
from tool arguments.

#### 6. Files expected to change

| File | Planned change |
|---|---|
| `verifier/agentic/tools/registry.py` | store role policy, filter schemas, reject unauthorized calls |
| `verifier/agentic/orchestrator.py` | request schemas for the active role |
| `verifier/agentic/tools/claims.py` | derive `raised_by` from runtime context |
| `tests/test_agentic_tool_registry.py` | test schema filtering and execution denial |
| `tests/test_agentic_orchestrator.py` | test role-specific schemas passed to agents |
| `tests/test_agentic_llm_agent.py` | test recoverable unauthorized calls in a workflow |

No changes should be required in `RunState`, the Agent JSON response protocol,
claim status values, evidence format, or debate ordering.

### Test plan

Add focused tests for:

1. Each role sees exactly its permitted tools.
2. Orchestrator can load artifacts but cannot use Agent-owned mutation tools.
3. Judge cannot run probes or mutate claims.
4. Experimenter cannot record a verdict or create claims.
5. Skeptic cannot execute probes or update claim status.
6. Describer cannot create claims or run experiments.
7. A manually emitted unauthorized call is rejected even though its schema was
   hidden from the Agent.
8. An unauthorized call records one error event and changes no domain state.
9. `record_claim` always records the real current role and cannot spoof
   `raised_by`.
10. Existing authorized workflows and dry-run behavior remain unchanged.

### Acceptance criteria

- Each agent receives only its allowed tool schemas.
- The registry also rejects unauthorized calls as a second guard.
- Unauthorized calls use the recoverable structured error format.
- Runtime identity cannot be spoofed through tool arguments.
- Read-only sharing remains possible without exposing mutation tools.
- Tests cover authorized and unauthorized calls for all four roles plus the
  Orchestrator.

## P0: Return Tool Errors to the Workflow

### Current problem

The registry records a failed `ToolEvent` and then raises the exception. The
exception can terminate the current entry before the Agent sees the failure and
before the run is persisted.

This weakens the evidence-driven loop because a failed probe or malformed tool
call should normally become information for the next agent turn.

### Intended behavior

Expected flow:

```text
agent tool call
  -> runtime executes
  -> success or structured error is recorded
  -> result is returned to shared state
  -> agent may retry, revise the claim, or mark it inconclusive
```

Fatal runtime errors should still stop the entry, but the partial state and
transcript must be persisted first.

### Implementation plan

Keep the change small. Successful tool calls retain their current return
format. Only failed calls receive a new structured result:

```json
{
  "ok": false,
  "event_id": "t8",
  "error": {
    "type": "ToolRegistryError",
    "message": "tool run_claim_probe missing required arg: claim_id",
    "recoverable": true
  }
}
```

The same object is stored in the failed `ToolEvent.output`, so the immediate
caller, later agents, `run.json`, `tool_events.jsonl`, and `transcript.md` all
see the same error representation.

#### 1. Change the registry error boundary

Update `ToolRegistry.call` in `verifier/agentic/tools/registry.py`:

1. Allocate the tool-event ID before tool lookup and argument validation.
2. Count every attempted tool call, including failed calls, toward the tool
   budget.
3. Move unknown-tool and schema validation into the protected execution block.
4. On a normal Python `Exception`, append one error `ToolEvent` and return its
   structured error output instead of re-raising.
5. Restore `current_tool_event_id` in `finally`, as the implementation does
   today.
6. Do not catch `BaseException`; `KeyboardInterrupt`, `SystemExit`, and process
   termination must still propagate.

The registry must append exactly one event for each attempted call. Success and
error paths must not create duplicate event IDs.

#### 2. Preserve normal Orchestrator control flow

`AgenticOrchestrator.apply_agent_response` should continue collecting tool
outputs in its existing format:

```json
{
  "tool": "run_claim_probe",
  "output": {
    "ok": false,
    "event_id": "t8",
    "error": {
      "type": "ValueError",
      "message": "probe code must be non-empty",
      "recoverable": true
    }
  }
}
```

No new agent-response protocol is needed. The failed event is already part of
`RunState.tool_events`, so it appears in the next agent prompt. The next agent
turn may correct the call, choose another tool, or attach inconclusive evidence
when the environment cannot complete the experiment.

#### 3. Do not treat failed probes as unconsumed results

The claim-coverage loop currently tracks probe events that still need
Experimenter interpretation. After this change:

- a successful `run_claim_probe` result remains consumable by
  `finalize_probe_evidence`;
- a failed `run_claim_probe` event is visible history, but is not a consumable
  evidence draft;
- the failed attempt still consumes tool budget and may justify a retry;
- repeated failures eventually stop through the existing claim-round or tool
  budget instead of creating an infinite consumption loop.

Update the probe-event checks in `verifier/agentic/orchestrator.py` to require a
successful event with a valid claim-bound result before classifying it as
unconsumed.

#### 4. Persist unexpected workflow failures

The registry change handles failures inside tools. LLM provider failures,
response parsing bugs, and unexpected Orchestrator exceptions can still escape
the tool boundary.

Update `verifier/agentic_run.py` so `_run_one_entry` persists the current state
before propagating an unexpected failure. The partial run should use a clear
stop reason such as:

```text
fatal_workflow_error
```

This fallback must not write a false verdict. It only preserves the transcript,
claims, tool events, and available probe artifacts for debugging.

#### 5. Files expected to change

| File | Planned change |
|---|---|
| `verifier/agentic/tools/registry.py` | return and record structured tool errors |
| `verifier/agentic/orchestrator.py` | exclude failed probes from unconsumed probe results |
| `verifier/agentic_run.py` | persist partial state on unexpected workflow failure |
| `tests/test_agentic_tool_registry.py` | test error return and event recording |
| `tests/test_agentic_orchestrator.py` | test failed-probe recovery and retry behavior |
| `tests/test_agentic_llm_agent.py` | test workflow continuation after a failed tool call |

No changes should be required in agent prompts, the JSON agent-response schema,
claim/evidence data classes, or existing successful tool outputs.

### Test plan

Add focused tests for:

1. An unknown tool returns a structured error and records one failed event.
2. Missing or unexpected arguments return a structured error.
3. A handler `ValueError` returns an error without terminating the agent loop.
4. The next agent turn can see the failed event and retry successfully.
5. Failed calls count toward the tool budget.
6. Failed `run_claim_probe` events are not reported as unconsumed evidence
   drafts.
7. A successful probe after a failed attempt can still be finalized normally.
8. An exception outside the tool boundary persists a partial transcript before
   it propagates.
9. `KeyboardInterrupt` and `SystemExit` are not swallowed.

### Acceptance criteria

- Normal tool errors are returned as structured results instead of crashing the
  debate.
- Successful tool return formats remain backward-compatible.
- Every attempted tool call creates exactly one `ToolEvent`.
- Failed probes do not enter the probe-consumption path.
- The next agent turn can inspect and react to a failed tool event.
- Unexpected fatal errors persist the partial run with an explicit stop reason.
- Batch runs retain a transcript for every failed entry.

## P0: Close the Claim Evidence/Status Loop

### Current problem

An open claim is considered covered as soon as it has any evidence. If evidence
is appended without updating the claim status, the Experimenter coverage loop
may stop while the claim remains `open`.

`finalize_probe_evidence` already updates evidence and status together, but the
lower-level evidence tools can still create an incomplete state.

### Intended behavior

Before Judge runs, every relevant claim should be in one of these states:

```text
confirmed | rebutted | inconclusive
```

An `open` claim should mean that investigation is still required, regardless of
whether a partial observation has already been attached.

### Minimal rule set

Do not add another claim tool or a collection of operator-specific rules. The
workflow only needs two general invariants:

```text
Rule 1: status=open always means pending.
Rule 2: a terminal status requires evidence supporting that status.
```

The second rule applies uniformly:

| Requested status | Required evidence |
|---|---|
| `confirmed` | at least one evidence entry with `supports=confirmed` |
| `rebutted` | at least one evidence entry with `supports=rebutted` |
| `inconclusive` | at least one evidence entry with `supports=inconclusive` |

These are protocol-consistency rules, not fixed verification policy. Agents
still decide what to investigate, what evidence means, and which terminal status
is appropriate.

### Implementation plan

#### 1. Treat every open claim as pending

Replace the current "open and has no evidence" coverage check with a status-only
check:

```python
pending_claim_ids = [
    claim.id
    for claim in state.claims
    if claim.status == ClaimStatus.OPEN
]
```

The claim-coverage loop continues while this list is non-empty. Partial evidence
does not close a claim.

For prompt compatibility, `claim_coverage` may continue to expose
`uncovered_open_claim_ids` temporarily, but the workflow decision must use
`pending_claim_ids`. The old field should eventually be removed or renamed so
its meaning is not ambiguous.

#### 2. Validate evidence before changing status

Keep `append_evidence` and `update_claim_status`; do not add another public tool.

Before `ClaimLedger.update_claim_status` changes an open claim to a terminal
status, verify that the claim contains evidence whose `supports` value matches
the requested status. If not, return a recoverable `LedgerError`.

This keeps the two-call source-evidence path safe:

```text
append_evidence
  -> claim remains open
update_claim_status
  -> succeeds only when matching evidence exists
```

If the second call is omitted or fails, the claim remains open and the
Orchestrator schedules the Experimenter again.

#### 3. Keep probe finalization atomic

The runtime-probe path remains unchanged:

```text
run_claim_probe
  -> finalize_probe_evidence
  -> append evidence + update status
```

`finalize_probe_evidence` already performs both mutations in one handler. It
should continue to be the preferred path for probe results.

#### 4. Gate final verdicts

Under the normal workflow, Judge runs only after no pending claims remain.

Add a final guard to prevent `trust` or `reject` while any claim is still open.
`needs_more_evidence` may still be recorded for an intentionally incomplete run,
for example when claim coverage is disabled or the available runtime cannot
resolve an important claim.

This guard is only a consistency fallback. The Orchestrator remains responsible
for normal scheduling.

#### 5. Files expected to change

| File | Planned change |
|---|---|
| `verifier/agentic/orchestrator.py` | treat all open claims as pending |
| `verifier/agentic/agents/base.py` | expose unambiguous pending claim IDs in prompt state |
| `verifier/agentic/ledger.py` | require matching evidence before terminal status |
| `verifier/agentic/tools/verdict.py` | block trust/reject when open claims remain |
| `tests/test_agentic_ledger.py` | test evidence-backed status transitions |
| `tests/test_agentic_orchestrator.py` | test partial-evidence claims remain pending |
| `tests/test_agentic_llm_agent.py` | test Experimenter is recalled before Judge |

No new tool, claim status, evidence kind, or Agent role is required.

### Test plan

Add focused tests for:

1. An open claim without evidence remains pending.
2. An open claim with partial evidence also remains pending.
3. A terminal status without evidence is rejected.
4. A status with evidence supporting a different status is rejected.
5. Matching evidence allows `confirmed`, `rebutted`, and `inconclusive`.
6. `finalize_probe_evidence` still updates evidence and status atomically.
7. Experimenter is called again after evidence is attached without a status
   update.
8. Judge cannot record `trust` or `reject` while an open claim remains.
9. `needs_more_evidence` remains available for intentionally incomplete runs.

### Acceptance criteria

- Open claims are always considered pending, with or without evidence.
- Terminal statuses require matching evidence.
- Judge cannot record `trust` or `reject` while a claim remains `open`.
- Probe evidence is normally attached through the atomic finalization tool.
- Incomplete source-evidence updates leave the claim visibly pending.
- No additional public claim tool is introduced.

## P1: Guarantee the Initial Description Model

### Current problem

The Describer is called before the Skeptic, but the Orchestrator does not require
the Describer to record a structured description update. The workflow can
continue with an empty description model.

The on-demand clarification path is implemented, but it depends on another
agent noticing ambiguity and calling `request_description`.

### Intended behavior

The first Skeptic turn should receive at least a minimal model containing:

- benchmark/input contract;
- observed kernel behavior and assumptions;
- likely risk areas;
- scope notes;
- unresolved questions.

Later agents may request focused clarification when new evidence changes the
interpretation.

### Acceptance criteria

- The initial Describer stage must either populate the model or record why it
  could not.
- Open description tasks are resolved before a dependent claim or verdict
  proceeds.
- Tests cover initial description and on-demand clarification.

## P1: Align the Diagrams with the Implementation

### `overall_pipeline.drawio.svg`

Known issues:

- `Author` should be `Describer`.
- The clarification label is clipped.
- The tools box lists a standalone result-comparison capability that is not a
  registered tool.
- Direct arrows between agents can imply direct communication, although agents
  communicate through Orchestrator-managed state.
- Judge is shown only as producing a verdict and not as requesting more debate.
- The description model and on-demand clarification path are not represented.

### `Orchestrator.drawio.svg`

Known issues:

- Evidence is shown as a separate ledger, but the current data model stores
  evidence inside each claim.
- The diagram says the Orchestrator selects tools. Agents select tools;
  Orchestrator validates and executes them.
- Judge is labeled read-only even though it writes convergence requests and the
  final verdict.
- Several displayed run-context fields do not exist in the current `RunState`.
- Description model, description tasks, history, tool events, and skeptic
  review are missing from the shared-state view.
- Some labels contain spelling errors or are clipped.

### Rendering issue

Both SVG files use a transparent background with dark text and lines. In dark
mode, important content can disappear. The diagrams should use an explicit
background or define reliable dark-mode colors.

### Acceptance criteria

- Diagram role names and arrows match the actual control flow.
- Shared-state boxes match the current `RunState`.
- The evidence loop shows Agent-selected tool calls and runtime execution.
- Both images remain readable in GitHub light and dark themes.

## P1: Complete Descriptor Documentation

The README mentions the description model, but the detailed tool and repository
tables do not yet include:

- `request_description`;
- `record_description_update`;
- `verifier/agentic/tools/description.py`;
- description tasks and updates persisted in `RunState`.

These should be added with small wording changes after the architecture and
diagrams are settled.

## P2: Make Agent Role Ownership Explicit

### Current problem

`_first_agent_with_role` searches a list and silently returns the first matching
Agent. That hides the actual invariant that each role has one responsible Agent.
If two Judges or two Experimenters are accidentally passed, the second one is
ignored without an error.

### Intended behavior

Build a role-to-Agent mapping once when the workflow starts:

```text
agents list
  -> validate one Agent per role
  -> agents_by_role
  -> direct role lookup
```

Duplicate roles should fail immediately. Optional roles may be absent only when
the selected workflow does not require them. The runtime should never silently
choose the first duplicate.

### Implementation plan

- Replace `_first_agent_with_role` with a mapping builder and duplicate-role
  validation.
- Use direct lookups for Describer, Skeptic, Experimenter, and Judge.
- Keep the current list input at the CLI boundary so the public CLI does not
  need to change.
- Validate required roles before entering the workflow loop.
- Update tests for unique roles, duplicate roles, optional roles, and the
  existing sequential debug runner.

This is a readability and configuration-safety improvement. It does not change
the debate policy or introduce multi-Agent role coordination.

### Acceptance criteria

- A complete workflow has exactly one Agent for each required role.
- Duplicate roles fail with an explicit configuration error.
- Missing optional roles retain the existing debug/fallback behavior.
- No code path silently selects the first Agent when duplicates exist.

## P2: Restrict Pytest Collection

Running `pytest` at the repository root currently collects tests from vendored
KernelBench code, dataset artifacts, and historical kernel logs. The project
tests pass when run explicitly with:

```bash
pytest tests/
```

Add a pytest collection boundary such as:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

This is an engineering cleanup issue, not a verification-logic failure.

## Recommended Order

Superseded by "Revised Priority Order" in the Independent Re-Review section above for items
1-3. Re-verified during this pass:

1. ~~Add role-specific tool authorization.~~ Done (see Status Correction table above).
2. ~~Make tool failures recoverable and persist partial runs.~~ Done (see Status Correction
   table above).
3. ~~Require resolved claim statuses before Judge.~~ Done (see Status Correction table above).
4. Enforce the initial description model. **Not re-verified in this pass** — the on-demand
   `request_description`/`record_description_update` path exists and is exercised by tests,
   but whether the Describer is *guaranteed* to populate the model before the first Skeptic
   turn (rather than only on request) was not checked against current
   `orchestrator.run_verification_workflow`.
5. Correct and re-export both diagrams. **Not re-verified** — `assets/readme/*.drawio.svg`
   were re-exported in commit `38f0436` per its diff, but this pass did not open the SVGs to
   confirm the "Author" vs. "Describer" label and other issues listed above are actually fixed.
6. Apply the small corresponding README updates. **Not re-verified** in detail; README.md was
   substantially rewritten in `38f0436` and reads as agentic-architecture-first, but was not
   line-checked against every item in the original P1 sections.
7. Restrict pytest collection. **Confirmed still open** — `pyproject.toml` has no
   `[tool.pytest.ini_options]` / `testpaths` entry as of this pass.

## Prototype Completion Boundary

The prototype can be considered architecturally complete when, in addition to the items
already satisfied below, the two P(-1) findings above are closed:

- role boundaries are enforced by code rather than prompts alone — **done**;
- every claim reaches an evidence-backed terminal status before verdict — **done**;
- failed tools remain visible to agents and in persisted transcripts — **done**;
- the description model is guaranteed and can be refreshed on demand — not re-verified;
- README diagrams accurately represent the implemented state and control flow — not
  re-verified;
- the project test command runs without collecting external artifacts — **still open**;
- **at least one real end-to-end run, with a real LLM provider, has produced a correct verdict
  on a known red-team dataset entry** — not done, not previously required by this document,
  and now considered the actual gate before calling this a working verifier rather than a
  well-tested state machine.
- **probe execution has a resource/network limit beyond a wall-clock timeout** — **done** (CPU
  ulimit, cgroup memory cap where available, netns network block where available, env hygiene,
  process-group kill on timeout).
