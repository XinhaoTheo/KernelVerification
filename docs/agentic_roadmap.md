# KernelVerification Agentic Refactor Roadmap

## 0. Goal

把 KernelVerification 从一个写死检查流程的 verification pipeline，重构成一个 agentic kernel verification framework。

新的系统不预设“必须跑 standard recheck / robustness battery / precision recheck”。这些固定检查会被删除或废弃。系统只提供：

- agents：负责推理、提出假设、解释证据、下 verdict；
- skills：规定 agentic verification 的工作流和证据纪律；
- tools：提供本地实验、源码读取、claim ledger 更新等能力；
- orchestrator：负责 LLM 调用、tool dispatch、history、ledger、artifact persistence。

核心原则：

> Agents decide what to investigate.  
> Skills constrain how investigation should be done.  
> Tools provide executable capabilities.  
> Runtime executes tools locally and returns evidence.

## 1. What We Are Removing

旧系统的问题不是实现不够好，而是方向错了：它把 verification policy 写死在代码里，让程序决定要检查什么。

以下模块不再作为新主系统的一部分：

- `verifier/run.py` 的固定流程：load -> recheck -> precision_recheck -> debate -> combine。
- `verifier/recheck.py` 作为主流程固定 correctness check。
- `verifier/precision_recheck.py` 作为主流程固定 operator-class check。
- `verifier/classify.py` 作为固定 operator router。
- `verifier/verdict.py` 的 rule-based combine。
- `agents/author.py`, `agents/skeptic.py`, `agents/verifier.py`, `agents/judge.py` 的旧顺序 debate API。
- README / roadmap 中把 fixed checks 当成核心阶段的叙述。

部分旧代码可以临时作为参考或迁移素材，但目标系统不应该依赖固定 check policy。

## 2. What We Keep

保留不包含 verification policy 的基础设施：

- `verifier/dataset.py`：加载 problem/kernel/test/seeds，后续可改名为 artifact loader。
- `verifier/generator.py` 和 `verifier/build_dataset.py`：继续用于从 KernelAgent 生成 kernel artifact。
- `verifier/gpu_pick.py`：本地 CUDA tool 需要选择可用 GPU。
- `verifier/llm_client.py`：可重构成 agentic LLM client。
- `dataset/`：作为 benchmark artifacts 和 regression fixtures，而不是固定检查策略的来源。

## 3. Target Architecture

目标目录结构：

```text
verifier/
  agentic/
    __init__.py

    orchestrator.py
    llm.py
    state.py
    ledger.py
    persistence.py

    agents/
      __init__.py
      describer.py
      skeptic.py
      experimenter.py
      judge.py

    tools/
      __init__.py
      registry.py
      artifacts.py
      source.py
      execution.py
      claims.py
      history.py

    skills/
      kernel-verification.md
      evidence-driven-review.md
      claim-lifecycle.md
      experiment-design.md
      convergence.md

  agentic_run.py
```

New CLI:

```text
kv-agentic-run <entry>
```

Old `kv-run`, `kv-recheck`, and `kv-precision-recheck` have been removed from the primary package. Use `kv-agentic-run`.

## 4. Core Runtime Model

### 4.1 RunState

`RunState` is the durable state for one verification run:

```python
{
    "entry": "softmax",
    "artifact": {...},
    "skills": [...],
    "history": [...],
    "tool_events": [...],
    "claims": [...],
    "verdict": {...} | None,
}
```

### 4.2 Turn

Each agent turn records reasoning output and any requested tool calls:

```python
{
    "role": "skeptic",
    "round": 1,
    "text": "...",
    "tool_calls": [...],
}
```

### 4.3 ToolEvent

Every tool execution is recorded:

```python
{
    "id": "t7",
    "tool": "run_python_probe",
    "args": {...},
    "status": "ok" | "error",
    "output": {...},
    "created_at": "...",
}
```

### 4.4 Claim

Claims are the center of the system:

```python
{
    "id": "c3",
    "statement": "For a stride-2 input tensor, the kernel reads contiguous memory and returns wrong values.",
    "rationale": "The source uses pointer arithmetic that ignores stride.",
    "status": "open" | "confirmed" | "rebutted" | "inconclusive",
    "raised_by": "skeptic",
    "evidence": [...]
}
```

### 4.5 Evidence

Evidence must come from source inspection or runtime tool output:

```python
{
    "kind": "runtime_probe" | "source_inspection" | "tool_error",
    "tool_event_id": "t7",
    "summary": "Probe returned max_abs_diff=0.42 on valid stride-2 input.",
    "supports": "confirmed" | "rebutted" | "inconclusive",
    "data": {...}
}
```

## 5. Agent Roles

### 5.1 Describer

Responsibilities:

- Explain what the kernel appears to implement.
- Identify key assumptions visible in source code.
- Cite concrete source lines or snippets.
- Avoid judging correctness.

It may call:

- `load_artifact`
- `inspect_kernel_source`
- `read_file_fragment`

### 5.2 Skeptic

Responsibilities:

- Propose concrete bug hypotheses.
- Turn hypotheses into specific, testable claims.
- Avoid generic doubt.
- Use existing evidence to refine or drop hypotheses.

It may call:

- `record_claim`
- `read_claim_ledger`
- `inspect_kernel_source`
- `retrieve_experiment_history`

### 5.3 Experimenter

Responsibilities:

- Decide what evidence is needed for open claims.
- Request local probes through tools.
- Interpret tool output only after runtime returns it.
- Update claim status with evidence.

It may call:

- `run_python_probe`
- `run_cuda_probe`
- `read_tool_event`
- `update_claim_status`
- `append_evidence`

### 5.4 Judge

Responsibilities:

- Read the final claim ledger and evidence.
- Decide one final verdict.
- Prefer measured evidence over rhetoric.
- Mark insufficient evidence explicitly.

Output:

```json
{
  "verdict": "trust" | "reject" | "needs_more_evidence",
  "confidence": 0.0,
  "decisive_claims": ["c1"],
  "reason": "..."
}
```

## 6. Skills

Skills are markdown workflow instructions loaded into agent context. They are not code and should not encode a fixed test sequence.

### 6.1 `kernel-verification.md`

Defines the overall process:

- understand artifact;
- raise concrete claims;
- gather evidence;
- update ledger;
- judge only from evidence.

### 6.2 `evidence-driven-review.md`

Rules:

- No claim should be considered confirmed without evidence.
- Evidence must cite source or tool output.
- Absence of failure is not proof unless the tested condition matches the claim.
- Runtime crashes on valid inputs are evidence.
- Tool failures are not kernel failures unless the failure happens inside a valid kernel invocation.

### 6.3 `claim-lifecycle.md`

Claim statuses:

- `open`: hypothesis recorded, not yet tested.
- `confirmed`: evidence supports the claim.
- `rebutted`: evidence contradicts the claim.
- `inconclusive`: experiment cannot decide or evidence is insufficient.

### 6.4 `experiment-design.md`

Guidance for good probes:

- construct minimal inputs targeting one hypothesis;
- separate reference computation from candidate kernel invocation;
- print structured metrics;
- avoid changing the claim while testing it;
- keep probes reproducible.

This skill may mention known kernel bug patterns, such as:

- shape specialization;
- stride / contiguity assumptions;
- boundary sizes;
- dtype assumptions;
- reduction order and accumulation precision;
- index selection bugs;
- fake GPU work;
- stateful caching or hardcoded outputs.

These are heuristics for agent reasoning, not fixed batteries.

### 6.5 `convergence.md`

Stop when:

- no open claims remain and all important hypotheses are resolved; or
- remaining claims are inconclusive and no available tool can resolve them; or
- max rounds/tool budget is reached.

Do not stop merely because a fixed checklist was completed.

## 7. Tool Layer

Tools expose local runtime capabilities. They should be narrow, structured, and auditable.

### 7.1 Artifact Tools

- `load_artifact(entry)`
- `inspect_kernel_source(entry, start_line=None, end_line=None)`
- `inspect_problem(entry)`
- `list_artifact_files(entry)`
- `read_artifact_file(entry, path)`

### 7.2 Claim Tools

- `record_claim(statement, rationale, raised_by)`
- `read_claim_ledger()`
- `update_claim_status(claim_id, status)`
- `append_evidence(claim_id, evidence)`

### 7.3 Execution Tools

- `run_python_probe(code, timeout_s=60, use_gpu=True)`
- `run_cuda_probe(code, timeout_s=60)`
- `run_shell_limited(argv, timeout_s=60)` only if needed and tightly constrained.

Execution tools must:

- run locally, not inside the LLM;
- capture stdout/stderr/exit code;
- write probe artifacts to a run directory;
- return structured output;
- never let the agent execute arbitrary filesystem operations directly.

### 7.4 History Tools

- `retrieve_experiment_history(entry)`
- `read_previous_tool_events(entry)`

These help avoid repeating experiments without turning history into evidence unless the relevant artifact is loaded.

## 8. Orchestrator

The orchestrator owns the loop:

```text
load artifact
load skills
initialize RunState
for round in max_rounds:
    call describer if needed
    call skeptic
    execute requested claim/source tools
    call experimenter for open claims
    execute requested runtime tools
    update ledger
    check convergence
call judge
persist result
```

The orchestrator, not the agent, is responsible for:

- validating tool call schema;
- dispatching local tools;
- appending tool events;
- applying claim status updates;
- enforcing max rounds/time/tool budget;
- writing final run artifacts.

## 9. LLM Tool-Call Protocol

To avoid provider lock-in at first, use a JSON protocol in the assistant response:

```json
{
  "message": "I suspect a stride handling bug.",
  "tool_calls": [
    {
      "tool": "record_claim",
      "args": {
        "statement": "For x=torch.randn(2048)[::2], the kernel ignores stride and returns wrong output.",
        "rationale": "Pointer arithmetic appears contiguous."
      }
    }
  ]
}
```

Later this can be swapped to native Anthropic/OpenAI tool calling without changing tool implementations.

## 10. Persistence

Each run writes:

```text
dataset/<entry>/agentic_runs/<timestamp>/
  run.json
  tool_events.jsonl
  claims.json
  verdict.json
  probes/
    t7_probe.py
    t7_stdout.txt
    t7_stderr.txt
```

The final `run.json` should be enough to audit why the verdict was reached.

## 11. Implementation Phases

### Phase 1: Skeleton

Deliver:

- `verifier/agentic/state.py`
- `verifier/agentic/ledger.py`
- `verifier/agentic/tools/registry.py`
- initial skill markdown files
- tests for ledger and registry

No LLM calls required yet.

### Phase 2: Artifact and Claim Tools

Deliver:

- artifact loading tools;
- source inspection tools;
- claim ledger tools;
- structured tool event recording.

Acceptance:

- a unit test can load an entry, inspect source, record a claim, append evidence, and serialize state.

### Phase 3: Execution Tools

Deliver:

- `run_python_probe`;
- optional GPU pinning via `gpu_pick`;
- probe artifact persistence;
- timeout/error handling.

Acceptance:

- a test probe can import `kernel.py`, run locally, and return structured stdout/stderr/exit code.

### Phase 4: Agent Loop

Deliver:

- minimal describer/skeptic/experimenter/judge agents;
- JSON tool-call parser;
- orchestrator loop;
- `kv-agentic-run`.

Acceptance:

- `kv-agentic-run elem_add --dry-run` shows loaded skills and available tools.
- `kv-agentic-run elem_add --max-rounds 1` produces a valid `run.json`.

### Phase 5: Delete Old Pipeline

Status: implemented.

Delivered:

- removed old fixed-check CLIs;
- removed old fixed-check code paths;
- updated README around agentic architecture;
- kept generation/data loading docs.

Acceptance:

- primary docs no longer describe standard/precision recheck as mandatory pipeline stages.
- verification entry point routes through agentic orchestrator.

### Phase 6: Harden

Deliver:

- budgets;
- tool allowlist;
- better schema validation;
- replay mode from `run.json`;
- experiment history retrieval;
- richer skills.

Acceptance:

- failed tool calls are auditable;
- judge can cite exact claim/evidence IDs;
- runs are reproducible enough for debugging.

## 12. Testing Strategy

Unit tests:

- ledger status transitions;
- tool registry schema validation;
- artifact loader;
- execution tool timeout/error behavior;
- persistence round-trip.

Integration tests:

- fake LLM agent returns deterministic tool calls;
- orchestrator executes tools and updates ledger;
- judge receives final ledger.

No test should assert that a fixed verification battery is always run.

## 13. Non-Goals

- Do not rebuild precision recheck as a hidden default tool call.
- Do not keep a fixed checklist in the orchestrator.
- Do not let agents execute arbitrary Python/filesystem operations directly.
- Do not make the judge a rule-based combiner.
- Do not treat passing one probe as proof of global correctness.

## 14. First Concrete PR Scope

The first implementation chunk should be small:

1. Add agentic state dataclasses.
2. Add claim ledger operations.
3. Add tool registry with two no-risk tools:
   - `load_artifact`
   - `record_claim`
4. Add initial skill markdown.
5. Add tests for the above.

This creates the foundation without touching GPU execution or LLM APIs.

## 15. Data Schema Specification

The agentic system must treat its run artifacts as structured data, not as loose logs. Every object below should be serializable to JSON and stable enough for judge prompting, replay, debugging, benchmark analysis, and future UI rendering.

Do not use unstructured free-form dictionaries unless the field is explicitly marked as tool-specific.

### 15.1 Common Scalar Types

Use these conventions everywhere:

```text
Id              string, stable within one run, prefixed by object type: "c1", "e3", "t7"
Timestamp       ISO-8601 UTC string, e.g. "2026-07-01T15:04:05Z"
Role            "describer" | "skeptic" | "experimenter" | "judge" | "orchestrator" | "tool"
Status          lower-case enum string
Path            repo-relative or run-directory-relative string
LineNumber      1-based integer
JsonValue       string | number | boolean | null | list | object
```

### 15.2 TensorSpec

Use `TensorSpec` whenever a tool or evidence describes an input/output tensor:

```json
{
  "name": "x",
  "shape": [1024],
  "dtype": "float32",
  "device": "cuda",
  "stride": [2],
  "layout": "non_contiguous",
  "requires_grad": false,
  "value_summary": {
    "kind": "random_normal",
    "seed": 0,
    "min": -2.41,
    "max": 2.87,
    "mean": 0.03
  }
}
```

Fields:

- `name`: variable name used in the probe.
- `shape`: tensor dimensions.
- `dtype`: PyTorch-style dtype string, without `torch.` prefix when possible.
- `device`: `"cpu"` or `"cuda"`; include device index only if relevant, e.g. `"cuda:1"`.
- `stride`: tensor stride if known.
- `layout`: one of `"contiguous"`, `"non_contiguous"`, `"broadcasted"`, `"transposed"`, `"sliced"`, `"unknown"`.
- `requires_grad`: optional; default false.
- `value_summary`: optional compact summary. Do not store large tensor values inline.

### 15.3 MetricBlock

Use `MetricBlock` for numerical comparisons:

```json
{
  "matches": false,
  "max_abs_diff": 0.42,
  "max_rel_diff": 0.31,
  "mean_abs_diff": 0.08,
  "rel_l2": 0.13,
  "num_mismatched": 117,
  "numel": 1024,
  "rtol": 0.001,
  "atol": 0.001
}
```

Fields are optional except where a tool explicitly requires them. `matches` means the candidate output matched the reference under the comparison policy used by that probe. It is not a global correctness verdict.

### 15.4 ArtifactRef

Use `ArtifactRef` when evidence points to generated files:

```json
{
  "kind": "probe_code",
  "path": "probes/t7_probe.py",
  "sha256": "optional-content-hash",
  "description": "Python probe generated by experimenter for claim c1"
}
```

Fields:

- `kind`: `"probe_code"`, `"stdout"`, `"stderr"`, `"source_snapshot"`, `"json_result"`, `"other"`.
- `path`: path relative to the run directory.
- `sha256`: optional, useful once replay is implemented.
- `description`: human-readable purpose.

## 16. RunState Schema

`RunState` is the top-level persisted object for one agentic verification run:

```json
{
  "schema_version": 1,
  "run_id": "run_20260701_150405",
  "entry": "softmax",
  "created_at": "2026-07-01T15:04:05Z",
  "artifact": {
    "entry": "softmax",
    "dataset_dir": "dataset",
    "problem_path": "dataset/softmax/problem.txt",
    "kernel_path": "dataset/softmax/kernel.py",
    "test_path": "dataset/softmax/test.py",
    "session_dir": "dataset/softmax"
  },
  "skills": [
    {"name": "kernel-verification", "path": "verifier/agentic/skills/kernel-verification.md"}
  ],
  "history": [],
  "tool_events": [],
  "claims": [],
  "verdict": null
}
```

Required fields:

- `schema_version`: integer; bump when persisted schema changes incompatibly.
- `run_id`: unique run identifier.
- `entry`: dataset entry under review.
- `created_at`: run creation timestamp.
- `artifact`: paths and metadata for loaded kernel artifact.
- `skills`: skills loaded into agent context.
- `history`: agent turns.
- `tool_events`: all tool executions.
- `claims`: claim ledger.
- `verdict`: final judge output, null until the judge runs.

## 17. Turn Schema

Each agent response becomes a `Turn`:

```json
{
  "id": "turn12",
  "role": "skeptic",
  "round": 2,
  "created_at": "2026-07-01T15:08:10Z",
  "message": "The kernel appears to assume contiguous x because the address expression ignores stride.",
  "tool_calls": [
    {
      "id": "call1",
      "tool": "record_claim",
      "args": {
        "statement": "For x=torch.randn(2048)[::2], the kernel ignores stride and returns wrong output.",
        "rationale": "The source address expression uses offsets only and does not account for x.stride()."
      }
    }
  ],
  "raw_response": "optional raw model text"
}
```

Fields:

- `id`: unique turn ID.
- `role`: agent role.
- `round`: orchestrator round index.
- `created_at`: timestamp.
- `message`: natural-language content visible to other agents.
- `tool_calls`: structured tool requests extracted from the model response.
- `raw_response`: optional raw model output for debugging.

## 18. ToolEvent Schema

`ToolEvent` records one tool execution by the local runtime:

```json
{
  "id": "t7",
  "call_id": "call1",
  "tool": "run_python_probe",
  "args": {
    "claim_id": "c1",
    "code": "...",
    "timeout_s": 60,
    "use_gpu": true
  },
  "status": "ok",
  "output": {
    "exit_code": 0,
    "timed_out": false,
    "stdout_excerpt": "{\"verdict\":\"mismatch\",\"metrics\":{\"max_abs_diff\":0.42}}",
    "stderr_excerpt": "",
    "parsed_result": {
      "verdict": "mismatch",
      "metrics": {"matches": false, "max_abs_diff": 0.42}
    }
  },
  "error": null,
  "artifacts": [
    {"kind": "probe_code", "path": "probes/t7_probe.py"}
  ],
  "created_at": "2026-07-01T15:09:00Z",
  "finished_at": "2026-07-01T15:09:04Z"
}
```

Required fields:

- `id`: tool event ID, e.g. `t7`.
- `call_id`: ID of the model-requested tool call that caused this event.
- `tool`: registered tool name.
- `args`: validated tool args after schema checking.
- `status`: `"ok"` or `"error"` for the tool execution itself.
- `output`: structured output when `status="ok"`.
- `error`: structured error when `status="error"`, otherwise null.
- `artifacts`: files written by the tool, using `ArtifactRef`.
- `created_at`, `finished_at`: timestamps.

Important distinction:

- `ToolEvent.status="error"` means the tool failed.
- It does not automatically mean the kernel is wrong.
- A tool failure only becomes claim evidence after the experimenter or orchestrator maps it into an `Evidence` object.

### 18.1 ToolError

Use this shape for `ToolEvent.error`:

```json
{
  "type": "timeout",
  "message": "Probe exceeded 60 seconds.",
  "phase": "probe_process",
  "kernel_fault_possible": false,
  "retryable": true
}
```

Fields:

- `type`: `"timeout"`, `"schema_error"`, `"file_not_found"`, `"probe_error"`, `"runtime_error"`, `"internal_error"`.
- `message`: short human-readable message.
- `phase`: `"schema_validation"`, `"artifact_loading"`, `"probe_process"`, `"reference_computation"`, `"candidate_kernel_invocation"`, `"result_parsing"`, `"unknown"`.
- `kernel_fault_possible`: true only if the error plausibly occurred during a valid candidate kernel invocation.
- `retryable`: whether a better tool call could plausibly resolve it.

## 19. Claim Schema

`Claim` is a testable hypothesis about the kernel:

```json
{
  "id": "c1",
  "statement": "For x=torch.randn(2048)[::2], the kernel ignores stride and returns wrong output.",
  "rationale": "The source address expression uses offsets only and does not account for x.stride().",
  "status": "open",
  "raised_by": "skeptic",
  "created_at": "2026-07-01T15:08:10Z",
  "updated_at": "2026-07-01T15:08:10Z",
  "evidence_ids": [],
  "evidence": [],
  "tags": ["stride", "non_contiguous"],
  "supersedes": [],
  "superseded_by": null
}
```

Required fields:

- `id`: claim ID, e.g. `c1`.
- `statement`: concrete, testable hypothesis. It should name the condition under which the kernel may fail.
- `rationale`: why the claim was raised before verification. This is not final evidence.
- `status`: `"open"`, `"confirmed"`, `"rebutted"`, or `"inconclusive"`.
- `raised_by`: role that raised it.
- `created_at`, `updated_at`: timestamps.
- `evidence_ids`: IDs of evidence attached to this claim.
- `evidence`: embedded evidence objects for single-file readability.

Optional fields:

- `tags`: short labels such as `"stride"`, `"dtype"`, `"boundary"`, `"race"`, `"fake-triton"`.
- `supersedes`: claim IDs this claim replaces.
- `superseded_by`: newer claim ID if this claim was refined.

### 19.1 Claim Statement Requirements

A claim statement must be specific enough that an experimenter can design a probe.

Good:

```text
For x=torch.randn(2048)[::2] with shape [1024] and stride [2], the kernel reads contiguous memory and returns values for x[:1024] instead of x[::2].
```

Bad:

```text
The kernel may not be robust.
```

If a concern is important but not yet testable, record it as agent reasoning, not as a claim, or create an `open` claim whose first requested action is source inspection.

### 19.2 Rationale vs Evidence

`rationale` is the pre-verification reason for suspicion:

```json
"rationale": "The source uses base + offsets for tl.load and no visible stride parameter."
```

`evidence` is the post-verification support or rebuttal:

```json
{
  "kind": "runtime_probe",
  "summary": "Stride-2 probe returned matches=false and max_abs_diff=0.42.",
  "supports": "confirmed"
}
```

The judge should treat `rationale` as context and `evidence` as the primary basis for verdict.

## 20. Evidence Schema

`Evidence` binds a tool result to a claim:

```json
{
  "id": "e4",
  "claim_id": "c1",
  "kind": "runtime_probe",
  "source": "tool",
  "tool_event_id": "t7",
  "supports": "confirmed",
  "summary": "Stride-2 probe returned matches=false and max_abs_diff=0.42.",
  "data": {
    "probe": {
      "exit_code": 0,
      "timed_out": false,
      "verdict": "mismatch"
    },
    "inputs": [
      {
        "name": "x",
        "shape": [1024],
        "dtype": "float32",
        "device": "cuda",
        "stride": [2],
        "layout": "non_contiguous"
      }
    ],
    "metrics": {
      "matches": false,
      "max_abs_diff": 0.42,
      "numel": 1024,
      "rtol": 0.001,
      "atol": 0.001
    }
  },
  "artifacts": [
    {"kind": "probe_code", "path": "probes/t7_probe.py"},
    {"kind": "stdout", "path": "probes/t7_stdout.txt"}
  ],
  "created_at": "2026-07-01T15:09:04Z"
}
```

Required fields:

- `id`: evidence ID, e.g. `e4`.
- `claim_id`: claim this evidence is attached to.
- `kind`: evidence subtype.
- `source`: `"tool"` or `"agent"`; most evidence should be `"tool"`.
- `tool_event_id`: required when `source="tool"`.
- `supports`: `"confirmed"`, `"rebutted"`, or `"inconclusive"`.
- `summary`: concise human-readable explanation.
- `data`: subtype-specific structured payload.
- `artifacts`: supporting files.
- `created_at`: timestamp.

### 20.1 Evidence Kinds

Allowed `kind` values:

```text
runtime_probe       A local Python/CUDA probe was executed.
source_inspection   A tool returned a source snippet or static observation.
artifact_read       A tool read problem/test/metadata/history artifacts.
tool_error          A tool failed; usually supports inconclusive.
agent_analysis      A structured agent interpretation. Use sparingly; should not replace tool evidence.
```

### 20.2 RuntimeProbeEvidence.data

Use this for `kind="runtime_probe"`:

```json
{
  "probe": {
    "exit_code": 0,
    "timed_out": false,
    "verdict": "match | mismatch | crash | invalid | inconclusive",
    "stdout_excerpt": "...",
    "stderr_excerpt": "..."
  },
  "inputs": [],
  "outputs": {
    "candidate": [],
    "reference": []
  },
  "metrics": {
    "matches": false,
    "max_abs_diff": 0.42,
    "rel_l2": 0.13
  },
  "comparison_policy": {
    "kind": "allclose",
    "description": "Probe compared candidate and reference with torch.allclose.",
    "rtol": 0.001,
    "atol": 0.001
  }
}
```

Rules:

- `verdict="mismatch"` can support `confirmed`.
- `verdict="match"` can support `rebutted`, but only for the exact condition tested.
- `verdict="crash"` supports `confirmed` only if the input is valid for the problem.
- `verdict="invalid"` or `"inconclusive"` supports `inconclusive`.

### 20.3 SourceInspectionEvidence.data

Use this for `kind="source_inspection"`:

```json
{
  "file": "dataset/softmax/kernel.py",
  "line_start": 12,
  "line_end": 31,
  "symbols": ["BLOCK_SIZE", "tl.load", "mask"],
  "snippet": "mask = offsets < n\nx = tl.load(x_ptr + offsets, mask=mask)",
  "observation": "The load is masked by n but no stride parameter appears in the address expression."
}
```

Rules:

- Source evidence can justify a claim or support `inconclusive`.
- Source evidence alone should only `confirm` a claim when the defect is directly visible without runtime ambiguity, such as importing `torch` inside a `@triton.jit` kernel or hardcoding an output value.

### 20.4 ArtifactReadEvidence.data

Use this for `kind="artifact_read"`:

```json
{
  "file": "dataset/softmax/problem.txt",
  "line_start": 1,
  "line_end": 80,
  "summary": "Problem defines a softmax-like operator over the last dimension.",
  "relevant_fields": {
    "operator": "softmax",
    "expected_input_rank": 2
  }
}
```

### 20.5 ToolErrorEvidence.data

Use this for `kind="tool_error"`:

```json
{
  "tool_event_id": "t9",
  "error_type": "timeout",
  "phase": "candidate_kernel_invocation",
  "message": "Probe exceeded 60 seconds during candidate call.",
  "kernel_fault_possible": true,
  "retryable": true
}
```

Rules:

- Tool errors default to `supports="inconclusive"`.
- A timeout/crash during a valid candidate kernel invocation may support `confirmed`, but only if the evidence summary explicitly states why the input is valid and why the failure is attributable to the kernel rather than the probe.

## 21. Tool Schemas

Every tool needs a name, description, input schema, output schema, and side-effect policy.

### 21.1 `load_artifact`

Input:

```json
{
  "entry": "softmax",
  "dataset_dir": "dataset"
}
```

Output:

```json
{
  "entry": "softmax",
  "paths": {
    "base": "dataset/softmax",
    "problem": "dataset/softmax/problem.txt",
    "kernel": "dataset/softmax/kernel.py",
    "test": "dataset/softmax/test.py",
    "meta": "dataset/softmax/meta.json"
  },
  "available_files": ["problem.txt", "kernel.py", "test.py", "meta.json"],
  "meta": {}
}
```

Side effects: none.

### 21.2 `inspect_kernel_source`

Input:

```json
{
  "entry": "softmax",
  "start_line": 1,
  "end_line": 120
}
```

Output:

```json
{
  "file": "dataset/softmax/kernel.py",
  "line_start": 1,
  "line_end": 120,
  "content": "...",
  "num_lines": 120
}
```

Side effects: none.

### 21.3 `record_claim`

Input:

```json
{
  "statement": "For x=torch.randn(2048)[::2], the kernel ignores stride and returns wrong output.",
  "rationale": "The source address expression uses offsets only.",
  "raised_by": "skeptic",
  "tags": ["stride", "non_contiguous"]
}
```

Output:

```json
{
  "claim_id": "c1",
  "claim": {}
}
```

Side effects: appends a new claim to the ledger.

### 21.4 `read_claim_ledger`

Input:

```json
{}
```

Output:

```json
{
  "claims": [],
  "summary": {
    "open": 1,
    "confirmed": 0,
    "rebutted": 0,
    "inconclusive": 0
  }
}
```

Side effects: none.

### 21.5 `append_evidence`

Input:

```json
{
  "claim_id": "c1",
  "evidence": {}
}
```

Output:

```json
{
  "evidence_id": "e4",
  "claim_id": "c1"
}
```

Side effects: appends evidence to a claim and updates `evidence_ids`.

### 21.6 `update_claim_status`

Input:

```json
{
  "claim_id": "c1",
  "status": "confirmed",
  "reason": "Runtime probe e4 supports confirmed."
}
```

Output:

```json
{
  "claim_id": "c1",
  "old_status": "open",
  "new_status": "confirmed"
}
```

Side effects: updates claim status. The orchestrator should reject status transitions that are not supported by evidence.

### 21.7 `run_python_probe`

Input:

```json
{
  "claim_id": "c1",
  "code": "import json\n...",
  "timeout_s": 60,
  "use_gpu": true
}
```

Output:

```json
{
  "exit_code": 0,
  "timed_out": false,
  "stdout_excerpt": "...",
  "stderr_excerpt": "",
  "parsed_result": {
    "schema": "kernel_verification_probe_result.v1",
    "verdict": "mismatch",
    "inputs": [],
    "metrics": {}
  },
  "artifacts": []
}
```

Side effects: writes probe code/stdout/stderr under the run directory.

## 22. Probe Result Protocol

Agent-generated probes must print one final JSON object on the last stdout line:

```json
{
  "schema": "kernel_verification_probe_result.v1",
  "verdict": "match | mismatch | crash | invalid | inconclusive",
  "summary": "Candidate differs from reference on stride-2 input.",
  "inputs": [],
  "outputs": {
    "candidate": [],
    "reference": []
  },
  "metrics": {},
  "comparison_policy": {
    "kind": "allclose",
    "rtol": 0.001,
    "atol": 0.001,
    "description": "torch.allclose(candidate, reference)"
  }
}
```

Rules:

- Free-form logs are allowed before the final JSON line.
- The final line must be parseable JSON.
- The probe should catch exceptions and encode them as `verdict="crash"` or `verdict="invalid"` depending on whether the input is valid.
- The probe should never decide final claim status directly. It reports a probe verdict; the experimenter/orchestrator maps that to evidence.

## 23. Claim Status Transition Rules

Allowed transitions:

```text
open -> confirmed
open -> rebutted
open -> inconclusive
confirmed -> inconclusive      only if later evidence invalidates the confirming probe
rebutted -> inconclusive       only if later evidence invalidates the rebuttal probe
inconclusive -> confirmed      if stronger evidence is later produced
inconclusive -> rebutted       if stronger evidence is later produced
```

Avoid direct:

```text
confirmed -> rebutted
rebutted -> confirmed
```

Instead create a new refined claim or move through `inconclusive` with an explanation.

Evidence requirements:

- `confirmed` requires at least one evidence with `supports="confirmed"`.
- `rebutted` requires at least one evidence with `supports="rebutted"` for the exact condition in the claim.
- `inconclusive` requires evidence explaining why the system cannot decide yet.
- A `rationale` alone can never move a claim out of `open`.
- A `tool_error` alone usually maps to `inconclusive`, not `confirmed`.

## 24. Judge Input Contract

The judge receives:

```json
{
  "artifact_summary": {},
  "claims": [],
  "tool_events_index": {},
  "unresolved_tool_errors": [],
  "skills": []
}
```

The judge should cite claim IDs, evidence IDs, and tool event IDs when relevant.

Judge output:

```json
{
  "verdict": "trust | reject | needs_more_evidence",
  "confidence": 0.0,
  "decisive_claims": ["c1"],
  "decisive_evidence": ["e4"],
  "reason": "Claim c1 is confirmed by runtime probe evidence e4 from tool event t7."
}
```

Rules:

- `trust` requires no confirmed genuine defects and no important unresolved open claims.
- `reject` requires at least one confirmed genuine defect, or directly visible source-level invalidity.
- `needs_more_evidence` is the correct verdict when important claims remain open/inconclusive.
- The judge should not turn rationale into evidence.
