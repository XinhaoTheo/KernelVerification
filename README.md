# Kernel Verification via Multi-Agent Debate

This system **decides whether to trust** a Triton kernel written by an LLM. A kernel is a
handwritten GPU routine, and Triton is the language used to write it. One agent is not enough to
decide whether a kernel is correct. The system gives several LLM agents different roles. One
describes the implementation, one finds possible bugs, one runs local experiments, and one judges
the collected evidence. The agents discuss the kernel and work toward an agreement. This follows
the way a careful human reviewer works: read the code, identify possible problems, run focused
experiments, and then decide whether the kernel should be trusted.

The current design keeps that debate evidence-driven:

```text
Agents decide what to investigate.
Skills set rules for the investigation.
Tools provide local actions that can be executed.
Runtime executes tools locally and records evidence.
```

The verifier does not ask the LLM to directly run Python, Triton, CUDA, or filesystem operations.
Agents send structured tool calls. The local runtime executes them and records the results in a
claim ledger, tool-event ledger, transcript, and final verdict.

---

## 1. Why this exists

Tools like Meta's KernelAgent can generate Triton kernels automatically, but the generated kernel usually comes with only a narrow test. Often the same LLM writes the kernel, writes the test, and only checks one or two simple input shapes. A kernel can pass that test and still be
wrong in ways that matter:

- **A single allclose test is not a reliable correctness test.** For matmul-like kernels, output error often tracks real error. For softmax, activations, top-k, and low-bit kernels, the same numeric tolerance can falsely accept bad kernels or falsely reject good ones.

- **Softmax and activations can hide errors.** Tail logits, negative ReLU inputs, and saturated GELU/SiLU regions can be computed incorrectly while the final output barely changes, so standard random tests may pass.

- **Selection operators need different metrics.** `topk`, `sort`, and sparse-attention selection are about which values are selected. Index recall, value error, and downstream output can disagree, especially around ties or cutoff boundaries.

- **Low-bit kernels break fixed tolerances.** FP8/FP4/INT4 can round meaningful differences to the same code, while correct low-bit outputs may still be far from FP32 references. One tolerance cannot separate expected low-precision error from real bugs.

This project is the independent reviewer. It takes a saved kernel artifact, lets agents reason about where it might fail, runs local experiments through controlled tools, and returns a trust decision backed by a transcript, claim ledger, tool events, probe outputs, and a final verdict.

The key difference from a normal test harness is that the verifier is **evidence-driven, not checklist-driven**. The verifier must inspect the operator, choose useful stress cases, pick the right metric, decide what is in scope, run targeted probes, and judge the evidence. Those choices depend on the kernel. Agents make them as the run proceeds, skills set the rules, and local tools perform the work.

---

## 2. Overall Pipeline

The project has one **offline** step to create or collect kernel artifacts, and one **online**
agentic verification step that runs each time we want a trust decision.

The diagram below shows how those two steps connect to the verifier's runtime design.
Skills guide how the agents investigate, agents send structured tool calls, and the
Orchestrator owns the shared state while running tools and moving the debate forward.

<p align="center">
  <img src="assets/readme/overall_pipeline.drawio.svg" alt="Overall multi-agent kernel verification pipeline" width="100%">
</p>

<p align="center"><em>Figure 1. End-to-end kernel verification pipeline and agent runtime protocol.</em></p>

- **Step 1 — Build or load the dataset (`kv-build`)**: run KernelAgent on a KernelBench
  problem and save the resulting artifact under `dataset/<entry>/`. This step is only about
  producing a self-contained folder with `problem.txt`, `kernel.py`, `test.py`, seeds, and
  metadata. The verifier does not treat the generator's test as the final authority.
- **Step 2 — Agentic verification (`kv-agentic-run`)**: load the artifact, let several LLM
  agents discuss it, allow only structured tool calls for local execution, and record every
  important conclusion as a claim with evidence.

### Step 1 — Build the dataset

A KernelBench problem is a PyTorch reference task. KernelAgent turns it into a Triton kernel
attempt and a test. `kv-build` stores the result as a self-contained dataset entry:

| File | Contents |
|---|---|
| `problem.txt` | the original problem or benchmark contract |
| `kernel.py` | the final generated kernel, or the closest failed attempt |
| `test.py` | KernelAgent's generated test, kept as benchmark/context evidence |
| `seed_*.py` | first-draft kernels, if available |
| `meta.json` | status, pass/fail metadata, generation rounds, and related fields |
| `error.txt` | error output when generation failed |

The important property is that `dataset/<entry>/` is self-contained. The verifier can run from this
folder without depending on the original KernelAgent working directory.

### Step 2 — Agentic verification

`kv-agentic-run` starts by loading the artifact and then runs agents through the JSON tool-call
protocol.

#### 2.1 Describer

The Describer explains what the kernel appears to implement. It may inspect `problem.txt`,
`kernel.py`, or other artifact files, but it does not record claims or make a verdict. The
goal is to give the later agents a clear implementation summary without turning that summary
into evidence by itself.

#### 2.2 Skeptic

The Skeptic raises concrete, testable claims about possible bugs. A good claim names:

- the condition being tested.
- the possible wrong behavior.
- why the source or benchmark context makes the claim likely.
- whether the case is `in_scope`, `out_of_scope`, or `unknown`.

The Skeptic is limited to a small number of claims per turn. This is intentional. We want the
highest-risk claims first, not a long list of guesses.

#### 2.3 Experimenter

The Experimenter is the only role that should run probes. It does not directly execute Python
or CUDA by itself. It calls local tools such as `run_claim_probe`, then reads the returned
tool event with `finalize_probe_evidence` or the lower-level evidence tools. Probe code runs
locally in the runtime, not inside the LLM.

For each important claim, the Experimenter should either confirm it, rebut it, or mark it
inconclusive with evidence.

#### 2.4 Skeptic review and Judge

After evidence is added, the Skeptic must review the latest ledger. If there are no more
high-quality in-scope claims, it records `record_no_new_claims`.

The Judge then reads the run state, claims, evidence, tool events, and skeptic review. It can
either:

- call `record_verdict` with `trust`, `reject`, or `needs_more_evidence`.
- call `request_more_debate` when more investigation is needed and debate budget remains.

The Judge participates at the end of each debate round once claim coverage and skeptic review
requirements are satisfied.

---

## 3. Component details

### 3.1 `verifier/build_dataset.py` and `verifier/dataset.py` — dataset artifacts

`kv-build` drives KernelAgent and saves the output through `verifier/dataset.py`. The dataset
layer only loads and saves artifact files. It does not decide whether a kernel is correct.

`test.py` is still useful, but its meaning has changed. It is no longer "the test we blindly
trust." It is evidence about the benchmark domain. For example, if `test.py` fixes
`features = 64`, then `features = 0` should not become an in-scope reason to reject unless
another benchmark artifact explicitly requires that case.

### 3.2 `verifier/agentic/orchestrator.py` — the workflow driver

The orchestrator owns the verification loop. It:

- initializes `RunState`.
- loads the artifact through tools.
- calls LLM agents.
- exposes the tool schemas.
- validates and executes tool calls.
- records every turn and every tool result.
- enforces loop budgets and convergence rules.

The Orchestrator therefore controls the workflow and owns the run state. The
figure below expands the evidence path: an open claim leads to a targeted probe, the runtime
records the raw tool result, and the Experimenter interprets that result before attaching it as
evidence and updating the claim status. A raw observation is not treated as evidence until it is
linked to a concrete claim.

<p align="center">
  <img src="assets/readme/Orchestrator.drawio.svg" alt="Orchestrator shared state and evidence loop" width="100%">
</p>

<p align="center"><em>Figure 2. Orchestrator responsibilities, shared run state, and the evidence loop.</em></p>

There are two loops:

- the **debate round loop**, where Describer and Skeptic can add or refine claims.
- the **claim coverage loop**, where Experimenter must gather evidence for open claims before
  Judge can finalize.

The default full workflow is:

```text
describer
skeptic
experimenter until open claims are covered
skeptic review
judge
repeat if judge requests more debate
```

### 3.3 `verifier/agentic/protocol.py` — JSON tool-call protocol

Agents must return exactly one JSON object:

```json
{
  "message": "I will inspect the kernel source and record one concrete claim.",
  "tool_calls": [
    {
      "tool": "inspect_kernel_source",
      "args": {"entry": "elem_add", "start_line": 1, "end_line": 120}
    }
  ]
}
```

The LLM is responsible for deciding what to investigate. The local runtime is responsible for
reading files, running probes, updating ledgers, and writing artifacts. This separates reasoning
from execution.

### 3.4 `verifier/agentic/agents/` — the four roles

Current roles:

| Agent | Responsibility |
|---|---|
| `describer.py` | explain implementation and visible assumptions |
| `skeptic.py` | raise concrete bug claims and later review whether new claims remain |
| `experimenter.py` | run local probes and attach evidence to claims |
| `judge.py` | decide final verdict or request more debate |

Each role loads the skill files that match its work. Skills guide agent behavior, while tools
perform the actual local actions.

### 3.5 `verifier/agentic/skills/` — verification guidance

Skills are markdown instructions added to agent prompts. They are not executable code. They
help agents follow the same verification rules while still choosing what to investigate for each
kernel.

| Skill | Read by | Used during |
|---|---|---|
| `kernel-verification.md` | Describer, Skeptic, Experimenter, Judge | the full evidence-driven workflow |
| `evidence-driven-review.md` | Describer, Skeptic, Experimenter, Judge | source review, evidence review, and the final decision |
| `claim-lifecycle.md` | Skeptic, Experimenter, Judge | claim creation, evidence updates, and claim review |
| `experiment-design.md` | Experimenter | probe design and result collection |
| `adversarial-precision.md` | Describer, Skeptic, Experimenter, Judge | kernels with softmax, activations, selection, or low precision |
| `metric-selection.md` | Skeptic, Experimenter, Judge | choosing probe metrics and judging results |
| `scope-policy.md` | Skeptic, Experimenter, Judge | deciding whether a claim belongs to the benchmark input domain |
| `convergence.md` | Judge | deciding whether to finish or request another debate round |

Skills provide guidance, not a fixed checklist. Agents decide which guidance matters for the
current kernel, and the Orchestrator keeps the workflow within its budgets and state rules.

### 3.6 `verifier/agentic/tools/` — local capabilities

Tools are small and focused. They provide actions, not fixed verification policy.

| Tool | Purpose |
|---|---|
| `load_artifact` | load one dataset entry into run state |
| `inspect_problem` | read `problem.txt` |
| `inspect_kernel_source` | read numbered slices of `kernel.py` |
| `list_artifact_files` | list files inside the artifact |
| `read_artifact_file` | read a controlled file inside the artifact |
| `record_claim` | add one concrete claim to the ledger |
| `read_claim_ledger` | read current claims and evidence |
| `record_no_new_claims` | let Skeptic state that latest evidence was reviewed |
| `append_evidence` | attach evidence manually to a claim |
| `update_claim_status` | set `open`, `confirmed`, `rebutted`, or `inconclusive` |
| `run_python_probe` | run exploratory or debugging Python that is not tied to a claim |
| `run_claim_probe` | run Python for an existing claim and return an evidence draft linked to it |
| `finalize_probe_evidence` | consume a `run_claim_probe` result and update the claim |
| `retrieve_experiment_history` | read prior probe history |
| `request_more_debate` | let Judge ask for another round |
| `record_verdict` | write the final verdict |

The tool registry validates required arguments and records both successful and failed tool
events. Tool failures are part of the transcript and can themselves become evidence.

`run_python_probe` and `run_claim_probe` use the same local Python runner. The first is for free
exploration and debugging. The second requires a `claim_id` and returns an evidence draft linked
to that claim. A later `finalize_probe_evidence` call attaches the result and updates the claim
status.

### 3.7 `verifier/agentic/ledger.py` and `state.py` — claims and evidence

The claim ledger is the center of the system. A claim is not just text. It has a status,
scope, rationale, and evidence list.

Claim statuses:

| Status | Meaning |
|---|---|
| `open` | proposed but not yet decided |
| `confirmed` | evidence supports the bug claim |
| `rebutted` | evidence argues against the exact claim |
| `inconclusive` | evidence was collected but does not decide it |

Claim scopes:

| Scope | Meaning |
|---|---|
| `in_scope` | required by the benchmark/test input contract |
| `out_of_scope` | useful note about unsupported inputs, but not a benchmark correctness failure |
| `unknown` | not enough contract evidence to make a decision |

Evidence records where a conclusion came from. It may point to source inspection, a runtime
probe, a tool error, or agent analysis. The important rule is that final claims should be
backed by something recorded in the ledger or tool events.

### 3.8 `verifier/agentic/tools/execution.py` — local probes

Probe tools write the generated Python to `probes/<tool_event_id>_probe.py`, run it locally,
capture stdout/stderr, parse the last stdout line as JSON when possible, and save the result
under the run directory.

The LLM can decide what code to run, but it cannot directly access the filesystem or GPU
outside the tool interface. This makes every probe easy to review. Each probe has source
code, captured output, and a corresponding `ToolEvent`.

### 3.9 `verifier/agentic/persistence.py` — run records and transcript

Each run is saved to disk. The most useful human file is `transcript.md`, which shows the
timeline, agent messages, tool calls, output summaries, claims, evidence, and verdict.

The structured files are for replay and automated analysis. The transcript is for debugging
why a verdict happened.

### 3.10 `verifier/agentic/llm.py` — provider adapter

The agentic verifier can call Anthropic or OpenAI/ChatGPT providers. The CLI accepts
`--provider anthropic`, `--provider openai`, or `--provider chatgpt`. The default comes from
`.env`.

---

## 4. The signals and the final verdict

The current system produces one main signal: the **claim ledger**. Everything else feeds into
that ledger.

| Signal | Comes from | Means |
|---|---|---|
| `history` | agent turns | what each agent said and requested |
| `tool_events` | local tool calls | what was actually read, executed, or changed |
| `claims` | Skeptic + ledger tools | the possible bugs under investigation |
| `evidence` | Experimenter + tools | source/probe support for each claim |
| `skeptic_review` | `record_no_new_claims` | Skeptic reviewed latest evidence and found no new high-quality claims |
| `verdict` | Judge | final trust decision |

Verdict values:

| Verdict | Meaning |
|---|---|
| `trust` | no confirmed in-scope correctness failure remains |
| `reject` | at least one confirmed in-scope correctness failure is strong enough to reject the kernel |
| `needs_more_evidence` | important in-scope claims remain unresolved |

`record_verdict` has a fixed check for `reject`. A reject verdict can only use
claims that are `confirmed`, `in_scope`, and backed by scope evidence from the benchmark/test
input domain. A claim about a useful test case outside the contract can be recorded, tested,
and discussed, but it should not reject a benchmark kernel by itself.

This scope rule is simple by design. The system does not try to build a smart static
contract parser. The LLM reads the artifact and explains scope. The tool layer only prevents
the most obvious unsafe reject.

---

## 5. Repository layout

```text
kernel_verification/
├── KernelAgent/              # kernel generator dependency
├── KernelBench/              # benchmark/problem source dependency
├── dataset/                  # saved kernel artifacts
│   └── <entry>/
│       ├── problem.txt
│       ├── kernel.py
│       ├── test.py
│       ├── meta.json
│       └── seed_*.py
├── verifier/
│   ├── build_dataset.py      # kv-build
│   ├── dataset.py            # artifact load/save helpers
│   ├── gpu_pick.py           # optional GPU selection helper
│   ├── agentic_run.py        # kv-agentic-run CLI
│   └── agentic/
│       ├── orchestrator.py   # agent loop, claim coverage, convergence
│       ├── protocol.py       # JSON response parsing
│       ├── state.py          # RunState, Turn, ToolEvent, Claim, Evidence
│       ├── ledger.py         # claim/evidence mutation rules
│       ├── persistence.py    # run.json, transcript.md, replay loading
│       ├── llm.py            # Anthropic/OpenAI provider adapter
│       ├── agents/
│       │   ├── describer.py
│       │   ├── skeptic.py
│       │   ├── experimenter.py
│       │   └── judge.py
│       ├── tools/
│       │   ├── artifacts.py
│       │   ├── claims.py
│       │   ├── execution.py
│       │   ├── history.py
│       │   ├── registry.py
│       │   └── verdict.py
│       └── skills/
│           ├── kernel-verification.md
│           ├── evidence-driven-review.md
│           ├── claim-lifecycle.md
│           ├── experiment-design.md
│           ├── adversarial-precision.md
│           ├── metric-selection.md
│           ├── scope-policy.md
│           └── convergence.md
├── tests/                    # unit tests for protocol, tools, workflow, persistence
├── agentic_roadmap.md        # design roadmap for the refactor
├── agentic_optimization_plan.md
├── README-original.md        # backup of the previous README before this rewrite
├── README.md
└── pyproject.toml
```

The old fixed verification modules and root-level debate agents are no longer the primary
system. New verification should use `verifier/agentic/` and `kv-agentic-run`.

---

## 6. CLI reference

### 6.1 Setup

Copy `.env.example` to `.env` and set only the keys you need:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

AGENTIC_PROVIDER=openai        # anthropic | openai | chatgpt
AGENTIC_MODEL=gpt-5            # or another model supported by the provider
AGENTIC_MAX_ROUNDS=4
```

KernelAgent generation can use its own settings in `.env`, but the verifier does not need
KernelAgent settings when it is only verifying existing `dataset/` entries.

### 6.2 Build dataset entries

```bash
uv run kv-build --list
uv run kv-build --problem elem_add
```

This creates or refreshes `dataset/<entry>/`.

### 6.3 Dry-run the tool protocol

```bash
uv run kv-agentic-run elem_add --dry-run
uv run kv-agentic-run --all --dry-run
```

Dry-run does not call an LLM. It is useful for checking artifact loading and saved run files.

### 6.4 Run one real agent

```bash
uv run kv-agentic-run elem_add \
  --provider openai \
  --model gpt-5 \
  --agent skeptic \
  --max-debate-rounds 1
```

This is mostly useful for prompt or protocol debugging.

### 6.5 Run the full agentic verifier

```bash
uv run kv-agentic-run elem_add \
  --provider openai \
  --model gpt-5 \
  --agents describer,skeptic,experimenter,judge \
  --max-debate-rounds 5 \
  --min-debate-rounds-before-judge 2 \
  --max-claim-rounds 3 \
  --max-claim-rounds-per-claim 3 \
  --tool-budget 150 \
  --max-tokens 4096
```

This is the normal verification command. It gives the Skeptic at least one chance to review
evidence before Judge finalizes.

### 6.6 Run every dataset entry

```bash
uv run kv-agentic-run --all \
  --dataset-dir dataset \
  --run-dir /tmp/kv-agentic-full-run \
  --provider openai \
  --model gpt-5 \
  --agents describer,skeptic,experimenter,judge \
  --max-debate-rounds 5 \
  --min-debate-rounds-before-judge 2 \
  --max-claim-rounds 3 \
  --max-claim-rounds-per-claim 3 \
  --tool-budget 150 \
  --max-tokens 4096
```

When `--all` is used with `--run-dir`, each entry gets a subdirectory:

```text
/tmp/kv-agentic-full-run/<entry>/
```

### 6.7 Continue from a saved run

```bash
uv run kv-agentic-run elem_add \
  --replay-run /tmp/kv-agentic-full-run/elem_add/run.json \
  --provider openai \
  --agent judge
```

Replay loads the saved `RunState` and continues from it.

### 6.8 Useful loop controls

| Option | Meaning |
|---|---|
| `--max-debate-rounds` | maximum outer debate rounds |
| `--min-debate-rounds-before-judge` | force at least this many debate rounds before Judge |
| `--max-claim-rounds` | baseline Experimenter rounds allowed for claim coverage |
| `--max-claim-rounds-per-claim` | dynamic Experimenter budget per uncovered claim |
| `--tool-budget` | stop after this many tool calls |
| `--stop-when-no-open-claims` | stop early when all claims are resolved |
| `--no-require-claim-coverage` | allow Judge even when open claims lack evidence |

---

### 6.9 Run artifacts and debugging

Each run directory contains:

```text
run.json
claims.json
tool_events.jsonl
transcript.md
verdict.json        # only when Judge records a verdict
probes/             # only when probe tools are used
```

| File | Who uses it | Purpose |
|---|---|---|
| `transcript.md` | humans | easiest file to read when debugging, with the timeline, messages, tools, claims, and verdict |
| `run.json` | humans + code | complete structured state for replay and automated analysis |
| `claims.json` | humans + code | claim ledger only |
| `tool_events.jsonl` | humans + code | one tool execution per line, useful for batch analysis |
| `verdict.json` | humans + code | final Judge verdict |
| `probes/*_probe.py` | humans | exact Python probe code generated by the agent |
| `probes/*_stdout.txt` | humans | captured stdout from the probe |
| `probes/*_stderr.txt` | humans | captured stderr from the probe |
| `probes/*_json_result.json` | humans + code | parsed JSON from the last stdout line, when available |

The best debugging path is usually:

1. Open `transcript.md`.
2. Find the final `record_verdict` tool event.
3. Check the claims used for the verdict in the `Claims` section.
4. Follow each claim's evidence to the matching `run_claim_probe` or source inspection
   tool event.
5. Open the probe files under `probes/` if the evidence summary is not enough.

---

## 7. Known limitations

- **The verifier still depends on LLM judgment.** The system records more evidence now, but
  the choice of possible bugs and interpretation of scope still come from agents. This is by
  design. The project is an agentic verifier, not a fixed static analyzer.

- **Scope handling is careful but not fully formal.** The system separates benchmark failures
  from failures on unsupported inputs, but it does not parse every possible input contract
  into a formal spec. A reject verdict is guarded against obvious out-of-scope claims, but
  harder scope cases still depend on the agents reading `problem.txt`, `test.py`, and metadata.

- **Probe quality varies.** Experimenter-generated probes are saved and easy to review, but the
  first probe may not be the best probe. The workflow allows more debate rounds so agents can
  review and improve the evidence.

- **Runtime cost can be high.** Full dataset verification calls LLM APIs many times and may
  run GPU probes. Use `--tool-budget`, smaller debate budgets, or single-entry runs while
  debugging.

- **`decisive_claims` is currently shared across verdict types.** For `reject`, it means the
  confirmed in-scope claims that caused rejection. For `trust`, it often means the claims whose
  evidence ruled them out. That works today, but the naming could be clearer in a later
  schema revision.
