# Kernel Verification via Agentic Debate

A system that **decides whether to trust** a Triton kernel written by an LLM. A kernel is a
hand-written GPU routine; Triton is the language used to write it. The hard part is not only
running a test. The hard part is deciding **what should be tested**, whether a suspicious case
is actually inside the benchmark contract, and whether the evidence is strong enough to reject
the kernel.

Earlier versions of this project tried to solve that with a fixed verification pipeline:
standard recheck, robustness batteries, operator classifiers, precision rechecks, and a
predefined checklist of cases. That direction was too rigid. It made the program decide what
to investigate before seeing the kernel's actual implementation. The current system is built
around a different rule:

```text
Agents decide what to investigate.
Skills constrain how investigation should be done.
Tools provide executable local capabilities.
Runtime executes tools locally and records evidence.
```

The verifier is now an **agentic kernel verification system**. LLM agents discuss the kernel,
raise concrete bug hypotheses, call local tools to inspect files or run probes, update a claim
ledger, and let a judge give a final verdict based on accumulated evidence.

---

## 1. Why this exists

Tools like Meta's KernelAgent can generate Triton kernels automatically, but the generated
kernel usually ships with a narrow test. Often the same LLM writes the kernel, writes the test,
and only checks one or two friendly input shapes. A kernel can pass that test and still be
wrong in ways that matter:

- **Read the wrong memory** because it assumes contiguous layout or ignores tensor strides.
- **Drop part of an input** because the implementation silently caps a block size or row size.
- **Use the wrong numerical path** and only look correct under loose bf16/fp16 tolerances.
- **Handle the benchmark case but fail an important nearby case**, where the hard question is
  whether that nearby case is actually in scope.
- **Cheat or overfit** by hard-coding the shape, dtype, or input pattern used by the original
  test.

This project is the independent reviewer. It takes a saved kernel artifact, lets agents reason
about where it might fail, runs local experiments through controlled tools, and returns a
trust decision backed by a transcript, claim ledger, tool events, probe outputs, and a final
verdict.

The key difference from a normal test harness is that the verifier is **evidence-driven, not
checklist-driven**. It does not assume every kernel should be tested with the same fixed
battery. The Skeptic proposes concrete hypotheses, the Experimenter runs targeted probes, and
the Judge decides only from recorded evidence.

---

## 2. Overall Pipeline

The project has one **offline** step to create or collect kernel artifacts, and one **online**
agentic verification step that runs each time we want a trust decision.

- **Step 1 — Build or load the dataset (`kv-build`)**: run KernelAgent on a KernelBench
  problem and save the resulting artifact under `dataset/<entry>/`. This step is only about
  producing a self-contained folder with `problem.txt`, `kernel.py`, `test.py`, seeds, and
  metadata. The verifier does not treat the generator's test as the final authority.
- **Step 2 — Agentic verification (`kv-agentic-run`)**: load the artifact, let several LLM
  agents discuss it, allow only structured tool calls for local execution, and record every
  important conclusion as a claim with evidence.

The online workflow looks like this:

```text
dataset/<entry>/
  problem.txt
  kernel.py
  test.py
  meta.json
  seed_*.py
        |
        v
AgenticOrchestrator
        |
        v
Describer -> Skeptic -> Experimenter -> Skeptic review -> Judge
        |
        v
Tool registry -> local tools -> ToolEvent ledger
        |
        v
Claim ledger + transcript + verdict.json
```

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

The important property is that `dataset/<entry>/` is portable. The verifier can run from this
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

The Skeptic raises concrete, testable bug hypotheses. A good claim names:

- the condition being tested;
- the possible wrong behavior;
- why the source or benchmark context makes the hypothesis plausible;
- whether the case is `in_scope`, `out_of_scope`, or `unknown`.

The Skeptic is limited to a small number of claims per turn. This is intentional. We want the
highest-risk hypotheses first, not a long speculative checklist.

#### 2.3 Experimenter

The Experimenter is the only role that should run probes. It does not directly execute Python
or CUDA by itself. It calls local tools such as `run_claim_probe`, then consumes the returned
tool event with `finalize_probe_evidence` or the lower-level evidence tools. Probe code runs
locally in the runtime, not inside the LLM.

For each important claim, the Experimenter should either confirm it, rebut it, or mark it
inconclusive with evidence.

#### 2.4 Skeptic review and Judge

After evidence is added, the Skeptic must review the latest ledger. If there are no more
high-quality in-scope claims, it records `record_no_new_claims`.

The Judge then reads the run state, claims, evidence, tool events, and skeptic review. It can
either:

- call `record_verdict` with `trust`, `reject`, or `needs_more_evidence`; or
- call `request_more_debate` when more investigation is needed and debate budget remains.

The Judge participates at the end of each debate round once claim coverage and skeptic review
requirements are satisfied.

---

## 3. Component deep dive

### 3.1 `verifier/build_dataset.py` and `verifier/dataset.py` — dataset artifacts

`kv-build` drives KernelAgent and saves the output through `verifier/dataset.py`. The dataset
layer deliberately stays simple: it loads and saves artifact files. It does not decide whether
a kernel is correct.

`test.py` is still useful, but its meaning has changed. It is no longer "the test we blindly
trust." It is evidence about the benchmark domain. For example, if `test.py` fixes
`features = 64`, then `features = 0` should not become a decisive in-scope rejection unless
another benchmark artifact explicitly requires that case.

### 3.2 `verifier/agentic/orchestrator.py` — the workflow driver

The orchestrator owns the verification loop. It:

- initializes `RunState`;
- loads the artifact through tools;
- calls LLM agents;
- exposes the tool schemas;
- validates and executes tool calls;
- records every turn and every tool result;
- enforces loop budgets and convergence rules.

There are two loops:

- the **debate round loop**, where Describer and Skeptic can add or refine hypotheses;
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
actually reading files, running probes, updating ledgers, and writing artifacts. This is the
main separation between reasoning and execution.

### 3.4 `verifier/agentic/agents/` — the four roles

Current roles:

| Agent | Responsibility |
|---|---|
| `describer.py` | explain implementation and visible assumptions |
| `skeptic.py` | raise concrete bug claims and later review whether new claims remain |
| `experimenter.py` | run local probes and attach evidence to claims |
| `judge.py` | decide final verdict or request more debate |

The prompts include skill files from `verifier/agentic/skills/`. These skills are markdown
workflow instructions. They are not executable code. They tell the agents how to behave:
prefer evidence, avoid vague claims, respect scope, design probes carefully, and converge only
after the latest evidence has been reviewed.

### 3.5 `verifier/agentic/tools/` — local capabilities

Tools are deliberately low-level. They provide capabilities, not fixed verification policy.

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
| `run_python_probe` | run agent-written Python locally and capture outputs |
| `run_claim_probe` | run a probe tied to one claim and return an evidence draft |
| `finalize_probe_evidence` | consume a `run_claim_probe` result and update the claim |
| `retrieve_experiment_history` | read prior probe history |
| `request_more_debate` | let Judge ask for another round |
| `record_verdict` | write the final verdict |

The tool registry validates required arguments and records both successful and failed tool
events. Tool failures are part of the transcript and can themselves become evidence.

### 3.6 `verifier/agentic/ledger.py` and `state.py` — claims and evidence

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
| `out_of_scope` | useful generalization note, but not a benchmark correctness failure |
| `unknown` | not enough contract evidence to make it decisive |

Evidence records where a conclusion came from. It may point to source inspection, a runtime
probe, a tool error, or agent analysis. The important rule is that final claims should be
backed by something recorded in the ledger or tool events.

### 3.7 `verifier/agentic/tools/execution.py` — local probes

Probe tools write the generated Python to `probes/<tool_event_id>_probe.py`, run it locally,
capture stdout/stderr, parse the last stdout line as JSON when possible, and save the result
under the run directory.

The LLM can decide what code to run, but it cannot directly access the filesystem or GPU
outside the tool interface. This keeps the agentic layer auditable: every probe has source
code, captured output, and a corresponding `ToolEvent`.

### 3.8 `verifier/agentic/persistence.py` — run records and transcript

Each run is persisted to disk. The most useful human file is `transcript.md`, which shows the
timeline, agent messages, tool calls, output summaries, claims, evidence, and verdict.

The structured files are for replay and programmatic analysis. The transcript is for debugging
why a verdict happened.

### 3.9 `verifier/agentic/llm.py` — provider adapter

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
| `tool_events` | local tool calls | what was actually read, executed, or mutated |
| `claims` | Skeptic + ledger tools | the hypotheses under investigation |
| `evidence` | Experimenter + tools | source/probe support for each claim |
| `skeptic_review` | `record_no_new_claims` | Skeptic reviewed latest evidence and found no new high-quality claims |
| `verdict` | Judge | final trust decision |

Verdict values:

| Verdict | Meaning |
|---|---|
| `trust` | no confirmed in-scope correctness failure remains |
| `reject` | at least one confirmed in-scope correctness failure is decisive |
| `needs_more_evidence` | important in-scope claims remain unresolved |

`record_verdict` has a deterministic guard for reject: a reject verdict can only use decisive
claims that are `confirmed`, `in_scope`, and backed by scope evidence from the benchmark/test
input domain. A claim about a useful but out-of-contract stress case can be recorded, tested,
and discussed, but it should not reject a benchmark kernel by itself.

This scope rule is intentionally simple. The system does not try to build a smart static
contract parser. The LLM reads the artifact and explains scope; the tool layer only prevents
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
system. New verification should route through `verifier/agentic/` and `kv-agentic-run`.

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

Dry-run does not call an LLM. It is useful for checking artifact loading and persistence.

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
| `transcript.md` | humans | easiest file to read when debugging; shows timeline, messages, tools, claims, verdict |
| `run.json` | humans + code | complete structured state for replay and programmatic analysis |
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
3. Check the decisive claims in the `Claims` section.
4. Follow each claim's evidence to the corresponding `run_claim_probe` or source inspection
   tool event.
5. Open the probe files under `probes/` if the evidence summary is not enough.

---

## 7. Known limitations

- **The verifier still depends on LLM judgment.** The system records more evidence now, but
  the choice of hypotheses and interpretation of scope still come from agents. This is by
  design; the project is an agentic verifier, not a fixed static analyzer.

- **Scope is conservative but not fully formal.** The system distinguishes benchmark-domain
  failures from generalization failures, but it does not parse every possible input contract
  into a formal spec. A reject verdict is guarded against obvious out-of-scope claims, but
  nuanced scope still depends on the agents reading `problem.txt`, `test.py`, and metadata.

- **Probe quality varies.** Experimenter-generated probes are saved and auditable, but the
  first probe may not be the best probe. The workflow allows more debate rounds so agents can
  critique and improve the evidence.

- **Runtime cost can be high.** Full dataset verification calls LLM APIs many times and may
  run GPU probes. Use `--tool-budget`, smaller debate budgets, or single-entry runs while
  debugging.

- **`decisive_claims` is currently shared across verdict types.** For `reject`, it means the
  confirmed in-scope claims that caused rejection. For `trust`, it often means the claims whose
  rebuttal was decisive. That is usable today, but the naming could be clearer in a later
  schema revision.

- **This README intentionally has no diagrams.** The old diagrams described the removed fixed
  pipeline and should not be treated as documentation for the current agentic system.
