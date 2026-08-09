# KernelVerification

KernelVerification is being refactored into an **agentic kernel verification system**.
The verifier no longer runs a hard-coded sequence such as standard recheck, precision
recheck, robustness battery, or operator-class routing. Instead, LLM agents decide what
to investigate, while local tools execute controlled actions and record evidence.

Core principle:

```text
Agents decide what to investigate.
Skills constrain how investigation should be done.
Tools provide executable local capabilities.
Runtime executes tools and returns evidence.
```

## Current Architecture

```text
dataset/<entry>/
  problem.txt
  kernel.py
  test.py
  meta.json
        |
        v
AgenticOrchestrator
        |
        v
LLM agents: describer -> skeptic -> experimenter -> judge
        |
        v
Tool registry -> local tools -> ToolEvent ledger
        |
        v
Claim ledger + evidence + verdict.json
```

The important modules are under `verifier/agentic/`:

| Path | Purpose |
|---|---|
| `state.py` | Structured run state: turns, tool events, claims, evidence, verdict |
| `ledger.py` | Claim and evidence mutation rules |
| `protocol.py` | Provider-independent JSON tool-call protocol |
| `orchestrator.py` | Agent loop, tool dispatch, budgets, convergence stops |
| `persistence.py` | `run.json`, `tool_events.jsonl`, `claims.json`, `verdict.json`, replay loading |
| `llm.py` | LLM client adapter, currently Anthropic |
| `agents/` | Describer, Skeptic, Experimenter, Judge |
| `tools/` | Artifact/source/claim/execution/history/verdict tools |
| `skills/` | Markdown workflow instructions for evidence-driven verification |

## CLI

Build dataset entries from KernelAgent:

```bash
uv run kv-build --list
uv run kv-build --problem elem_add
```

Run the agentic verifier without an LLM call, useful for plumbing checks:

```bash
uv run kv-agentic-run elem_add --dry-run
```

Run one real LLM agent:

```bash
uv run kv-agentic-run elem_add --agent skeptic --max-debate-rounds 1
```

Run with OpenAI/ChatGPT provider:

```bash
uv run kv-agentic-run elem_add --provider openai --model gpt-5 --agent skeptic --max-debate-rounds 1
```

Run the full current agent chain:

```bash
uv run kv-agentic-run elem_add --agents describer,skeptic,experimenter,judge --max-debate-rounds 1
```

Run every dataset entry. This may use significant API/GPU budget when not using `--dry-run`:

```bash
uv run kv-agentic-run --all --dry-run
uv run kv-agentic-run --all --provider openai --model gpt-5 --agents describer,skeptic,experimenter,judge --max-debate-rounds 1
```

Continue from a saved run:

```bash
uv run kv-agentic-run elem_add --replay-run dataset/elem_add/agentic_runs/describer+skeptic+experimenter+judge/run.json --agent judge
```

Useful loop controls:

```bash
uv run kv-agentic-run elem_add \
  --agents describer,skeptic,experimenter,judge \
  --max-debate-rounds 3 \
  --tool-budget 20 \
  --stop-when-no-open-claims
```

## Configuration

Copy `.env.example` to `.env` and set the API key locally. Do not commit `.env`.

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
AGENTIC_PROVIDER=anthropic   # anthropic | openai | chatgpt
AGENTIC_MODEL=claude-sonnet-4-6
AGENTIC_MAX_ROUNDS=4
```

`--dry-run` and unit tests do not need a real API key.

## Tool Protocol

Agents must return one JSON object:

```json
{
  "message": "I want to inspect the kernel source first.",
  "tool_calls": [
    {
      "tool": "inspect_kernel_source",
      "args": {"entry": "elem_add", "start_line": 1, "end_line": 80}
    }
  ]
}
```

The orchestrator parses this response, validates tool names and required args, executes the
local Python handler, records a `ToolEvent`, and returns updated state to later agents.

## Agents

- **Describer** explains what the kernel appears to implement. It does not record claims or verdicts.
- **Skeptic** raises concrete, testable bug claims with rationale.
- **Experimenter** uses tools such as `run_python_probe`, `append_evidence`, and `update_claim_status` to gather evidence.
- **Judge** records the final verdict with `record_verdict` based on the claim ledger and tool events.

## Tools

Current tool layer:

- `load_artifact`
- `inspect_problem`
- `inspect_kernel_source`
- `list_artifact_files`
- `read_artifact_file`
- `record_claim`
- `read_claim_ledger`
- `append_evidence`
- `update_claim_status`
- `run_python_probe`
- `retrieve_experiment_history`
- `record_verdict`

Tools are deliberately low-level capabilities. They do not encode a fixed verification policy.

## Run Artifacts

Each run writes files under the selected run directory:

```text
run.json
claims.json
tool_events.jsonl
verdict.json        # only when judge records a verdict
probes/             # only when run_python_probe is used
```

`run.json` is the audit record for why the verifier reached its state or verdict.

## Removed Legacy Pipeline

The old fixed verification pipeline has been removed from the primary package. There is no
`kv-run`, `kv-recheck`, `kv-precision-recheck`, operator classifier, standard recheck, or
precision recheck entry point. Verification now routes through `kv-agentic-run` and the
agentic tool protocol.
