# Complete experiment traces

All GLM traces use one directory layout, without provider subdirectories:

```
benchmark_fn_fp/
├── traces_glm/
│   ├── case_01/solo/legacy/
│   ├── case_35/debate/legacy/
│   ├── case_36/single_call/r1/
│   ├── INDEX.md
│   └── migration_*.json
└── traces_opus5/     claude-opus-5; existing historical paths retained
```

New GLM runs use `<case>/<arm>/<trial>/`. Main historical GLM runs use `legacy` as the trial. case_36/case_37 historical runs retain `r1` and `r2_64k`. Provider and exact API model are provenance fields inside `trace_meta.json`, alongside dataset, case, arm and trial. They do not create separate directories or sections in the GLM index. The finite-workload case_36/case_37 pair is dataset `correlation_pair`.

The 2026-09-23 migration combined 12 main GLM runs and 8 case_36/case_37 runs. The initial provider subdirectories have been flattened into `traces_glm/<case>/<arm>/<trial>/`; `traces_glm_fireworks` is no longer a separate tree. Every original file is checked by SHA-256, and migration manifests retain the inventories and path changes. Historical case_36/case_37 source archives remain under `correlation_pair/traces_fireworks` so old reports/links still work; the canonical index and scorer use the unified copies, once each.

Start with [the GLM index](traces_glm/INDEX.md).

## A tool-enabled run

```
<case>/{solo,debate}/<trial>/
├── trace_meta.json    identity, provenance, settings, status
├── transcript.md      complete human-readable agent transcript
├── verdict.json      recorded final verdict (absent if no verdict)
├── claims.json       hypotheses, scope, evidence and resolution
├── tool_events.jsonl every tool call and response
├── run.json          full orchestrator state and per-turn token usage
├── probes/           actual code, stdout, stderr and parsed results
├── llm_calls/        new raw chat-completion capture, one directory per SDK call
│   └── <call_id>/
│       ├── request.json   exact system/user/tool request before the API call
│       ├── response.json  full SDK response, reasoning, usage, finish_reason
│       ├── metadata.json  timing and capture status
│       └── error.json     sanitized failure details, when applicable
├── runner_stdout.txt complete runner log
└── runner_error.txt
```

Set `AGENTIC_LLM_TRACE_DIR` to enable raw chat-completion capture. The shared Modal runner sets this to the run's `llm_calls` directory and exports the whole run archive. API headers, client credentials and dotenv contents are not recorded. Internal SDK HTTP retries are represented by their enclosing SDK call; this is not a packet capture. Process-level hard termination can leave a started call without a response, which must remain incomplete rather than be counted as a wrong judgment.

The Modal runner collects each task independently. A remote failure or an archive import failure does not stop collection of the other tasks. Failed imports retain the returned archive as `recovery_run_*.tar.gz`, with sanitized failure details and `status: error`. The batch exits nonzero after collecting its results if any task failed. Container files are returned when the remote function finishes; a container killed before that return can still lose its unreturned files.

Historical tool runs preserve full state and probes, but lacked raw per-call API payloads. Those missing requests/responses cannot be reconstructed. Their metadata says so; importing them does not manufacture old raw traces.

## A source-only run

A `single_call` has no claim ledger or probes. It records exact `request.json`, system/user prompts, `raw_response.json`, `response_text.json`, `response_thinking.txt`, `usage.json`, `transcript.md`, `verdict.json`, and raw `llm_calls/` for new runs. An empty final answer remains `verdict: null` in the readable verdict view; `finish_reason=length` is scored separately as token exhaustion.

## Repeated experiments and scoring

Choose one new trial name for a comparison batch. Before a paid request, the runner reserves the trial directory. Existing trials cannot be overwritten; differing archive files also fail rather than replacing history. To resume a known batch with completed tool trials, use `--skip-existing --trial <same-name>`. Incomplete trials are preserved; retry them under a new trial ID.

```
python benchmark_fn_fp/eval_scripts/audit_traces.py
python benchmark_fn_fp/eval_scripts/summarize_traces.py
python benchmark_fn_fp/eval_scripts/index_traces.py
```

Readers recursively discover metadata-aware trials and retain legacy paths. Reports group by dataset, provider, exact model, arm and trial, and distinguish correct, wrong, abstention, token limit and no verdict. Prices come from the exact model profile, not the common GLM directory name; unknown prices remain unknown.

When raw API responses exist, their reported usage takes precedence over parsed agent history, including calls that failed during parsing. Missing responses or usage are marked as incomplete cost coverage. Historical runs without raw responses retain their history-based estimates; those estimates do not certify complete API capture.

A trace cannot be rebuilt from a scoreboard. A scoreboard can be rebuilt from traces. No API calls are required to index or audit existing records.
