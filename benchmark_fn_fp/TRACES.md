# Complete experiment traces

All GLM traces use one directory layout, without provider subdirectories:

```
benchmark_fn_fp/
├── traces_glm/
│   ├── case_01/solo/r1/
│   ├── case_35/debate/r1/
│   ├── case_36/single_call/r1/
│   ├── INDEX.md
│   └── migration_*.json
└── traces_opus5/     claude-opus-5; existing historical paths retained
```

All GLM runs use `<case>/<arm>/rN/`, with `r1`, `r2`, and so on numbering attempts within each case/arm, including failed or incomplete attempts. The same number across arms does not imply the same model, budget or experimental configuration. `trace_meta.json` retains the former label in `original_trial`, the former location in `original_trace_path`, and historical selection order in `selection_sort_key`. Provider and exact API model remain provenance fields alongside dataset, case, arm and current trial. They do not create separate directories or sections in the GLM index. The finite-workload case_36/case_37 pair is dataset `correlation_pair`.

The 2026-09-23 migration combined 12 main GLM runs and 8 case_36/case_37 runs. The initial provider subdirectories were flattened into `traces_glm/<case>/<arm>/<trial>/`; `traces_glm_fireworks` is no longer a separate tree. Every original file was checked by SHA-256, and migration manifests retain the inventories and path changes. The later [correlation-pair cleanup](traces_opus5/migration_20260930_correlation_cleanup.json) relocated its nine Opus attempts to `traces_opus5/case_36/` and `case_37/` and removed eight duplicate GLM source archives after verification. The canonical model trees now hold the pair's records once each; [its consolidated audit](correlation_pair/README.md#evidence-audit) preserves the separate Opus, GLM 32K and GLM 64K findings.

The later `rN` normalization changes directory names and metadata identity/provenance fields, while preserving raw requests, responses, transcripts and probe payloads. Its [migration manifest](traces_glm/migration_20260930_trial_names.json) retains the original metadata bytes and path mapping. Earlier migration manifests remain unchanged; the subsequent correlation-pair cleanup separately records source-archive relocation and deduplication.

Start with [the GLM index](traces_glm/INDEX.md): its first table covers all 104
active cases and three arms, and its lower section retains every historical
trial. Each case/arm links to its earliest completed judgment, including wrong
answers and valid abstentions. Running, failed and missing slots remain visible.
The [2026-09-30 completion report](traces_glm/COMPLETION_20260930.md)
records the completed 312/312 coverage, new results and costs. In general, the
presence of a reserved directory does not mean that its experiment is complete.

Study settings and reviewed evidence are consolidated in
[numerical accuracy](single_call_vs_tools_challenges/README.md) for case_38–case_61 and
[reference and coverage](solo_vs_debate_challenges/README.md) for case_62–case_81.
Their public inputs now share `triton_eval_cases/` with the original suite and
correlation pair (80 active cases); the 24 early pilot cases remain separate.
These directory changes preserve historical trace dataset IDs and raw payloads.

## A tool-enabled run

```
<case>/{solo,debate}/rN/
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

Omit `--trial` to reserve the next unused `rN`, chosen above the largest existing number across the selected cases/arms. You can instead provide an unused number such as `--trial r2`. Before a paid request, the runner reserves the trial directory. Existing trials cannot be overwritten; differing archive files also fail rather than replacing history. To resume a known tool batch, pass its actual reserved number, for example `--skip-existing --trial r2`. Incomplete trials are preserved; retry them under a new `rN`.

For coverage completion, the shared Modal tool runner also accepts
`--only-missing`, which checks valid judgments across earlier trials in the
same dataset and model-family tree:

```sh
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --dataset numerical_pilot --arm both --all --provider fireworks --only-missing --total-output-tokens 32768
```

Correct, incorrect and `needs_more_evidence` judgments all fill a slot; tool
errors, token limits and unfinished attempts do not. The flag preserves old
OpenRouter GLM results as well as Fireworks results. This fills coverage without
establishing a matched-model or matched-budget comparison. Do not launch
duplicate jobs for slots already running. The source-only runner requires an
explicit missing-case list; it currently has no `--only-missing` option.

```
python benchmark_fn_fp/eval_scripts/audit_traces.py
python benchmark_fn_fp/eval_scripts/summarize_traces.py
python benchmark_fn_fp/eval_scripts/index_traces.py
```

Readers recursively discover metadata-aware trials and retain legacy compatibility. Reports group by dataset, provider, exact model, arm and original experimental batch (`original_trial`), with current per-case `rN` paths for links. Saved selection ordering preserves which historical judgment was chosen before the rename. Reports distinguish correct, wrong, abstention, token limit and no verdict. Prices use a run's `pricing_snapshot` when present, otherwise its historical exact-model profile; unknown prices remain unknown. New Fireworks snapshots record the source and check date (2026-09-30 for this batch), with cached input conservatively estimated at the uncached rate. Updating current rates does not reprice old traces. These estimates are not invoices and exclude Modal charges.

When raw API responses exist, their reported usage takes precedence over parsed agent history, including calls that failed during parsing. Missing responses or usage are marked as incomplete cost coverage. Historical runs without raw responses retain their history-based estimates; those estimates do not certify complete API capture.

A trace cannot be rebuilt from a scoreboard. A scoreboard can be rebuilt from traces. No API calls are required to index or audit existing records.
