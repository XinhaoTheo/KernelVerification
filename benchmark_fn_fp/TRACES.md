# Complete experiment traces

Current GLM traces use one directory layout, without provider subdirectories.
Excluded input-based cases and the retired pilot keep their original records under the archive:

```
benchmark_fn_fp/
├── traces_glm/
│   ├── case_01/solo/r1/
│   ├── case_36/debate/r1/
│   ├── case_90/single_call/r1/
│   ├── INDEX.md
│   └── migration_*.json
├── traces_opus5/     claude-opus-5; existing historical paths retained
└── archive/
    ├── input_specific/traces_glm/     299 trials, case_91–case_138
    ├── input_specific/traces_opus5/   11 trials, case_91–case_138
    └── numerical_pilot/traces_glm/    72 trials, case_139–case_162
```

All GLM runs use `<case>/<arm>/rN/`, with `r1`, `r2`, and so on numbering attempts within each case/arm, including failed or incomplete attempts. The same number across arms does not imply the same model, budget or experimental configuration. `trace_meta.json` retains the former label in `original_trial`, the former location in `original_trace_path`, and historical selection order in `selection_sort_key`. Provider and exact API model remain provenance fields alongside dataset, case, arm and current trial. They do not create separate directories or sections in the GLM index. Cases93/94 now belong to `single_call_vs_tools_challenges`; their original `correlation_pair` source field remains in metadata.

The 2026-09-23 migration combined 12 main GLM runs and 8 case_93/case_94 runs. The initial provider subdirectories were flattened into `traces_glm/<case>/<arm>/<trial>/`; `traces_glm_fireworks` is no longer a separate tree. Every original file was checked by SHA-256, and migration manifests retain the inventories and path changes. The later [correlation-pair cleanup](traces_opus5/migration_20260930_correlation_cleanup.json) relocated its nine Opus attempts to the case directories now stored at `archive/input_specific/traces_opus5/case_93/` and `case_94/` and removed eight duplicate GLM source archives after verification. The canonical model trees now hold the pair's records once each; [its consolidated audit](archive/input_specific/single_call_vs_tools_challenges/README.md#correlation-audit) preserves the separate Opus, GLM 32K and GLM 64K findings.

The later `rN` normalization changes directory names and metadata identity/provenance fields, while preserving raw requests, responses, transcripts and probe payloads. Its [migration manifest](traces_glm/migration_20260930_trial_names.json) retains the original metadata bytes and path mapping. Earlier migration manifests remain unchanged; the subsequent correlation-pair cleanup separately records source-archive relocation and deduplication.

Start with [the GLM index](traces_glm/INDEX.md): it covers all 90 active cases and three arms, with 392 recorded GLM trials. Each case/arm links to its earliest completed judgment, including wrong answers and valid abstentions. Running, failed and missing slots remain visible.
The [2026-09-30 completion report](traces_glm/COMPLETION_20260930.md)
records the historical 104-case/312-slot completion, including 24 now-retired
pilot cases. Their 72 GLM trials are preserved in
[the pilot archive](archive/numerical_pilot/README.md), excluded from current
coverage and scoring; expansion and reruns have stopped. In general, the
presence of a reserved directory does not mean that its experiment is complete.

Cases 91–138 (previously 34–81) and their GLM and Opus records now live in [archive/input_specific](archive/input_specific/README.md). These studies selected data, parameters or input domains to change labels for the same computation. They are excluded from current coverage and scoring. Current public cases are 01–90: originals 01–32 and [kernel challenges](real_kernel_challenges/README.md) 33–90. Directory IDs were compacted on October 19, 2026. Raw records and historical verdicts are preserved; case_map.json records every old ID.

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
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --dataset real_kernel_challenges --arm both --all --provider fireworks --only-missing --total-output-tokens 32768
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

Readers discover metadata-aware trials in current trace trees and retain legacy compatibility. All archived cases are excluded by default. `summarize_traces.py --include-archived --json` reads both archive groups without overwriting the current scoreboard. Reports group by dataset, provider, exact model, arm and original experimental batch (`original_trial`), with current per-case `rN` paths for links. Saved selection ordering preserves which historical judgment was chosen before the move. Reports distinguish correct, wrong, abstention, token limit and no verdict. Prices use a run's `pricing_snapshot` when present, otherwise its historical exact-model profile; unknown prices remain unknown. New Fireworks snapshots record the source and check date (2026-09-30 for this batch), with cached input conservatively estimated at the uncached rate. Updating current rates does not reprice old traces. These estimates are not invoices and exclude Modal charges.

When raw API responses exist, their reported usage takes precedence over parsed agent history, including calls that failed during parsing. Missing responses or usage are marked as incomplete cost coverage. Historical runs without raw responses retain their history-based estimates; those estimates do not certify complete API capture.

A trace cannot be rebuilt from a scoreboard. A scoreboard can be rebuilt from traces. No API calls are required to index or audit existing records.

## Allclose records

`traces_allclose/<case>/rN/` stores the numerical baseline separately from model calls. Each trial saves source files, source hashes, generator settings, comparison rules, full stdout, errors, environment and elapsed time. The root contains a protocol manifest and a result summary for each batch. `r1` records a cancelled startup; `r2` contains all 90 completed checks. See the [allclose index](traces_allclose/README.md).
