# Evaluation scripts

Programs for running the evaluation arms, auditing traces, and rebuilding the scoreboard.

| Arm | Command |
|---|---|
| Allclose / numerical checks, all active cases | `python -m modal run baseline1_allclose_modal.py --all --trial r3` |
| One model call, no tools | `python baseline2_single_llm.py --all` |
| One Fireworks call, no tools | `python run_single_fireworks.py --all` |
| One agent with the full toolset | `modal run run_agentic_modal.py --arm solo --all` |
| Four-role debate | `modal run run_agentic_modal.py --arm debate --all` |

```bash
python benchmark_fn_fp/eval_scripts/audit_traces.py       # first: is the run sound?
python benchmark_fn_fp/eval_scripts/summarize_traces.py   # then: what did it score?
```

Audit before scoring. A correct verdict reached through a broken run is not a
result — one case was scored correct for condemning genuine upstream code, which
is how a benchmark case that could not be answered at all went unnoticed and
inflated every figure that counted it.

Allclose has its own [saved results and protocol](../traces_allclose/README.md). `r2` is the complete 90-case measurement; `r1` failed during startup. A new `--trial` preserves earlier results. `--resume` is only for unchanged sources and the same protocol. No model API is used.

## Where numbers come from

`scoreboard.json` is rebuilt from the `../traces_*/` trees by `summarize_traces.py`, and
nothing else writes it.

It used to work the other way. Each runner wrote its own
`results_baselineN.json` and overwrote it wholesale, so a batch of 14 cases
replaced a run of 32: `results_baseline3.json` ended up holding 14 of the 32
debate results, and the single-call arm was spread across eight files that only
meant anything added together. Those files are gone. Their numbers are
reproducible from the traces, which are committed.

A trace cannot be rebuilt from a summary. A summary can always be rebuilt from
traces.

GLM runs use `../traces_glm/<case>/<arm>/rN/`, with API provider,
model and dataset recorded in `trace_meta.json`. Omit `--trial` to choose the next
unused number across the selected cases/arms, or specify an unused `--trial r2`.
Each number denotes an attempt within a case/arm, not a shared configuration.
Former labels and paths are retained as `original_trial` and
`original_trace_path`, with `selection_sort_key` preserving historical ordering.
Both Fireworks runners capture raw API
requests/responses, and tool runs retain their full transcripts, ledgers and
probe artifacts. See [the trace guide](../TRACES.md) and [GLM index](../traces_glm/INDEX.md).
The scorer discovers nested trials and groups them by dataset, provider, model,
arm and original experimental batch, while linking to current per-case `rN`
paths; repeats and the historical case_105/case_106 cohorts are not pooled silently.

## Filling missing arms

The [GLM index](../traces_glm/INDEX.md) starts with coverage for all 102 active
cases, including cases with no traces. It counts valid wrong answers and
`needs_more_evidence` as completed; errors, truncated answers and running jobs
remain separate. The [2026-09-30 completion report](../traces_glm/COMPLETION_20260930.md)
records the historical 104-case batch and its costs, including 24 now-archived
pilot cases. Current reports exclude the archive.

```sh
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --dataset benchmark_fn_fp --arm both --all --provider fireworks --only-missing --total-output-tokens 32768
```

The command above selects the original 32 cases. Use `--dataset real_kernel_challenges` for cases 33–102.

`--only-missing` skips case/arm slots with an existing valid judgment anywhere
in the selected model-family trace tree, including historical GLM providers.
It does not rerun wrong answers or abstentions. Existing incomplete trials are
preserved; use a new `rN` for a retry and do not submit a second batch
while the same slots are still running. The source-only Fireworks runner has
no such flag: pass an audited missing-case list with `--cases`.

Current selections are the original suite (`benchmark_fn_fp`, cases 01–32) and `real_kernel_challenges` (cases 33–102). The input-specific challenge groups (cases 103–150) are archived and excluded from new experiments.
The [early pilot](../archive/numerical_pilot/README.md) is archived and closed
to expansion or reruns. Coverage completion combines historical and new
records, so it does not establish identical model/settings across arms.

New runs record `pricing_snapshot` in trace metadata, including the price
source and check date for Fireworks. Scoring uses that snapshot; older traces
without it retain the historical profile estimates rather than being repriced.
These are API estimates, not invoices, and exclude Modal charges.

## Historical 32-case Opus results

Both agentic arms over all 32 cases, on the code in this commit:

| Arm | Correct | FN (reject) | FP (trust) | Cost |
|---|---|---|---|---|
| solo | 29/32 | 20/21 | 9/11 | $22 |
| debate | 30/32 | 21/21 | 9/11 | $56 |

The debate is right about one more case and costs 2.5x to get there. That case
is `case_01`, a missed defect the solo agent called trust, and it is the only
case in 32 where the two arms disagree at all.

They fail identically on the two they get wrong. `case_25` and `case_28` are
both correct kernels, both rejected, at 0.85 and 0.88–0.93 confidence — and all
four traces are clean: no truncation, no tool failure, no process defect, no
logical inconsistency. The arms did not stumble into those answers; they
reasoned their way to the same wrong one. The adversarial structure did not
improve false-positive suppression, which is the property it was meant to
provide.

These figures describe the historical Opus experiment. Current GLM single-call, Solo and Debate results are listed in the [current benchmark summary](../README.md#current-accuracy). The saved allclose r2 baseline covers the October 19 set of 90 cases. It scores 49/90 under the two documented protocols. The six October 20 cases have reviewed numerical comparison rules but do not yet have allclose results; see the [saved baseline protocol](../traces_allclose/README.md).
