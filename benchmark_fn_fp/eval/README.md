# The harness

Four arms, one scoreboard, one rule about where numbers come from.

| Arm | Command |
|---|---|
| Fixed-tolerance `allclose` | `modal run baseline1_allclose_modal.py --all` |
| One model call, no tools | `python baseline2_single_llm.py --all` |
| One agent with the full toolset | `modal run run_agentic_modal.py --arm solo --all` |
| Four-role debate | `modal run run_agentic_modal.py --arm debate --all` |

```bash
python benchmark_fn_fp/eval/audit_traces.py       # first: is the run sound?
python benchmark_fn_fp/eval/summarize_traces.py   # then: what did it score?
```

Audit before scoring. A correct verdict reached through a broken run is not a
result — one case was scored correct for condemning genuine upstream code, which
is how a benchmark case that could not be answered at all went unnoticed and
inflated every figure that counted it.

## Where numbers come from

`scoreboard.json` is rebuilt from `../traces/` by `summarize_traces.py`, and
nothing else writes it.

It used to work the other way. Each runner wrote its own
`results_baselineN.json` and overwrote it wholesale, so a batch of 14 cases
replaced a run of 32: `results_baseline3.json` ended up holding 14 of the 32
debate results, and the single-call arm was spread across eight files that only
meant anything added together. Those files are gone. Their numbers are
reproducible from the traces, which are committed.

A trace cannot be rebuilt from a summary. A summary can always be rebuilt from
traces.

## Current results

Both agentic arms over all 32 cases, on the code in this commit:

| Arm | Correct | FN (reject) | FP (trust) | Cost |
|---|---|---|---|---|
| solo | 29/32 | 20/21 | 9/11 | $22 |
| debate | 30/32 | 21/21 | 9/11 | $56 |

The debate is right about one more case and costs 2.5x to get there. That case
is `case_01`, a missed defect the solo agent called trust, and it is the only
case in 32 where the two arms disagree at all.

They fail identically on the two they get wrong. `case_26` and `case_29` are
both correct kernels, both rejected, at 0.85 and 0.88–0.93 confidence — and all
four traces are clean: no truncation, no tool failure, no process defect, no
logical inconsistency. The arms did not stumble into those answers; they
reasoned their way to the same wrong one. The adversarial structure did not
improve false-positive suppression, which is the property it was meant to
provide.

The `allclose` and single-call arms have not been re-measured on current code.
Their scripts are unchanged and still needed — without arm 1 there is no
evidence the benchmark defeats fixed tolerances at all, and without arm 2 there
is no baseline showing what execution is worth. Re-running both costs about $2:
arm 1 makes no model calls.
