# Results in this directory predate two changes and are not directly comparable

**`case_03` is void.** Its defect lived in `test.py`, which `eval_cases` does not
ship, so the verifier was asked a question it could not answer from what it was
given, and any verdict of "reject" scored correct regardless of reasoning. Every
score recorded for `case_03` in the JSON files here must be dropped, not
re-interpreted:

| run | as recorded | with `case_03` dropped |
|---|---|---|
| baseline 2, single call | 30 / 32 | 30 / 31 |
| baseline 3, debate | 12 / 14 | 11 / 13 |

`case_33` (`fn21_gptq_group_count_floor_division`) replaces it and has not been
included in any baseline run yet. The id `case_03` is retired rather than
reused, so nothing here is silently re-attributed to the new case.

**The agents have changed since these runs.** The preload now hands over the
file list and metadata and numbers the kernel source; the Experimenter batches
independent probes instead of spending a turn on each half-step; the Describer
is told the shape its fields take. Measured on one case, the debate went from
11 turns to 7 and from $2.59 to $1.93. Figures here were produced before all of
that and should be re-measured before being quoted against anything current.
