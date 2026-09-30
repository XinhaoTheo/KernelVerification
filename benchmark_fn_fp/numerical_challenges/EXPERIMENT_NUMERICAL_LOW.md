# GLM numerical-challenge experiment

Storage update (2026-09-30): Batch labels and repeat shorthand below retain their original experimental meaning; original labels are saved in `trace_meta.json.original_trial`. Canonical links use per-case/arm `rN` folders; see [trace naming and migration](../TRACES.md).

This is the completed two-repeat experiment on the six frozen numerical cases.
The same Fireworks GLM model (`accounts/fireworks/models/glm-5p3`) was used for
all three arms. `reasoning_effort=low`, `max_tokens=8192`, and the original
experimental labels were `numerical_low_r1` and `numerical_low_r2`. Every completed
attempt has a raw API trace; tool attempts also have the full GPU probe and
agent trace.

| Trial | Single call, no tools | Solo + tools | Debate + tools |
|---|---:|---:|---:|
| numerical_low_r1 | 4/6 | 6/6 | 6/6 |
| numerical_low_r2 | 2/6 | 6/6 | 6/6 |

The frozen truth is case_38/case_40/case_43 = trust and case_39/case_41/case_42 = reject. In repeat 1 the source
only arm misclassified F and H. In repeat 2 it misclassified C, D, F and H.
Both tool arms classified all six cases correctly in both repeats. Thus the
tool advantage is repeated: single-call accuracy was 4/6 and 2/6, while both
tool arms were 6/6.

There is no observed debate-over-solo accuracy gain in this experiment: solo
and debate were both 6/6 in both repeats. This is evidence against claiming a
debate-specific benefit for these cases, not evidence that debate can never
help. It also means the cases are useful for separating execution from source
only reasoning, but are not yet a successful debate-vs-solo benchmark.

Recorded API estimates for the completed trials were:

| Trial | Single call | Solo | Debate | Total |
|---|---:|---:|---:|---:|
| numerical_low_r1 | $0.015880 | $0.085919 | $0.286048 | $0.387847 |
| numerical_low_r2 | $0.012254 | $0.098141 | $0.300138 | $0.410533 |

The earlier `numerical_r3`, `numerical_r4`, `numerical_probe32768` and
`numerical_smoke8k` attempts are retained separately as gateway errors,
timeouts, or interrupted incomplete traces. They are excluded from the two
repeat accuracy table. The initial 65K/32K failures were caused by long hidden
reasoning; adding the recorded low-effort setting made the complete runs finish.

Canonical traces are under
`benchmark_fn_fp/traces_glm/<case>/<arm>/rN/`, and the generated global
scoreboard is `benchmark_fn_fp/eval_scripts/scoreboard.json`.
