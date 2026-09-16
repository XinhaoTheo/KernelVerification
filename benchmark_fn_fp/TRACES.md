# Complete agent traces

One tree per model. Within a tree, one directory per case, and inside that one
directory per arm — so `traces_opus5/case_33/debate/` and
`traces_glm/case_33/debate/` are the same case and the same arm, run by
different models.

```
benchmark_fn_fp/
├── traces_opus5/     claude-opus-5      32 cases x {solo, debate}
└── traces_glm/       z-ai/glm-5.3-flash  1 case  x {solo, debate}
```

The tree comes from the model, via `eval/models.py`. A model with no profile
gets its own `traces_<slug>/` rather than sharing one, because a $1 run on an
open model must not land anywhere near the $88 of Opus runs the current numbers
come from. `tests/` fails if two models ever name the same tree.

Nothing here is a summary. `eval/scoreboard.json` is derived from these trees by
`summarize_traces.py` and can be rebuilt at any time; a trace cannot be rebuilt
from a scoreboard.

## What one run holds

```
traces_<model>/<case_id>/<arm>/
├── transcript.md      the run as prose, in order  ← start here
├── verdict.json       verdict, confidence, reasoning
├── claims.json        the claim ledger: hypotheses, evidence, scope, status
├── tool_events.jsonl  one line per tool call, with arguments and result
├── run.json           full state, including per-turn token usage
├── probes/            every probe the agent wrote: source, stdout, stderr
└── runner_stdout.txt  the runner's own log
```

Only `transcript.md` is meant for a person. It has four sections and reads
fastest backwards: `## Verdict` → `## Claims` → then `## Timeline` if something
needs checking.

`probes/` is the part a single model call has no counterpart for. `tN_probe.py`
is code the agent wrote and ran on a real GPU during the run, `tN_stdout.txt` is
what came back, and `tN_stderr.txt` is how it failed when it did. Every
measurement a verdict cites is reproducible from those files.

## What the two trees show so far

`traces_opus5/` is the measurement: 32 cases, both arms, and the source of every
number in `eval/README.md`.

`traces_glm/` is one case, run twice, to answer whether an open model can carry
the tool-calling load at all. It can — both arms reached the correct verdict —
but the probe code it writes is a different story:

| tree | arm | probes | failed | turns | cost |
|---|---|---|---|---|---|
| `traces_opus5` | solo | 2 | 0 | 5 | $0.47 |
| `traces_opus5` | debate | 9 | 0 | 10 | $3.41 |
| `traces_glm` | solo | 5 | **4** | 10 | $0.018 |
| `traces_glm` | debate | 8 | **5** | 10 | $0.022 |

Opus wrote 11 probes and none failed. GLM wrote 13 and 9 failed — tensor-shape
errors in its int4 bit-packing, and one CUDA illegal memory access that showed
up as an XID fault in the host log. It recovered each time by reading the
traceback and rewriting, which is the loop working as intended, but the turns
that recovery costs come out of the same budget the run needs for real work.

## Why full traces are kept

The first fifteen runs kept only the last 20,000 characters of `transcript.md`
and let the container discard the rest. Three defects that each changed a
conclusion — a Skeptic spending its turn re-reading material already in its
prompt, `record_description_update` rejecting the Describer's first call in every
single run, an agent reaching the right verdict for three wrong reasons — were
invisible until full traces existed, and every one of them had been happening on
every run the whole time. `tests/` now fails if a runner stops writing a trace.
