# Complete agent traces

Four full runs: two cases, each verified twice — once by a single agent holding
the whole toolset (`solo/`), once by the four-role debate (`debate/`). Same
model, same GPU, same container image, same answer-free copy of the case, so a
reader comparing the two arms on one case is seeing a difference in agent
structure and nothing else.

The evaluation runners keep only the last 20,000 characters of `transcript.md`;
everything else stays in the Modal container and is discarded when it stops.
These are captured with `capture_traces_modal.py`, which brings the whole run
directory back.

## Why these two cases

`case_04` is one of the two cases in the 32 that a single model call with no
tools gets wrong — it raises a false alarm there. `case_33` is a replacement
case, built after the case it replaces turned out to be unanswerable from the
files the verifier is given; these are its first runs.

Two cases chosen this way cannot establish an accuracy difference between the
arms. Read them as worked examples of what a run looks like, not as a
measurement of how often one arm beats another.

## What happened

| case | ground truth | single call | solo | debate |
|---|---|---|---|---|
| `case_33` | reject | not yet run | reject, conf 0.95 | reject, conf 0.93 |
| `case_04` | trust  | false alarm | trust, conf 0.85  | trust, conf 0.68  |

| case | arm | turns | tool calls | probes | wall clock | cost |
|---|---|---|---|---|---|---|
| `case_33` | solo   | 5 |  8 | 2 |  73 s | $0.50 |
| `case_33` | debate | 7 | 17 | 4 | 262 s | $1.93 |
| `case_04` | solo   | 9 | 14 | 2 | 217 s | $1.31 |
| `case_04` | debate | 8 | 19 | 4 | 298 s | $1.62 |

Cost is list price for the model used, counting cached reads at 0.1x and 1h
cache writes at 2x.

`case_04` is worth reading first. Both arms decide it the same way and for the
same reason: they measure the deviation instead of judging it from the source.
The worst relative error is about one fp32 ulp, which is what the operation's
own reciprocal-square-root costs, so the deviation is the contract being met
rather than broken. The single call sees a large relative error and rejects.
That is the shape of difference worth looking for: the verdict turns on a
quantity that is not in the source and can only be measured.

`case_04` is also unstable. Two debate runs on identical code returned opposite
verdicts, both at confidence 0.68, and both times the disputed claim was the
same one: the kernel returns fp32 for a bf16 input, and `problem.txt` never
states the output storage dtype. Whether the Skeptic scopes that `unknown` or
`in_scope` decides the verdict, because the Judge is told an `unknown`-scope
claim should usually produce trust. One case, one scope call, opposite answers.

## Layout

    <case>/<arm>/
      transcript.md      the run as prose, in order — start here
      run.json           full state: every turn, its tool calls, token usage
      tool_events.jsonl  one line per tool call, with arguments and result
      claims.json        the claim ledger, with evidence and scope
      verdict.json       the final verdict, confidence, and reason
      probes/            every probe: source, stdout, stderr
      runner_stdout.txt  the runner's own log

`probes/` is the part that has no counterpart in a single call. `tN_probe.py` is
code the agent wrote and ran on the GPU during the run, and `tN_stdout.txt` is
what came back. Everything the verdict cites as a measurement is reproducible
from those files.

## Reproducing

    modal run benchmark_fn_fp/eval/capture_traces_modal.py --cases case_33 --arm solo   --max-rounds 10
    modal run benchmark_fn_fp/eval/capture_traces_modal.py --cases case_33 --arm debate --max-rounds 4

`--max-tokens` defaults to 16384. Leave it there: adaptive thinking is billed
against `max_tokens`, and at the 4096 default whole turns return no text and no
tool call because the budget is spent inside the thinking block.
