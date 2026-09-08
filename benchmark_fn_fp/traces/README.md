# Complete agent traces

Four full runs: two cases, each verified twice — once by a single agent holding
the whole toolset (`solo/`), once by the four-role debate (`debate/`). Same
model, same GPU, same container image, same answer-free copy of the case, so a
reader comparing the two arms on one case is seeing a difference in agent
structure and nothing else.

These are the first complete traces taken. The evaluation runners keep only the
last 20,000 characters of `transcript.md`; everything else stayed in the Modal
container and was discarded when it stopped.

## Why these two cases

The benchmark has 32 cases. A single model call with no tools gets 30 of them
right. `case_03` and `case_04` are the two it gets wrong — one missed defect and
one false alarm — so they are the only cases in the set where the arms can
currently be told apart at all. Both agentic arms get both right.

Read them as two worked examples, not as a measurement. Two cases chosen
*because* the single call fails on them cannot establish an accuracy difference;
they show what the difference looks like when there is one.

## What happened

| case | ground truth | single call | solo | debate |
|---|---|---|---|---|
| `case_03` | reject | missed the defect | reject, conf 0.90 | reject, conf 0.93 |
| `case_04` | trust  | false alarm       | trust, conf 0.85  | trust, conf 0.80  |

| case | arm | turns | tool calls | probes | wall clock | cost |
|---|---|---|---|---|---|---|
| `case_03` | solo   | 7  | 15 | 3 | 268 s | $1.28 |
| `case_03` | debate | 11 | 19 | 2 | 316 s | $2.35 |
| `case_04` | solo   | 9  | 14 | 2 | 217 s | $1.31 |
| `case_04` | debate | 14 | 23 | 3 | 432 s | $3.53 |

Cost is list price for the model used, counting cached reads at 0.1x and 1h
cache writes at 2x.

Both arms decide `case_04` the same way and for the same reason: they measure
the deviation instead of judging it from the source. The worst relative error is
about one fp32 ulp, which is what the operation's own reciprocal-square-root
costs, so the deviation is the contract being met rather than broken. The single
call sees a large relative error described in the problem statement and rejects.
That is the shape of difference worth looking for: the verdict turns on a
quantity that is not in the source and can only be measured.

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

    modal run benchmark_fn_fp/eval/capture_traces_modal.py --cases case_03 --arm solo   --max-rounds 10
    modal run benchmark_fn_fp/eval/capture_traces_modal.py --cases case_03 --arm debate --max-rounds 4

`--max-tokens` defaults to 16384. Leave it there: adaptive thinking is billed
against `max_tokens`, and at the 4096 default whole turns return no text and no
tool call because the budget is spent inside the thinking block.
