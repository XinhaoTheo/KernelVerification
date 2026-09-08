# benchmark_fn_fp

An adversarial benchmark for GPU kernel verification, and the harness that runs
it.

**The question it asks.** Given a kernel's source and its contract, can a
verifier decide whether the kernel satisfies that contract — catching real
defects without condemning legitimate implementation differences?

**What it is built against.** Fixed-tolerance `allclose` comparison. Every case
is constructed so that method fails: either the defect is invisible on ordinary
inputs (FN cases), or a correct implementation is rejected by the tolerance
(FP cases).

---

## Layout

```
benchmark_fn_fp/
│
├── triton/          32 cases with their answer keys   ← scorer only
├── eval_cases/      the same 32, answers removed      ← what a verifier sees
├── case_map.json    the mapping between them          ← lives outside eval_cases
│
├── eval/            harness and scoreboard
├── traces/          the complete record of every run
│
├── generation/      design notes and the case builders
├── numerical_pilot/ early numerical survey (finished, kept for reference)
└── modal_runner.py  shared Modal GPU execution wrapper
```

---

## `triton/` — the 32 cases, with answers

One directory per case, named for the case. `fn` means a real defect is present;
`fp` means the kernel is correct.

| File | Contents |
|---|---|
| `kernel.py` | The kernel under test, adapted from real upstream code (AutoGPTQ, Liger-Kernel, vLLM, SGLang, FlashAttention, Mamba) |
| `problem.txt` | **The contract** — what the operator must do. The sole basis for a verdict |
| `test.py` | Reference implementation and triggering conditions, proving the case actually holds |
| `meta.json` | **The answer** — ground truth, the mechanism, why `allclose` misses it |

```json
{
  "group": "FN",                  // FN = a real defect, FP = actually correct
  "seed_class": "FN3",            // which seed it belongs to
  "mechanism": "...",             // how the defect stays hidden
  "expected": {
    "correct_verdict": "BUGGY",   // BUGGY = reject, CORRECT = trust
    "naive_allclose_verdict": "PASS on divisible K=4096, FAIL on K=304"
  }
}
```

**This directory must never reach a verifier**: `meta.json` holds the answer and
`test.py` holds a reference implementation.

### Two families, thirteen seeds

**FN (false negative) — the defect is real but ordinary testing misses it**

| Seed | How the defect hides |
|---|---|
| FN1 | A tie-break that contradicts the contract, on ties random data almost never produces |
| FN2 | A missing clamp or mask, on a branch mild inputs never reach |
| FN3 | Whole blocks handled correctly, the tail block wrong, and the test sizes divide evenly |
| FN4 | Reads the wrong head, expert or page — but every source holds similar data |
| FN5 | A quantization rule that only breaks on the outliers of real distributions, which Gaussian noise does not contain |
| FN6 | Per-step error too small to see, exceeding tolerance only after long accumulation |
| FN7 | Two formulas indistinguishable at ordinary magnitudes, divergent near zero |

**FP (false positive) — the kernel is correct but a tolerance rejects it**

| Seed | Why it gets rejected |
|---|---|
| FP1 | The contract admits several answers on a tie; the reference picked one |
| FP2 | The same information in a different layout |
| FP3 | The same additions in a different order, so rounding differs |
| FP4 | A low-precision format rounds by design, judged against an fp32 tolerance |
| FP5 | The kernel is deliberately random (stochastic rounding, sampling) |
| FP6 | Near zero, relative error stops being a meaningful metric |

---

## `eval_cases/` — what a verifier actually sees

Built from `triton/` by
`benchmark_fn_fp/generation/generators/build_eval_cases.py`.

```
eval_cases/case_33/
├── kernel.py     (the docstring naming the case is stripped)
├── problem.txt
└── meta.json     {"name","status","passed":null} — no answer
```

### The four differences

| | Answer key (`triton/`) | Verifier's copy (`eval_cases/`) |
|---|---|---|
| Directory name | `fn21_gptq_group_count_floor_division` | `case_33` |
| `meta.json` | 2153 bytes, ground truth and mechanism | 68-byte stub |
| `kernel.py` first line | `"""Triton kernel under test: fn21_..."""` | `import torch` |
| `test.py` | present | absent |
| `problem.txt` | — | **identical** |
| `kernel.py` body | — | **identical** |

**The dangerous one is `meta.json`, not the missing `test.py`.** The answer key's
copy states `"group": "FN"` (a defect is present), `"mechanism": "...K //
group_size instead of ceil(...)"` (what the defect is and where), and
`"correct_verdict": "BUGGY"` (the answer outright). Shipping it ends the case.

**The directory name is not a detail either.** It enters every agent's prompt on
every turn through `state.entry`, and `list_artifact_files` exposes it again. The
`fn` prefix announces that a defect exists and the rest of the name announces
what it is. Hence `case_NN`, assigned in shuffled order — otherwise the numbering
itself would sort FN before FP.

**`kernel.py`'s first line** was `"""Triton kernel under test: <case name>."""`,
pinning the answer to the top of the source. Removed; the body is untouched.

**`problem.txt` is shipped in full.** It is the contract and the only basis for a
verdict. Without it a verifier has no standard to check against and the case does
not exist.

`case_03` is a retired id. That case put its defect in `test.py`, which is not
shipped, so the verifier could not see the faulty code and any verdict of
"reject" scored correct regardless of its reasoning. The case was withdrawn and
its id retired rather than reused; `case_33` replaces it.

---

## `case_map.json`

```json
{ "cases": { "case_33": "fn21_gptq_group_count_floor_division", ... } }
```

Deliberately outside `eval_cases/`, read only by the scorer. A verifier sees
`case_26` and cannot tell it is an FP case.

---

## `eval/` — the harness

### The four arms

| Arm | Command | Can it run experiments? |
|---|---|---|
| Fixed-tolerance `allclose` | `baseline1_allclose_modal.py` | — |
| One model call, no tools | `baseline2_single_llm.py` | No — reading and reasoning only |
| **One agent with the full toolset** | `run_agentic_modal.py --arm solo` | Yes — writes code and runs it on a GPU |
| **Four-role debate** | `run_agentic_modal.py --arm debate` | Yes, and the roles challenge each other |

The four are an ablation. Arm 1 to arm 2 measures what reasoning is worth; arm 2
to arm 3 measures what execution is worth; arm 3 to arm 4 measures what the
adversarial structure is worth.

The last two share one script. They were previously two nearly identical files
plus a third for single-case capture — 506 lines whose only meaningful
difference was the agent roster. Keeping three copies in step was the direct
cause of the worst bug here: `--max-tokens` was added to two and missed on the
third, and a full 32-case debate run at the 4096 default produced three cases
with zero claims and zero probes and about $33 of nothing.

### Supporting files

| File | Purpose |
|---|---|
| `common.py` | Loads cases, scores against `case_map.json` |
| `traces.py` | **Every runner writes its trace through this** — see below |
| `audit_traces.py` | Eight automated checks over every trace |
| `summarize_traces.py` | **Recomputes the scoreboard from traces**, writes `scoreboard.json` |
| `README.md` | Which historical results are void, and why |

**The scoreboard is derived, never authored.** Each runner used to write its own
`results_baselineN.json` and overwrite it wholesale, so a batch of 14 cases
replaced a run of 32 — `results_baseline3.json` ended up holding 14 of the 32
debate results, and the single-call arm was scattered across eight files that
only meant anything added together. A trace cannot be rebuilt from a summary; a
summary can always be rebuilt from traces. So `summarize_traces.py` is the only
thing that writes one.

**Why traces are mandatory.** The first fifteen runs kept only the last 20,000
characters of `transcript.md` and let the container discard everything else.
Three defects that each changed a conclusion — a Skeptic spending its turn
re-reading material already in its prompt, `record_description_update` rejecting
the Describer's first call in every single run, an agent reaching the right
verdict for three wrong reasons — were invisible until full traces existed, and
every one of them had been happening on every run the whole time. `tests/` fails
if a runner stops writing a trace or omits `--max-tokens`.

---

## `traces/` — the record of every run

```
traces/<case_id>/<arm>/
├── transcript.md      the run as prose, in order  ← start here
├── verdict.json       verdict, confidence, reasoning
├── claims.json        the claim ledger: hypotheses, evidence, scope, status
├── tool_events.jsonl  one line per tool call, with arguments and result
├── run.json           full state, including per-turn token usage
├── probes/            every probe the agent wrote: source, stdout, stderr
└── runner_stdout.txt  the runner's own log
```

`<arm>` is `solo` or `debate`, so both configurations of one case sit side by
side.

**How to read one.** Only `transcript.md` is meant for a person. It has four
sections and reads fastest backwards: `## Verdict` → `## Claims` → then
`## Timeline` if something needs checking.

`probes/` is the part a single model call has no counterpart for: `tN_probe.py`
is code the agent wrote and ran on a real GPU during the run, and
`tN_stdout.txt` is what came back. Every measurement a verdict cites is
reproducible from those files.

---

## `generation/` — design notes and case builders

| File | Contents |
|---|---|
| `benchmark_design.md` | The original design |
| `benchmark_design_generalized.md` | The thirteen seeds in full, and every hypothesis measurement has falsified |
| `generators/batch*.py` | The builders, each responsible for a few seeds |
| `generators/build_eval_cases.py` | Produces `eval_cases/` from `triton/`, with leak checks |
| `generators/sanitize_for_eval.py` | The earlier stripping script, superseded by the above |

`benchmark_design_generalized.md` records hypotheses that were **tested and
found false** — that an obscure repository is harder, that post-cutoff code is
harder — so nobody spends the money testing them again.

---

## `numerical_pilot/` — early survey (kept for reference)

Measurements taken before the benchmark existed: how large the deviation from a
*legitimate* implementation difference actually gets, which is what the FP cases'
tolerances had to be set against. `REPORT.md` has the conclusions,
`answer_key.json` and `candidate_measurements.json` the raw numbers. Finished;
not modified further.

---

## Running it

```bash
# One case. Run this after changing anything, before running the full set.
modal run benchmark_fn_fp/eval/run_agentic_modal.py --arm debate --cases case_33

# Full set
modal run benchmark_fn_fp/eval/run_agentic_modal.py --arm solo   --all
modal run benchmark_fn_fp/eval/run_agentic_modal.py --arm debate --all

# Resume after an interruption, skipping cases that already have a trace.
# Only valid when the agents themselves have not changed.
modal run benchmark_fn_fp/eval/run_agentic_modal.py --arm debate --all --skip-existing

# Audit first, score second.
python benchmark_fn_fp/eval/audit_traces.py
python benchmark_fn_fp/eval/summarize_traces.py
```

`--max-rounds` defaults per arm: 4 for debate, 10 for solo.

`--max-tokens` defaults to 16384. Do not lower it. Adaptive thinking is billed
against `max_tokens`, and at 4096 a whole turn can be spent inside the thinking
block and return no text and no tool call — one full run produced three cases
with zero claims and zero probes, with a Judge writing "the debate produced no
claims and no evidence" and recording a verdict anyway. Measured peaks per role
are 6.3k–8.5k tokens, so 16384 leaves about half in reserve.

---

## Rules for changing the benchmark

1. **The defect must live in a file the verifier is given.** `eval_cases/` ships
   `kernel.py`, `problem.txt` and a stub `meta.json`, and nothing else. A case
   whose defect hides in `test.py` is unanswerable, and worse, every verdict of
   "reject" scores correct regardless of reasoning — which is how
   `fn3_gptq_dequant_group_div_coverage` was withdrawn. A defect in a host
   wrapper is fine **provided the wrapper ships alongside `kernel.py`**.

2. **Rebuild after adding a case**:
   `python benchmark_fn_fp/generation/generators/build_eval_cases.py`
   It assigns ids to new cases, rebuilds `eval_cases/`, and runs the leak checks
   (banned words, real case names, group-encoding filenames, matched-pair
   indistinguishability). Existing ids are never reshuffled — results are stored
   under them.

3. **Retire a withdrawn case's id; never reuse it.** Reuse makes old results look
   like scores for the new case.

4. **Smoke-test one case before running the full set.** This rule was bought:
   launching a full run on a runner that had not been executed once since being
   edited cost six hours and forty-three minutes to an import in the wrong scope,
   and about $33 to a missing `--max-tokens`.
