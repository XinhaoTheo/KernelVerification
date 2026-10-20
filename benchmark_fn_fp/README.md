# benchmark_fn_fp

快速导航：

| 看什么 | 入口 | 案例 |
|---|---|---|
| 有工具与无工具的区分案例 | [Tools vs No Tools](SUCCESS_CASES.md) | 9 个案例的总表、详情和复测结果；尚未证明 Solo 与 Debate 有稳定差异 |
| 专门构造的真实 kernel 验证挑战 | [real_kernel_challenges](real_kernel_challenges/README.md) | case_33–case_90、case_163–case_168，共64题 |
| kernel、输入生成器与合同 | [统一公开题目录](triton_eval_cases/) / [题目索引](CASE_INDEX.md) | case_01–case_90、case_163–case_168，共96题 |
| 按输入构造标签的旧实验（已归档） | [archive/input_specific](archive/input_specific/README.md) | case_91–case_138，共48题；不进入当前统计和实验 |
| 早期24题pilot（已归档） | [archive/numerical_pilot](archive/numerical_pilot/README.md) | case_139–case_162及72条GLM记录；停止扩展和重跑 |
| 每次调用、失败和probe | [GLM traces](traces_glm/INDEX.md) | 当前96题，共416条GLM记录，按case/arm/rN |

GPU kernel verification datasets, evaluation programs, and recorded experiments.
Start with [CASE_INDEX.md](CASE_INDEX.md) for the meaning and location of every
case. All datasets and trace directories use the same global `case_<number>` ID;
`case_map.json` is the single authoritative mapping.

**The question it asks.** Given a kernel's source and its contract, can a
verifier decide whether the kernel satisfies that contract — catching real
defects without condemning legitimate implementation differences?

The original FN/FP suite targets fixed-tolerance `allclose`. The later kernel challenges test implementation correctness over a complete public input domain. Selected examples can expose a defect, but do not receive separate labels. Old studies that assigned labels by changing fixed inputs, parameters or input domains are archived and excluded from current results. Historical model and budget settings differ, so aggregate accuracy is a history summary rather than a controlled comparison.

## Current accuracy

October 20, 2026. The current set contains 96 cases. For each model method, use the earliest completed record per case from the [GLM index](traces_glm/INDEX.md). Abstentions and cases without a completed record stay in the denominator and do not count as correct. Allclose still covers the previous 90-case set; the six new cases have no allclose measurement.

| Method | Correct | Wrong verdict | Abstention / unfinished / not run | Accuracy |
|---|---:|---:|---:|---:|
| [Allclose / numerical checks](traces_allclose/README.md) | 49/90 tested | 41 | 6 new cases not run | 54.4% of tested cases |
| Single call, no tools | 78/96 | 17 | 1 | 81.3% |
| Solo + tools | 91/96 | 5 | 0 | 94.8% |
| Debate + tools | 85/96 | 5 | 6 | 88.5% |

Allclose was rerun on the previous 90 cases on October 19. The original 32 cases keep their existing conventional checks; the other 58 use public `make_inputs(seed=0)` defaults and public numerical requirements. These are two different protocols, so the aggregate must be read with the split below. A passed sample is not a proof over the full input domain. The October 20 experiment adds six new cases and keeps their first trials separate from conditional repeats. Historical model settings differ across older cases.

| Method | Original cases 01–32 | Previous challenges 33–90 | New cases 163–168 | Combined |
|---|---:|---:|---:|---:|
| Allclose / numerical checks | 1/32 (3.1%) | 48/58 (82.8%) | Not run | 49/90 tested (54.4%) |
| Single call, no tools | 30/32 (93.8%) | 43/58 (74.1%) | 5/6 (83.3%) | 78/96 (81.3%) |
| Solo + tools | 30/32 (93.8%) | 55/58 (94.8%) | 6/6 (100%) | 91/96 (94.8%) |
| Debate + tools | 27/32 (84.4%) | 52/58 (89.7%) | 6/6 (100%) | 85/96 (88.5%) |

The original suite deliberately targets limitations of naive comparisons, so its 3.1% is not a general estimate of allclose quality. [Allclose protocol, per-case results and costs](traces_allclose/README.md) are saved separately from model traces.

---

## Layout

```
benchmark_fn_fp/
│
├── CASE_INDEX.md       all case ranges, purposes, and links
├── case_map.json       single canonical ID/source/dataset mapping
├── SUCCESS_CASES.md    tools vs no tools: table, case details and repeats
├── success_cases/      one description per documented success case
├── triton/             32 current original cases, with answers ← scorer only
├── triton_eval_cases/  96 public cases: case_01–case_90 and case_163–case_168
├── archive/input_specific/ case_91–case_138, old labels and experiment records
├── archive/numerical_pilot/ case_139–case_162, historical results and 72 GLM trials
├── real_kernel_challenges/ 64 purpose-built challenges; separate results
│
├── eval_scripts/       evaluation scripts and derived scoreboard
├── traces_opus5/       every run on claude-opus-5 (see TRACES.md)
├── traces_glm/         active-case GLM runs, by case/arm/trial (see INDEX.md)
├── traces_allclose/    numerical baseline, frozen protocol and case/rN results
│
├── generation/         design notes and the case builders
└── modal_runner.py     shared Modal GPU execution wrapper
```

There are **96 active cases**: 32 original FN/FP cases and 64 purpose-built kernel challenges. Another **72 cases are archived**: 48 input-specific cases and 24 early pilot cases. The historical total is 168. On October 19, 2026, current cases were renumbered as `case_01`–`case_90`, and archived cases as `case_91`–`case_162`. Six new cases, `case_163`–`case_168`, were added on October 20 without reusing archived IDs. The previous numbering, including the old retired case 03, is recorded in `case_map.json`.

Current public cases share `triton_eval_cases/`: `case_01`–`case_90` and `case_163`–`case_168`. Archived public files and traces live under their archive group. They are excluded from current coverage and default scoring.
Runner `--all` selects the requested dataset's registered cases, not every folder
in the shared directory. Current CLI aliases use the new descriptive names;
historical dataset IDs remain in trace metadata for compatibility.

The old challenge groups retain answer keys, construction searches and GPU records in `archive/input_specific/`. Their author programs remain under `eval_scripts/` and target archived paths. Historical reports can still be read without restoring these cases to the current benchmark. Private answer files are never agent inputs.

---

## `triton/` — the 32 current original cases, with answers

One directory per global case ID, matching `triton_eval_cases/` and the traces.
The historical source names remain in `case_map.json` and private metadata:
`fn` means a real defect is present; `fp` means the kernel is correct.

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

## `triton_eval_cases/` — what a verifier actually sees

The current original public copies are built from `triton/` by
`benchmark_fn_fp/generation/generators/build_eval_cases.py`. The same directory
also holds frozen kernel challenges `case_33`–`case_90`. Cases `case_91`–`case_138` and their measurements are archived. The example
and source-to-public comparison below describe an original-suite case.

```
triton_eval_cases/case_32/
├── kernel.py     (the docstring naming the case is stripped)
├── problem.txt
└── meta.json     {"name","status","passed":null} — no answer
```

### The four differences

| | Answer key (`triton/`) | Verifier's copy (`triton_eval_cases/`) |
|---|---|---|
| Directory name | `case_32` | `case_32` |
| `meta.json` | 2153 bytes, ground truth and mechanism | 68-byte stub |
| `kernel.py` first line | `"""Triton kernel under test: fn21_..."""` | `import torch` |
| `test.py` | present | absent |
| `problem.txt` | — | **identical** |
| `kernel.py` body | — | **identical** |

**The dangerous one is `meta.json`, not the missing `test.py`.** The answer key's
copy states `"group": "FN"` (a defect is present), `"mechanism": "...K //
group_size instead of ceil(...)"` (what the defect is and where), and
`"correct_verdict": "BUGGY"` (the answer outright). Shipping it ends the case.

**Public names use neutral IDs.** Both copies now use `case_NN`; the historical
descriptive source name stays in private metadata. The original IDs were assigned
in shuffled order, so their order does not sort FN before FP. Later datasets also
use the shared global numbering rather than restarting at 01.

**`kernel.py`'s first line** was `"""Triton kernel under test: <case name>."""`,
pinning the answer to the top of the source. Removed; the body is untouched.

**`problem.txt` is shipped in full.** It is the contract and the only basis for a
verdict. Without it a verifier has no standard to check against and the case does
not exist.

The old `case_03` was withdrawn because its defect was in a test file that the verifier could not see. Its replacement is now `case_32` (previously `case_33`). After the October 19 renumbering, current `case_03` is a different, retained kernel; `case_map.json` records this distinction.

---

## `case_map.json`

```json
{
  "cases": { "case_32": "case_32" },
  "case_details": {
    "case_32": {
      "dataset": "benchmark_fn_fp",
      "source_name": "fn21_gptq_group_count_floor_division",
      "public_dir": "triton_eval_cases/case_32",
      "answer_dir": "triton/case_32"
    }
  }
}
```

Illustrative excerpt only. The full registry covers every dataset, previous IDs,
source names and current paths. It stays outside verifier-visible directories;
`CASE_INDEX.md` is generated from it. Do not create a second competing case map.

---

## `eval_scripts/` — evaluation programs and scoreboard

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
third, and a historical full 32-case debate run at the 4096 default produced three cases
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

## `traces_*/` — the record of every run

GLM runs share one directory, arranged by case, arm and trial:

```
benchmark_fn_fp/traces_opus5/case_32/debate/              historical Opus run
benchmark_fn_fp/traces_glm/case_32/debate/r1/
benchmark_fn_fp/traces_glm/case_36/debate/r1/
```

Both GLM model profiles in `eval_scripts/models.py` write to `traces_glm/`. Metadata
retains the API model ID and provider for provenance and cost calculation.
New runs reserve a fresh trial directory and cannot overwrite earlier results.
Trial folders are `r1`, `r2`, and so on within each case/arm, including incomplete
attempts. Original labels and paths remain in `trace_meta.json` as
`original_trial` and `original_trace_path`; `selection_sort_key` preserves
historical selection order. The same `rN` across arms does not establish matching
model settings or budgets.

[`TRACES.md`](TRACES.md) documents the artifacts and migration.
[`traces_glm/INDEX.md`](traces_glm/INDEX.md) links to the GLM runs.

---

## `generation/` — design notes and case builders

| File | Contents |
|---|---|
| `benchmark_design.md` | The original design |
| `benchmark_design_generalized.md` | The thirteen seeds in full, and every hypothesis measurement has falsified |
| `generators/batch*.py` | The builders, each responsible for a few seeds |
| `generators/build_eval_cases.py` | Produces the original-suite copies in `triton_eval_cases/` from `triton/`, with leak checks |
| `generators/sanitize_for_eval.py` | The earlier stripping script, superseded by the above |

`benchmark_design_generalized.md` records hypotheses that were **tested and
found false** — that an obscure repository is harder, that post-cutoff code is
harder — so nobody spends the money testing them again.

---

## `archive/input_specific/` — excluded input-based studies

Cases 34–81 are excluded from current kernel verification. Cases 34–35 use identical code with different input domains. Cases 36–81 change selected input data or fixed parameter data while keeping the computation unchanged. The archive retains 48 public cases, their answers, 299 GLM trials and 11 Opus trials. [The archive index](archive/input_specific/README.md) records the reasons and locations. Case IDs and raw experiment records are preserved.

## `archive/numerical_pilot/` — retired early survey

The [archived pilot](archive/numerical_pilot/README.md) retains `case_139`–`case_162`:
eight attention, eight quantization, and eight recurrence cases. Expansion and
reruns have stopped. Its [original Opus report](archive/numerical_pilot/REPORT.md)
preserves 18 correct answers and six token-limit outcomes. The later GLM
completion batch added all three arms; its 72 attempts are retained under the
archive’s `traces_glm/`. The [historical completion report](traces_glm/COMPLETION_20260930.md)
keeps its original 104-case/312-slot figures. Public cases, private construction
measurements and original responses remain available, outside current scoring.

---

## Running it

```bash
# One case. Run this after changing anything, before running the full set.
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --arm debate --cases case_32

# Full GLM set; automatically reserve the next unused rN across both arms
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --arm both --all --provider fireworks

# Resume after an interruption, skipping cases that already have a trace.
# Only valid when the agents themselves have not changed. Replace r2 with the
# actual trial number printed by the original batch.
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --arm both --all --provider fireworks --trial r2 --skip-existing

# Fireworks whole-domain kernel comparison; each command reserves a new unused rN
python benchmark_fn_fp/eval_scripts/run_single_fireworks.py --dataset real_kernel_challenges --cases case_85,case_86 --max-tokens 32768
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --dataset real_kernel_challenges --arm both --cases case_85,case_86 --provider fireworks --max-tokens 32768

# Audit first, score second.
python benchmark_fn_fp/eval_scripts/audit_traces.py
python benchmark_fn_fp/eval_scripts/summarize_traces.py
python benchmark_fn_fp/eval_scripts/index_traces.py
```

`--max-rounds` defaults per arm: 4 for debate, 10 for solo.

`--max-tokens` defaults to the model profile: 16384 for Opus and 32768 for the
current GLM profiles. Reasoning consumes that budget too. GLM has exhausted
32768 tokens without a final answer, so a repeat may explicitly use 65536.
Record the setting and score token exhaustion separately from a wrong verdict.

---

## Rules for changing the benchmark

1. **The defect must live in a file the verifier is given.** `triton_eval_cases/` ships
   `kernel.py`, `problem.txt` and a stub `meta.json`, and nothing else. A case
   whose defect hides in `test.py` is unanswerable, and worse, every verdict of
   "reject" scores correct regardless of reasoning — which is how
   `fn3_gptq_dequant_group_div_coverage` was withdrawn. A defect in a host
   wrapper is fine **provided the wrapper ships alongside `kernel.py`**.

2. **Rebuild after adding an original-suite case**:
   `python benchmark_fn_fp/generation/generators/build_eval_cases.py`
   It assigns ids to new cases, rebuilds the original-suite public copies, and runs the leak checks
   (banned words, real case names, group-encoding filenames, matched-pair
   indistinguishability). Existing ids are never reshuffled — results are stored
   under them.

3. **Retire a withdrawn case's id; never reuse it.** Reuse makes old results look
   like scores for the new case.

4. **Smoke-test one case before running the full set.** This rule was bought:
   launching a full run on a runner that had not been executed once since being
   edited cost six hours and forty-three minutes to an import in the wrong scope,
   and about $33 to a missing `--max-tokens`.
