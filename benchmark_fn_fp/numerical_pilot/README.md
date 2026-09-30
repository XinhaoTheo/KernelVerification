# Numerical-compliance pilot

This pilot tests whether source-only judgment can determine actual numerical
error on a fully specified finite workload. It does not establish that debate
is better than a single agent with execution tools. The historical study
evaluated only the source-only Opus baseline. The separate GLM completion batch
has completed all three evaluation arms; its results are recorded in
the [completion report](../traces_glm/COMPLETION_20260930.md).

There are three synthetic implementations of real operator families:

* Attention: scaled dot-product attention, with probabilities rounded to FP16.
  Background: https://github.com/Dao-AILab/flash-attention
* Quantization: symmetric per-row 4-bit-style quantized matrix-vector product.
  Background: https://github.com/IST-DASLab/gptq (this is not the GPTQ algorithm).
* Recurrence: diagonal affine state update, with FP16 state storage each step.
  Background: https://github.com/state-spaces/mamba

No upstream code is copied. No defect in an upstream project is alleged.

## Contract and labels

Each case exposes the kernel and its complete deterministic input generator.
The input generator uses NumPy 2.2.6 PCG64, performs construction on the CPU, casts
inputs to FP32, and then transfers them to the T4. A label concerns only that
workload, not all possible inputs. Input hashes and dependency versions are
stored with the private answer key.

The output metric is the flattened L2 error relative to a CPU FP64 reference,
divided by max(reference L2 norm, 0.001 * sqrt(output size)). Fixed budgets,
declared before measuring any candidates, are 0.001 for attention, 0.12 for
quantization, and 0.003 for recurrence. These are explicit exploratory benchmark
requirements, not claimed production accuracy standards.

Every candidate executes three times on a real T4. A second reference formulation
checks the FP64 oracle (NumPy versus Torch for attention/GEMV; sequential versus
convolution for recurrence). Construction selects passing cases at <=75% of the
budget and failing cases at >=125%, preventing labels from depending on a tiny
threshold crossing. A reference comparison is sufficient for these finite
contracts; it is not a formal proof over an input domain.

The builder selects four pairs per family and assigns randomized opaque IDs.
Pairs share structural parameters, so 24 cases are not 24 independent algorithm
families. Numeric generation parameters can still be informative shortcuts.
The candidate search is driven by measured error, never LLM answers. All
candidate measurements remain available, including unsuccessful candidates.

Current global IDs are `case_82`–`case_105`; see the [case index](../CASE_INDEX.md).

## Files and execution

* `kernels.py`: kernel templates and verifier-visible input generator.
* `build_modal.py`: candidate generation, GPU validation, balanced selection.
* `eval_cases/case_*/`: `kernel.py`, `problem.txt` and neutral `meta.json`; no labels.
* `answer_key.json`: private labels, errors, hashes and environment.
* `candidate_measurements.json`: all GPU construction measurements.
* `run_single.py`: historical Opus source-only evaluation.
* `results_single.json`: preserved historical Opus responses, usage and scores.
* `../traces_glm/case_82/` through `case_105/`: separate GLM single-call, solo and debate records as they are collected.

From the repository root:

```sh
modal run benchmark_fn_fp/numerical_pilot/build_modal.py
/opt/anaconda3/bin/python benchmark_fn_fp/numerical_pilot/run_single.py
```

The historical single-call runner uses the configured AGENTIC_MODEL or the existing
baseline default, claude-opus-5. It makes one request per case, at most two
concurrently, with 8192 output tokens, no retries and no automatic budget
escalation. Interrupted runs resume recorded cases without resending them.
The answer key is used solely for local scoring. Confidence is recorded but does
not influence labels or scoring. API failures, missing final answers and
needs_more_evidence are reported separately.

Historical Opus cost estimates use the runner's $5/$25 per million input/output token
assumption. They are not billing records, and exclude Modal GPU/build charges.

The shared `eval_scripts` runners now accept `--dataset numerical_pilot`.
The Modal runner automatically selects the NumPy 2.2.6 image for this dataset
(with PyTorch 2.8.0, Triton 3.4.0 and T4), preserving the frozen input generator.
It checks `answer_key.json` hashes before submitting a paid run. For example:

```sh
modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --dataset numerical_pilot --arm both --all --provider fireworks --only-missing --total-output-tokens 32768
```

For GLM source-only calls, use `eval_scripts/run_single_fireworks.py` with
`--dataset numerical_pilot --cases <missing-case-list>`.
Omitting `--trial` reserves the next unused `rN`; an explicit unused number such
as `--trial r2` is also accepted. Numbers are local to each case/arm, so matching
numbers alone do not establish matching experimental settings.
That runner has no `--only-missing` option; select missing cases from the
[GLM coverage table](../traces_glm/INDEX.md) first. The old Opus results are
retained separately and do not count as GLM coverage. New traces carry dated
pricing snapshots; see [trace accounting](../TRACES.md).

## Interpretation

Low source-only accuracy establishes at most that these workloads are difficult
to judge without calculation. Abstention is a legitimate recognition of missing
evidence, not a confident wrong answer. Distinguish incorrect verdicts from
abstentions. A later study needs single-agent-with-tools and budget-matched
multi-agent evaluation to attribute an advantage to debate.

Unlike the original FN/FP seed suite, these cases may be decided by a simple
script once the correct reference and metric have been supplied. They test
numeric compliance, not superiority to a correctly configured numerical test.
The original comparison used 32 cases at that time; the current original suite has 34. Do not pool this pilot with that historical result.
