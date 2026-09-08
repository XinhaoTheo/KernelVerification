# Numerical-compliance pilot

This pilot tests whether source-only judgment can determine actual numerical
error on a fully specified finite workload. It does not establish that debate
is better than a single agent with execution tools. Only the source-only
baseline is evaluated here.

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
The input generator uses NumPy PCG64, performs construction on the CPU, casts
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

## Files and execution

* `kernels.py`: kernel templates and verifier-visible input generator.
* `build_modal.py`: candidate generation, GPU validation, balanced selection.
* `eval_cases/case_*/`: only `kernel.py` and `problem.txt`; no labels.
* `answer_key.json`: private labels, errors, hashes and environment.
* `candidate_measurements.json`: all GPU construction measurements.
* `run_single.py`: reuses existing baseline prompt and output schema.
* `results_single.json`: model responses, token usage, scores and cost estimate.

From the repository root:

```sh
modal run benchmark_fn_fp/numerical_pilot/build_modal.py
/opt/anaconda3/bin/python benchmark_fn_fp/numerical_pilot/run_single.py
```

The single-call evaluation uses the configured AGENTIC_MODEL or the existing
baseline default, claude-opus-5. It makes one request per case, at most two
concurrently, with 8192 output tokens, no retries and no automatic budget
escalation. Interrupted runs resume recorded cases without resending them.
The answer key is used solely for local scoring. Confidence is recorded but does
not influence labels or scoring. API failures, missing final answers and
needs_more_evidence are reported separately.

Cost estimates use the existing runner's $5/$25 per million input/output token
assumption. They are not billing records, and exclude Modal GPU/build charges.

## Interpretation

Low source-only accuracy establishes at most that these workloads are difficult
to judge without calculation. Abstention is a legitimate recognition of missing
evidence, not a confident wrong answer. Distinguish incorrect verdicts from
abstentions. A later study needs single-agent-with-tools and budget-matched
multi-agent evaluation to attribute an advantage to debate.

Unlike the original FN/FP seed suite, these cases may be decided by a simple
script once the correct reference and metric have been supplied. They test
numeric compliance, not superiority to a correctly configured numerical test.
Do not merge their accuracy into the original 32-case result.
