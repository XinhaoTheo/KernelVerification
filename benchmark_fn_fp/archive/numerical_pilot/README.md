# Numerical-compliance pilot（已归档）

本组是已停止扩展和重跑的早期探索：24题（case_82–case_105）及72条GLM运行均归档在本目录，不计入当前80题的覆盖或评分。公开题、原始响应、构造代码和冻结答案保留用于追溯；历史报告的数字不改。当前实验入口见 [benchmark指南](../../README.md)。

This pilot tests whether source-only judgment can determine actual numerical
error on a fully specified finite workload. It does not establish that debate
is better than a single agent with execution tools. The historical study
evaluated only the source-only Opus baseline. The separate GLM completion batch
has completed all three evaluation arms; its results are recorded in
the [completion report](../../traces_glm/COMPLETION_20260930.md).

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

Current global IDs are `case_82`–`case_105`; see the [case index](../../CASE_INDEX.md).

## Preserved files and historical execution

* `kernels.py`: kernel templates and verifier-visible input generator.
* `build_modal.py`: candidate generation, GPU validation, balanced selection.
* `eval_cases/case_*/`: `kernel.py`, `problem.txt` and neutral `meta.json`; no labels.
* `answer_key.json`: private labels, errors, hashes and environment.
* `candidate_measurements.json`: all GPU construction measurements.
* `run_single.py`: historical Opus source-only evaluation.
* `results_single.json`: preserved historical Opus responses, usage and scores.
* [REPORT.md](REPORT.md): original Opus source-only results and explanations.
* [traces_glm/](traces_glm/): all 72 archived GLM attempts, cases82–105 × single-call/solo/debate.

The scripts are preserved as historical provenance, not current execution entry points.
Do not rebuild these frozen cases or submit new pilot calls.

The historical single-call runner uses the configured AGENTIC_MODEL or the existing
baseline default, claude-opus-5. It makes one request per case, at most two
concurrently, with 8192 output tokens, no retries and no automatic budget
escalation. Interrupted runs resume recorded cases without resending them.
The answer key is used solely for local scoring. Confidence is recorded but does
not influence labels or scoring. API failures, missing final answers and
needs_more_evidence are reported separately.

Historical Opus cost estimates use the runner's $5/$25 per million input/output token
assumption. They are not billing records, and exclude Modal GPU/build charges.

The later GLM completion batch used the frozen NumPy 2.2.6 generator,
PyTorch 2.8.0, Triton 3.4.0 and T4; source/contract hashes were checked before
submission. Its three arms each produced 24 valid judgments: source-only
18 correct, 5 wrong and 1 abstention; solo 24 correct; debate 23 correct and
1 abstention. The 72 original trials are retained here, including all raw API
requests/responses and probes. This is separate from the original Opus report,
whose six missing answers were token-limit outcomes. See the unchanged
[historical completion results](../../traces_glm/COMPLETION_20260930.md).

Original batches remain in metadata; local `rN` paths do not imply equal
settings. Archived traces are excluded from the current
[GLM index](../../traces_glm/INDEX.md) and shared scoring;
[trace accounting](../../TRACES.md) explains historical pricing and capture.

## Interpretation

Low source-only accuracy establishes at most that these workloads are difficult
to judge without calculation. Abstention is a legitimate recognition of missing
evidence, not a confident wrong answer. Distinguish incorrect verdicts from
abstentions. The later GLM study included both tool arms but showed no extra debate
accuracy advantage; broader attribution requires matched-budget evidence.

Unlike the original FN/FP seed suite, these cases may be decided by a simple
script once the correct reference and metric have been supplied. They test
numeric compliance, not superiority to a correctly configured numerical test.
The original comparison used 32 cases at that time; the current original suite has 34. Do not pool this pilot with that historical result.
