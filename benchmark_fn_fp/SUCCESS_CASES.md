# Tools vs No Tools

Updated: October 20, 2026. Model: GLM-5p3 via Fireworks.

These 11 cases were selected to compare a single model call without tools with workflows that use tools. Solo and Debate both use tools. A correct decision can be **Accept** or **Reject**.

**These cases target the difference between tools and no tools. The results do not establish a stable difference between Solo and Debate.** Their results are shown separately to keep each workflow's successes and failures visible.

Each kernel has **one fixed Accept or Reject label for its full problem description**. If a kernel fails on an input that the problem description permits, the kernel is incorrect.

We ran all tests on an NVIDIA T4 with PyTorch 2.8.0 and Triton 3.4.0.

Case numbers match the current [case index](CASE_INDEX.md).

Select a case number for the detailed explanation and traces. The table shows the initial trials. The repeat table below gives the results from five more trials for cases 74, 76, 78, 86, and 90, and two more trials for cases 94 and 100.

| Case | Kernel | Ground truth | Allclose | Single call, no tools | Solo + tools | Debate + tools |
|---|---|---|---|---|---|---|
| [36](success_cases/case_36.md) | RMSNorm backward | Reject | Reject | Accept | Reject | No verdict: token limit |
| [41](success_cases/case_41.md) | Paged sliding-window attention | Reject | Accept | Accept | Reject | Reject |
| [54](success_cases/case_54.md) | erf approximation | Reject | Reject | Accept | Reject | Reject |
| [55](success_cases/case_55.md) | GELU approximation | Accept | Accept | Reject | Accept | Accept |
| [74](success_cases/case_74.md) | Atan approximation | Reject | Reject | Accept | Reject | Reject |
| [76](success_cases/case_76.md) | Sigmoid rounding | Reject | Accept | Accept | Reject | Reject |
| [78](success_cases/case_78.md) | NTT coefficient conversion | Reject | Reject | Accept | Reject | Reject |
| [86](success_cases/case_86.md) | Recurrence order | Reject | Reject | Accept | Reject | Reject |
| [90](success_cases/case_90.md) | FFT address | Reject | Reject | Accept | Reject | Reject |
| [94](success_cases/case_94.md) | log1p range reduction | Reject | Not run | Accept | Reject | Reject |
| [100](success_cases/case_100.md) | asinh branch range | Reject | Not run | Accept | Reject | Reject |

Allclose is a separate numerical test baseline, with no model calls. It uses the complete October 19 `r2` batch: existing original checks and one public default input per new kernel. It returned correct decisions for 7 of the 9 selected cases tested in that batch; it missed cases 41 and 76. Cases 94 and 100 have no Allclose result. See the [protocol and saved results](traces_allclose/README.md). The model columns retain the original trials.

## Repeat results

The table below counts correct decisions in **repeat trials only**, without the initial trial, `r1`. Cases 74, 76, 78, 86, and 90 have five repeat trials, `r2` through `r6`, per method. Cases 94 and 100 have two repeat trials, `r2` and `r3`, per method. The kernel code and problem description stayed the same.

| Case | Single call, no tools: correct | Solo + tools: correct | Debate + tools: correct |
|---|---|---|---|
| [74](success_cases/case_74.md) | 2/5 | 5/5 | 5/5 |
| [76](success_cases/case_76.md) | 2/5 | 5/5 | 4/5 |
| [78](success_cases/case_78.md) | 0/5 | 5/5 | 5/5 |
| [86](success_cases/case_86.md) | 2/5 | 5/5 | 4/5 |
| [90](success_cases/case_90.md) | 2/5 | 5/5 | 5/5 |
| [94](success_cases/case_94.md) | 1/2 | 2/2 | 2/2 |
| [100](success_cases/case_100.md) | 0/2 | 2/2 | 2/2 |

**74:** Tools found failures in all five more trials. But the model without tools also returned the correct decision in two trials.

**76:** Solo returned the correct decision in all five more trials. Debate returned four correct decisions and one response with insufficient evidence. Errors in its test code prevented a full test near the midpoint. The model without tools returned the correct decision in two trials.

**78:** Each trial from `r2` through `r6` gave the same result. The model without tools accepted the incorrect kernel. Solo and Debate correctly rejected it. Each correct decision from a method with tools included an actual GPU failure on a permitted input.

**86:** Solo returned the correct decision in all five more trials. Debate returned four correct decisions. In `r4`, two early role calls reached their output limits. The run entered its final-response stage before it ran a GPU test and returned insufficient evidence. This was not an API or GPU outage. The model without tools returned the correct decision in two trials.

**90:** Solo and Debate returned the correct decision in all five more trials. Each included an actual GPU failure on a permitted input. The model without tools returned the correct decision in two trials. Some correct responses without tools used numerical examples outside the permitted row lengths. Their correct decisions remain counted, but those examples are not valid test evidence.

**94:** Solo and Debate found a permitted input that failed in all three trials. Including the initial trial, the correct counts are **1/3 without tools, 3/3 for Solo, and 3/3 for Debate**. The model without tools accepted the kernel in `r1` and `r2`, but correctly rejected it in `r3`. This case shows a benefit from tools in two trials, not a stable difference in every trial.

**100:** The model without tools accepted the incorrect kernel in all three trials. Solo and Debate each found an actual GPU failure on a permitted input in all three trials. Including `r1`, the correct counts are **0/3 without tools, 3/3 for Solo, and 3/3 for Debate**. The difference repeated in these three trials. This small sample does not establish how often it will occur in future trials, and it shows no advantage of Debate over Solo.

For case 78, both workflows with tools were correct and the single call without tools was wrong in all five repeat trials. Cases 74, 76, 86, and 90 show benefits from workflows with tools, but this difference does not occur in every trial. Cases 86 and 90 did not meet the planned requirement for a difference between tools and no tools in all five repeat trials.

These results compare complete workflows, so they do not separate tool access from additional reasoning. **They do not establish a stable difference between Solo and Debate or show that Debate is better than Solo.**

Full trial records: [74](real_kernel_challenges/private_data/source_atan_pair.json) · [76](real_kernel_challenges/private_data/source_rounding_pair.json) · [78](real_kernel_challenges/private_data/source_ntt_pair.json) · [86](real_kernel_challenges/private_data/source_affine_scan_pair.json) · [90](real_kernel_challenges/private_data/source_realfft_pair.json) · [94](real_kernel_challenges/private_data/experiment_20261020.json) · [100](real_kernel_challenges/private_data/experiment_20261020_followup.json).
