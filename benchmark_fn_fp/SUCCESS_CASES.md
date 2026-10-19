# Success Cases

Updated: October 19, 2026. Model: GLM-5p3 via Fireworks.

This file describes cases where tools helped the model correct an incorrect decision. A correct decision can be **Accept** or **Reject**.

Each kernel has **one fixed Accept or Reject label for its full problem description**. If a kernel fails on an input that the problem description permits, the kernel is incorrect.

We ran all tests on an NVIDIA T4 with PyTorch 2.8.0 and Triton 3.4.0.

Select a case number for the detailed explanation and traces. The table shows the initial trials. For cases 147, 149, 151, 159, and 163, it shows the first trial, `r1`. A separate table gives the results from five more trials.

| Case | Kernel | Ground truth | No tools | Solo + tools | Debate + tools |
|---|---|---|---|---|---|
| [109](success_cases/case_109.md) | RMSNorm backward | Reject | Accept | Reject | No verdict: token limit |
| [114](success_cases/case_114.md) | Paged sliding-window attention | Reject | Accept | Reject | Reject |
| [127](success_cases/case_127.md) | erf approximation | Reject | Accept | Reject | Reject |
| [128](success_cases/case_128.md) | GELU approximation | Accept | Reject | Accept | Accept |
| [147](success_cases/case_147.md) | Atan approximation | Reject | Accept | Reject | Reject |
| [149](success_cases/case_149.md) | Sigmoid rounding | Reject | Accept | Reject | Reject |
| [151](success_cases/case_151.md) | NTT coefficient conversion | Reject | Accept | Reject | Reject |
| [159](success_cases/case_159.md) | Recurrence order | Reject | Accept | Reject | Reject |
| [163](success_cases/case_163.md) | FFT address | Reject | Accept | Reject | Reject |

## Repeat results

Each case in this table has one initial trial, `r1`, and five more trials, `r2` through `r6`, for each method. The table below counts correct decisions in the **five more trials only**. The kernel code and problem description stayed the same.

| Case | No tools: correct | Solo + tools: correct | Debate + tools: correct |
|---|---|---|---|
| [147](success_cases/case_147.md) | 2/5 | 5/5 | 5/5 |
| [149](success_cases/case_149.md) | 2/5 | 5/5 | 4/5 |
| [151](success_cases/case_151.md) | 0/5 | 5/5 | 5/5 |
| [159](success_cases/case_159.md) | 2/5 | 5/5 | 4/5 |
| [163](success_cases/case_163.md) | 2/5 | 5/5 | 5/5 |

**147:** Tools found failures in all five more trials. But the model without tools also returned the correct decision in two trials.

**149:** Solo returned the correct decision in all five more trials. Debate returned four correct decisions and one response with insufficient evidence. Errors in its test code prevented a full test near the midpoint. The model without tools returned the correct decision in two trials.

**151:** Each trial from `r2` through `r6` gave the same result. The model without tools accepted the incorrect kernel. Solo and Debate correctly rejected it. Each correct decision from a method with tools included an actual GPU failure on a permitted input.

**159:** Solo returned the correct decision in all five more trials. Debate returned four correct decisions. In `r4`, two early role calls reached their output limits. The run entered its final-response stage before it ran a GPU test and returned insufficient evidence. This was not an API or GPU outage. The model without tools returned the correct decision in two trials.

**163:** Solo and Debate returned the correct decision in all five more trials. Each included an actual GPU failure on a permitted input. The model without tools returned the correct decision in two trials. Some correct responses without tools used numerical examples outside the permitted row lengths. Their correct decisions remain counted, but those examples are not valid test evidence.

These results show a consistent difference for case 151 across the five more trials. Cases 147, 149, 159, and 163 show benefits from tools, but the difference does not occur in every trial. Cases 159 and 163 did not meet the planned requirement for separation in all five repeat trials. These results compare complete workflows, so they do not separate tool access from additional reasoning. These selected cases do not show that Debate is better than Solo.

Full trial records: [147](real_kernel_challenges/private_data/source_atan_pair.json) · [149](real_kernel_challenges/private_data/source_rounding_pair.json) · [151](real_kernel_challenges/private_data/source_ntt_pair.json) · [159](real_kernel_challenges/private_data/source_affine_scan_pair.json) · [163](real_kernel_challenges/private_data/source_realfft_pair.json).
