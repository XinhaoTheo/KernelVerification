# Success Cases

Updated: October 19, 2026. Model: GLM-5p3 via Fireworks.

This file describes cases where tools helped the model correct an incorrect decision. A correct decision can be **Accept** or **Reject**.

Each kernel has **one fixed Accept or Reject label for its full problem description**. If a kernel fails on an input that the problem description permits, the kernel is incorrect.

We ran all tests on an NVIDIA T4 with PyTorch 2.8.0 and Triton 3.4.0.

Case numbers were updated on October 19, 2026. [CASE_INDEX.md](CASE_INDEX.md) gives the previous numbers; raw traces retain their original text.

Select a case number for the detailed explanation and traces. The table shows the initial trials. For cases 74, 76, 78, 86, and 90, it shows the first trial, `r1`. A separate table gives the results from five more trials.

| Case | Kernel | Ground truth | Allclose | No tools | Solo + tools | Debate + tools |
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

Allclose uses the complete October 19 `r2` batch: existing original checks and one public default input per new kernel. It returned correct decisions for 7 of these 9 selected cases; it missed cases 41 and 76. See the [protocol and saved results](traces_allclose/README.md). The model columns retain the original trials.

## Repeat results

Each case in this table has one initial trial, `r1`, and five more trials, `r2` through `r6`, for each method. The table below counts correct decisions in the **five more trials only**. The kernel code and problem description stayed the same.

| Case | No tools: correct | Solo + tools: correct | Debate + tools: correct |
|---|---|---|---|
| [74](success_cases/case_74.md) | 2/5 | 5/5 | 5/5 |
| [76](success_cases/case_76.md) | 2/5 | 5/5 | 4/5 |
| [78](success_cases/case_78.md) | 0/5 | 5/5 | 5/5 |
| [86](success_cases/case_86.md) | 2/5 | 5/5 | 4/5 |
| [90](success_cases/case_90.md) | 2/5 | 5/5 | 5/5 |

**74:** Tools found failures in all five more trials. But the model without tools also returned the correct decision in two trials.

**76:** Solo returned the correct decision in all five more trials. Debate returned four correct decisions and one response with insufficient evidence. Errors in its test code prevented a full test near the midpoint. The model without tools returned the correct decision in two trials.

**78:** Each trial from `r2` through `r6` gave the same result. The model without tools accepted the incorrect kernel. Solo and Debate correctly rejected it. Each correct decision from a method with tools included an actual GPU failure on a permitted input.

**86:** Solo returned the correct decision in all five more trials. Debate returned four correct decisions. In `r4`, two early role calls reached their output limits. The run entered its final-response stage before it ran a GPU test and returned insufficient evidence. This was not an API or GPU outage. The model without tools returned the correct decision in two trials.

**90:** Solo and Debate returned the correct decision in all five more trials. Each included an actual GPU failure on a permitted input. The model without tools returned the correct decision in two trials. Some correct responses without tools used numerical examples outside the permitted row lengths. Their correct decisions remain counted, but those examples are not valid test evidence.

These results show a consistent difference for case 78 across the five more trials. Cases 74, 76, 86, and 90 show benefits from tools, but the difference does not occur in every trial. Cases 86 and 90 did not meet the planned requirement for separation in all five repeat trials. These results compare complete workflows, so they do not separate tool access from additional reasoning. These selected cases do not show that Debate is better than Solo.

Full trial records: [74](real_kernel_challenges/private_data/source_atan_pair.json) · [76](real_kernel_challenges/private_data/source_rounding_pair.json) · [78](real_kernel_challenges/private_data/source_ntt_pair.json) · [86](real_kernel_challenges/private_data/source_affine_scan_pair.json) · [90](real_kernel_challenges/private_data/source_realfft_pair.json).
