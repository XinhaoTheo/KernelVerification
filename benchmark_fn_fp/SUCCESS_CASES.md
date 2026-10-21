# Tools vs No Tools

Updated: October 20, 2026. Model: GLM-5p3 through Fireworks.

We selected these 11 cases to compare a single model call without tools with workflows that use tools. Solo and Debate both use tools. A correct decision can be **Accept** or **Reject**.

**These cases show differences between tools and no tools. The results do not show that Debate is better than Solo.** The table gives a different column for each workflow.

Each kernel has **one fixed Accept or Reject label for its full problem description**. If an input is permitted and the kernel gives an incorrect result for that input, the kernel is incorrect.

We did all GPU tests on an NVIDIA T4 with PyTorch 2.8.0 and Triton 3.4.0.

The table shows the 11 selected cases in order from the smallest current case ID to the largest. The [case index](CASE_INDEX.md), public kernel folders and traces use these same IDs. We added case 100 last.

Each case number is a link to the explanation and traces. The table shows the first trials. The second table gives results from five more trials for cases 74, 76, 78, 86, and 90. Cases 94 and 100 have two more trials.

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

Allclose is a numerical test method that does not use model calls. It uses the full October 19 `r2` batch. That batch uses the initial checks for the initial cases and one public default input for each new kernel. Allclose gave correct decisions for 7 of the 9 selected cases in that batch. It did not find the defects in cases 41 and 76. Cases 94 and 100 have no Allclose result.

The [protocol and saved results](traces_allclose/README.md) give the information about the Allclose tests. The model columns show the first trials.

## Repeat results

This table counts correct decisions in **more trials only**, without the first trial, `r1`. Cases 74, 76, 78, 86, and 90 have five more trials, `r2` through `r6`, for each method. Cases 94 and 100 have two more trials, `r2` and `r3`, for each method. The kernel code and problem description stayed the same.

| Case | Single call, no tools: correct | Solo + tools: correct | Debate + tools: correct |
|---|---|---|---|
| [74](success_cases/case_74.md) | 2/5 | 5/5 | 5/5 |
| [76](success_cases/case_76.md) | 2/5 | 5/5 | 4/5 |
| [78](success_cases/case_78.md) | 0/5 | 5/5 | 5/5 |
| [86](success_cases/case_86.md) | 2/5 | 5/5 | 4/5 |
| [90](success_cases/case_90.md) | 2/5 | 5/5 | 5/5 |
| [94](success_cases/case_94.md) | 1/2 | 2/2 | 2/2 |
| [100](success_cases/case_100.md) | 0/2 | 2/2 | 2/2 |

**74:** The workflows with tools found defects in all five trials. But the model without tools also gave the correct decision in two trials.

**76:** Solo gave the correct decision in all five trials. Debate gave four correct decisions and one response without sufficient test results. Errors in its test code prevented a full test near the midpoint. The model without tools gave the correct decision in two trials.

**78:** Each trial from `r2` through `r6` gave the same result. The model without tools accepted the incorrect kernel. Solo and Debate correctly rejected it. Each correct decision from a method with tools had a GPU test that showed an incorrect output for a permitted input.

**86:** Solo gave the correct decision in all five trials. Debate gave four correct decisions. In `r4`, two role calls used all permitted output tokens. The workflow gave its last response before a GPU test, without an Accept or Reject decision. The API and GPU were available. The model without tools gave the correct decision in two trials.

**90:** Solo and Debate gave the correct decision in all five trials. Each found an incorrect GPU output for a permitted input. The model without tools gave the correct decision in two trials. Some correct responses without tools used examples with row lengths outside the permitted range. We count those decisions as correct, but those examples do not show a defect for the permitted input domain.

**94:** Solo and Debate found an incorrect output for a permitted input in all three trials. With the first trial, the correct counts are **1/3 without tools, 3/3 for Solo, and 3/3 for Debate**. The model without tools accepted the kernel in `r1` and `r2`, but correctly rejected it in `r3`. Tools helped in two trials, but the difference did not occur in each trial.

**100:** The model without tools accepted the incorrect kernel in all three trials. Solo and Debate each found an incorrect GPU output for a permitted input in all three trials. With `r1`, the correct counts are **0/3 without tools, 3/3 for Solo, and 3/3 for Debate**. The same difference occurred in all three trials. This small sample does not show how frequently the difference will occur in future trials. It does not show that Debate is better than Solo.

For case 78, both workflows with tools were correct in all five trials. The single call without tools was incorrect in all five trials. For cases 74, 76, 86, and 90, tools helped, but this difference did not occur in each trial. We planned to find a difference in all five trials for cases 86 and 90. The results did not agree with that requirement.

We compared full workflows with different tool access and reasoning. Thus, these results do not measure the effect of tools alone. **They do not show that Debate is better than Solo.**

Full trial records: [74](real_kernel_challenges/private_data/source_atan_pair.json) · [76](real_kernel_challenges/private_data/source_rounding_pair.json) · [78](real_kernel_challenges/private_data/source_ntt_pair.json) · [86](real_kernel_challenges/private_data/source_affine_scan_pair.json) · [90](real_kernel_challenges/private_data/source_realfft_pair.json) · [94](real_kernel_challenges/private_data/experiment_20261020.json) · [100](real_kernel_challenges/private_data/experiment_20261020_followup.json).
