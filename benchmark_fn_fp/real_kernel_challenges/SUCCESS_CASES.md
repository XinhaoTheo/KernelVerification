# Observed Tool-Assisted Verification Gains

Updated: October 11, 2026. Model: GLM-5p3 via Fireworks.

These four cases show at least one tool-assisted workflow correcting a wrong
verdict from a single model call without tools. Each implementation has one fixed
accept/reject label over its full public contract; inputs only provide evidence.

| Case | Kernel | Ground truth | No tools | Solo + tools | Debate + tools |
|---|---|---|---|---|---|
| 109 | RMSNorm backward | Reject | Accept | Reject | No verdict: token limit |
| 114 | Paged sliding-window attention | Reject | Accept | Reject | Reject |
| 127 | erf approximation | Reject | Accept | Reject | Reject |
| 128 | GELU approximation | Accept | Reject | Accept | Accept |

**109 — Precision loss before FP32 accumulation.** The contract requires FP16
rounding for certain per-element products, but the block path also reduces those terms in FP16 before
adding to an FP32 buffer. The no-tool model incorrectly treats the reduction as
FP32. Solo finds a legal `[512, 128]` input whose weight-gradient error is
**8.94 times the allowed tolerance**. Storing the result in FP32 cannot recover
precision already lost.

**114 — Zero attention weights do not suppress NaN values.** The contract allows
NaNs in unreachable cache slots. The kernel masks their scores but still feeds
their values into the weighted sum, where `0 * NaN` contaminates the output.
Debate reproduces a legal case with **64/64 NaN outputs and a fully finite
reference**. Both tool workflows reject; the no-tool model overlooks this path.
Debate identified the mechanism before execution, so its probe confirms the
diagnosis rather than discovering it.

**127 — Modified coefficients invalidate the assumed error guarantee.** The erf
approximation uses coefficients rounded to five decimal places. The no-tool
model accepts it using the original formula's error bound. GPU evaluation finds
violations; an independent high-precision oracle confirms an error of
**1.29175e-6 at `x = 0.50830078125`**, above the **1e-6** tolerance.

**128 — A loose error bound causes a false rejection.** The no-tool model
multiplies a global approximation-error bound by the largest input, ignoring
that the error decays in the tails. Both tool workflows accept after numerical
checks. Independent qualification over all **63,488 finite FP16 encodings**
gives a maximum error bound of **5.26772e-7**, below **1e-6**, with additional
shape, offset and storage checks. A loose upper bound is not a counterexample.

These selected, purpose-built cases have one model run per case and method,
excluding infrastructure-only startup retries. They do not establish a general
or stable advantage. They show workflow gains, without
isolating tools from additional reasoning. Stable gains from debate over solo
remain unproven.
