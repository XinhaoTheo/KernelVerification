# 24-case numerical pilot: single-call results

Model: `claude-opus-5`. One request per case, no tools or debate, 8192 maximum output tokens, no retries or token escalation.

## Results

| Family | Cases | Correct | Wrong verdict | Needs evidence | No answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention | 8 | 7 | 0 | 0 | 1 |
| quantization | 8 | 8 | 0 | 0 | 0 |
| recurrence | 8 | 3 | 0 | 0 | 5 |

Overall: **18/24 correct**, 0 incorrect verdicts, 0 abstentions, 6 missing answers.

Among 18 determinate trust/reject answers, 18 are correct. This conditional rate must be read alongside answer coverage, not substituted for overall accuracy.

Of the missing answers, 6 hit the output-token cap. These are budget exhaustions, not incorrect semantic judgments. The original baseline can escalate its token cap; this cost-bounded pilot deliberately does not, so raw accuracy is not a controlled head-to-head comparison to the old run.

Recorded API tokens: 52860 input, 149332 output. Estimated API cost: **$3.9976**, using the existing baseline's $5/$25 per million token assumption. Excludes Modal charges; not an invoice.

## Ground-truth checks

All 24 selected cases were run three times on NVIDIA T4. Each family has four passing and four failing workloads. Selected errors are <=0.75x or >=1.25x their predeclared family budget. CPU FP64 references were cross-checked against a second computation. No label depends on the model answer.

## Case details

Error/budget below 1 means compliant. Confidence is recorded, not used in scoring.

| Case | Family | Error/budget | Truth | Verdict | Confidence | Correct |
| --- | --- | ---: | --- | --- | ---: | --- |
| case_01 | attention | 1.5762 | reject | reject | 0.79 | True |
| case_02 | recurrence | 1.3399 | reject | no_verdict | — | False |
| case_03 | attention | 0.3220 | trust | trust | 0.72 | True |
| case_04 | quantization | 1.3525 | reject | reject | 0.76 | True |
| case_05 | quantization | 0.6604 | trust | trust | 0.85 | True |
| case_06 | recurrence | 1.6539 | reject | no_verdict | — | False |
| case_07 | attention | 2.0608 | reject | reject | 0.85 | True |
| case_08 | recurrence | 0.6621 | trust | no_verdict | — | False |
| case_09 | recurrence | 1.2510 | reject | no_verdict | — | False |
| case_10 | quantization | 1.4479 | reject | reject | 0.77 | True |
| case_11 | recurrence | 2.2760 | reject | no_verdict | — | False |
| case_12 | quantization | 1.4183 | reject | reject | 0.75 | True |
| case_13 | attention | 1.9033 | reject | reject | 0.85 | True |
| case_14 | quantization | 1.2849 | reject | reject | 0.8 | True |
| case_15 | attention | 1.4552 | reject | no_verdict | — | False |
| case_16 | recurrence | 0.4156 | trust | trust | 0.76 | True |
| case_17 | recurrence | 0.3757 | trust | trust | 0.72 | True |
| case_18 | attention | 0.4127 | trust | trust | 0.76 | True |
| case_19 | quantization | 0.7238 | trust | trust | 0.82 | True |
| case_20 | recurrence | 0.4200 | trust | trust | 0.76 | True |
| case_21 | attention | 0.5821 | trust | trust | 0.7 | True |
| case_22 | attention | 0.3845 | trust | trust | 0.72 | True |
| case_23 | quantization | 0.7234 | trust | trust | 0.75 | True |
| case_24 | quantization | 0.7101 | trust | trust | 0.72 | True |

## Model explanations

### case_01

The kernel deliberately rounds the softmax probabilities to float16 before the weighted sum of V, injecting ~2e-4 RMS relative perturbation into each p_i; since make_inputs shifts V by 0.9*(p@V), the true output is only 0.1x the uncancelled magnitude, amplifying that perturbation ~10x, so E ≈ 2e-3 (the ||p||_2 factor cancels between error and signal), roughly double the 1e-3 budget.

### case_02

No parseable final output

### case_03

The kernel computes the softmax-attention correctly; the only accuracy loss is the deliberate round-trip of the probabilities through float16 (rms relative error ~2e-4 per weight). Since the reference output equals 0.5*(p@v_orig) due to the center=0.5 cancellation, the sqrt(sum p_i^2) factor cancels in the error ratio, giving E ≈ 2 x 2e-4 ≈ 4-6e-4, within the 1e-3 budget, with all values finite.

### case_04

The kernel performs symmetric per-row 4-bit quantization (scale = max|w|/7) of the weights, but the workload's x is deliberately aligned (residual=0.5) with the summed quantization residual direction. Estimating: residual std ≈ s/√12 ≈ 0.12 with s ≈ 0.41, giving per-row error variance ≈ 0.12² (random) + 0.12² bias/fluctuation ≈ 0.032 versus reference variance ≈ ||x||² ≈ 1.25, so E ≈ √(0.032/1.25) ≈ 0.16, exceeding the 0.12 budget (pure quantization without the residual alignment would sit right at ~0.12).

### case_05

The kernel computes an int4 (max/7, round-to-nearest, clamped) quantized matvec; for 32x128 standard-normal weights the per-weight quantization error std is ~0.12 while output magnitudes are ~5-6, giving a relative L2 error around 0.015, well under the 0.12 budget, and the indexing/reduction logic (K=128 power of two, per-row program) is correct with finite outputs.

### case_06

No parseable final output

### case_07

The kernel deliberately rounds the softmax probabilities to float16 (u=2^-11) before the P@V contraction. Because the inputs are constructed with center=0.9, the reference output is the residual 0.1*(p@v_orig), so the relative error is amplified ~10x: E \u2248 (u/sqrt(3))/0.1 \u2248 2.8e-3 (this ratio is independent of the softmax distribution since both error and reference scale with sqrt(sum p_i^2)*sqrt(d)), well above the 1e-3 budget.

### case_08

No parseable final output

### case_09

No parseable final output

### case_10

The kernel applies plain per-row 4-bit (levels ±7) round-to-nearest quantization of W with no error compensation, while make_inputs deliberately injects the normalized quantization-residual direction into x. Estimating the resulting error: per-row error ≈ σ_q·sqrt(k/m) vs reference row mean ≈ 0.75·sqrt(k/m) with σ_q = (max|w|/7)/sqrt(12) ≈ 0.128, giving E ≈ ||e||/||y|| ≈ 0.40/2.47 ≈ 0.16, comfortably above the 0.12 budget.

### case_11

No parseable final output

### case_12

The kernel dequantizes each row to INT4 with max/7 scaling (per-element error std ≈ Δ/√12 ≈ 0.127σ), and make_inputs deliberately adds a unit-norm component along the summed dequantization residual, so the error vector has ‖Ex‖ ≈ 2.2 while the reference ‖Wx‖ ≈ 14, giving E ≈ 0.15–0.17, comfortably above the 0.12 budget.

### case_13

The kernel rounds the softmax probabilities to float16 before the P@V contraction, injecting ~2.8e-4 RMS relative perturbation per weight; since make_inputs centers V (center=0.9) so the reference output is only 0.1*(p@V), this cancellation amplifies the relative error ~10x, giving E ≈ 2.8e-3, well above the 1e-3 budget.

### case_14

The kernel replaces the exact matvec with per-row symmetric 4-bit (
tl.floor(w/scale+0.5), clipped to ±7) quantized weights; for this workload the quantization error per element has std ≈ scale/√12 ≈ 0.12 relative to unit-variance weights, which alone puts E near the 0.12 limit, and make_inputs deliberately adds 0.5× the unit vector along the summed quantization residual (using the same rounding rule), biasing each row's error by ≈0.5·||e_i||/√m ≈ 0.12 on top of the random part. Estimated ||Δ||≈0.94 vs ||ref||≈5.9 gives E ≈ 0.16, comfortably above the 0.12 budget.

### case_15

No parseable final output

### case_16

The kernel implements the recurrence correctly (h init 0, h=a*h+b, all states stored, correct indexing for T=896,D=32); the only deviation is rounding the running state to fp16 each step. With h≈1.0 (ulp≈5e-4–1e-3), per-step rounding noise sd≈2.2e-4, and the a=0.99 recurrence amplifying independent noise by 1/sqrt(1-0.99^2)≈7.1, giving error RMS≈1.6e-3 against reference RMS≈0.91, so E≈0.0017 (worst-case ~0.0022), within the 0.003 budget.

### case_17

The kernel computes the scan correctly (indexing, order, all states stored); its only deviation is rounding the carried state to fp16 each step. With a=0.99 the per-step rounding noise (relative std ~2.2e-4 near |h|~1) accumulates with amplification 1/sqrt(1-a^2)≈7.1, giving E≈1.5e-3, below the 3e-3 budget, and all outputs remain finite.

### case_18

The kernel computes the correct math; the only deliberate precision loss is rounding the softmax probabilities to fp16 (relative rms error ~2e-4). Since the output norm is only reduced by a factor 2 by the centering (center=0.5), the relative error E ≈ 2×2e-4 ≈ 4e-4, comfortably under the 1e-3 budget, with fp32 accumulation errors negligible.

### case_19

The kernel implements per-row symmetric 4-bit (\u00b17) quantized matvec matching the scheme the adversarial input was built around; estimating the residual-aligned error gives ||err||\u22481.1 versus ||y_ref||\u224816, i.e. E\u22480.07, comfortably under the 0.12 budget, and all values are finite with correct indexing/reduction.

### case_20

The kernel implements the recurrence correctly (h init 0, sequential scan over T, contiguous indexing, D=32 power of two); the only accuracy concern is the deliberate fp16 rounding of each state. With h≈1 (ulp≈4.9e-4–9.8e-4), per-step rounding noise σ≈1.4–2.2e-4 is amplified by the AR(1) pole 0.99 (variance gain ≈50), giving accumulated RMS error ≈1.1–1.6e-3 against reference RMS ≈0.9, i.e. E≈1.2–1.8e-3 < 3e-3. Consecutive states move ~2 ulps, so rounding errors are effectively decorrelated and no large DC bias amplification occurs.

### case_21

The only meaningful error source is rounding the softmax probabilities to fp16 (RMS relative error ~2e-4); the centering with center=0.5 halves the reference norm, so E ≈ 2*2e-4 ≈ 4-6e-4, and even in the worst (single-token-dominated) case the error is bounded near 2*2^-11 ≈ 9.8e-4, still within the 1e-3 budget. Everything else (fp32 dot products, exp, final accumulation) contributes ~1e-7 level error and all outputs are finite.

### case_22

The only meaningful error source is the deliberate float16 round-trip of the normalized probabilities (relative RMS ~2e-4); with the mild center=0.5 cancellation (output = 0.5*p@v) the relative error is roughly 2x that, ~4e-4, comfortably under the 1e-3 budget, and softmax weights are large enough that fp16 subnormal/flush effects are negligible. Kernel logic (max-subtraction softmax, fp32 accumulation, constexpr N=128/D=32 powers of two) is otherwise correct and produces finite outputs.

### case_23

The kernel correctly implements per-row symmetric int4 quantization (matching the scale/rounding/clipping used to build the adversarial residual direction) and indexes/accumulates correctly; estimating the resulting error, the systematic residual-aligned bias gives ||err||\u22481.3 versus ||ref||\u224815.5, i.e. E\u22480.08, comfortably within the 0.12 budget, with all outputs finite.

### case_24

The kernel implements per-row symmetric 4-bit (levels -7..7) quantization with round-half-up, matching the scheme the adversarial input targets; estimating the resulting error for this fixed workload gives ||err||u2248 0.95 versus ||ref||u2248 11.7, i.e. E u2248 0.08, comfortably under the 0.12 budget, and all values are finite with correct shapes/indexing (K=128 power of two).

## Limits

This is an exploratory finite-workload accuracy test, not universal kernel verification. The 24 cases are correlated variants of three kernels. Budgets are benchmark design choices, not externally established production standards. Deterministic seed arrays are fully specified in source, but a no-tool model must mentally reconstruct them, so computation access is a major confound. Abstention is not a confident error. A correct fixed numerical test can solve these cases. No result here establishes a multi-agent advantage; that requires comparing against a single agent with the same tools and matched budget.

Construction history: first GPU launch had a module-import error and was stopped. The first successful candidate sweep did not supply enough failing quantization/recurrence cases. The budgets were kept fixed; residual-aligned quantization inputs and a lower-noise recurrence configuration were added before any LLM calls. Both candidate sweep records are retained.
