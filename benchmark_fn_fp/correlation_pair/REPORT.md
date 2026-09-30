# Correlation-pair experiment

Case labels and links use the current global numbering. Verbatim historical model explanations below retain their original wording and may mention previous IDs; see the global case map.

Frozen before model evaluation; every trial below is retained. CPU construction and GPU verification are separate from the answer-free artifacts given to the models.

## GPU labels

| Case | Relative L2 error | Budget | Label |
| --- | ---: | ---: | --- |
| case_36 | 0.2241158064 | 0.1 | reject |
| case_37 | 0.0336471468 | 0.1 | trust |

Each case ran ten times on T4; input hashes match CPU construction and two FP64 reference implementations agree. GPU environment: {"gpu": "Tesla T4", "torch": "2.8.0+cu128", "triton": "3.4.0", "numpy": "1.26.4"}

## All trials

| Case | Arm | Trial | Verdict | Correct | Confidence | API estimate |
| --- | --- | --- | --- | --- | ---: | ---: |
| case_36 | single_call | neutral1 | reject | True | 0.85 | $0.0715 |
| case_36 | single_call | r1 | trust | False | 0.7 | $0.2020 |
| case_36 | single_call | r2 | trust | False | 0.6 | $0.2122 |
| case_36 | debate | r1 | reject | True | 0.95 | $1.5342 |
| case_37 | single_call | neutral1 | reject | False | 0.86 | $0.0812 |
| case_37 | single_call | r1 | reject | False | 0.84 | $0.1007 |
| case_37 | single_call | r2 | reject | False | 0.83 | $0.0776 |
| case_37 | solo | r1 | trust | True | 0.9 | $0.5002 |
| case_37 | debate | r1 | trust | True | 0.9 | $1.1241 |

Recorded API estimate: **$3.9037**. Uses the existing project model profile, not a billing invoice; Modal charges are excluded.

## Trial evidence

### case_36 / single_call / neutral1

The kernel replaces the exact dot products with per-row INT4 (absmax/7) quantization of both A and B. With ~N(0,1) rows (absmax≈2.9, step s≈0.41), per-element quantization error std ≈ s/sqrt(12)≈0.12, and since ||x||=1 each row's dot-product error std ≈0.12 per matrix (≈0.17 combined), while the reference row values are ≈1.0 by construction. This gives E ≈ 0.17 >> 0.1, and the 64-row averaging makes a lucky pass essentially impossible.

Stop reason: end_turn; output tokens: 2482.

Prompt ablation: removed only the explanatory sentence allowing either branch to exceed 0.1 individually; final-output contract, threshold, source and inputs are unchanged.

### case_36 / single_call / r1

The kernel implements a self-consistent per-row symmetric INT4 quantization (scale=absmax/7, round-to-nearest, clamp [-7,7], correct dequant and indexing) for both A and B and sums the two branches; no structural bug (indexing, masking, axes, dtypes) is present. Per-branch quantization error (~0.15-0.2 relative) exceeds 0.1 alone, but the hard-coded row permutation of B is evidently an offline-computed error-cancelling matching (as hinted by the statement that either branch may individually exceed 0.1), so the summed output should meet E<=0.1.

Stop reason: end_turn; output tokens: 7695.

### case_36 / single_call / r2

The kernel implements a canonical symmetric INT4 per-row quantization (scale = max|w|/7, round-half-to-nearest, clamp to ±7) of both A and B and sums both branches in fp32; the hardcoded offline-computed row permutation and the statement that either branch alone may exceed 0.1 indicate the workload was constructed so the two branches' quantization errors cancel under exactly this scheme, leaving final E well under 0.1. No indexing, masking, or reduction defects are present (K=128 is a power of two, all loads are in range).

Stop reason: end_turn; output tokens: 8103.

### case_36 / debate / r1

The operative contract (problem.txt) is a single fixed workload — run(*make_inputs()) with PCG64 SEED=921000 on a T4 — judged by exactly two conditions: all outputs finite, and E = ||out - ref||_2 / max(||ref||_2, 0.001*sqrt(64)) <= 0.1. Claim c1 is confirmed, in_scope, and its scope evidence cites precisely that stated domain (problem.txt metric/threshold plus kernel.py lines 25-42 fixing the seed, permutation and shapes). Direct measurement on a Tesla T4 (t8) gives E = 0.22412, 2.24x the threshold, with error norm 1.790 against ||ref||_2 = 7.988 and max absolute error 0.576 on a per-row reference of ~1.0. An independent NumPy fp64 emulation of the kernel's own quantization (absmax/7, floor(v/s+0.5), clamp ±7) reproduces E = 0.2241158 to seven digits, localizing the failure to the INT4 rounding noise itself rather than to indexing, the enable_fp_fusion launch option, or fp32 accumulation order. Clamp hits of 100/8192 (A) and 101/8192 (B) rule out clipping as the mechanism; per-branch dequant error norms 0.850 and 0.962 add in quadrature as predicted, because B is row-permuted and the two branches are independent. This is not a deviation the contract accounts for: the contract explicitly licenses approximate intermediates and explicitly allows either branch to individually exceed 0.1, but conditions compliance on the FINAL output satisfying E <= 0.1, which it does not; nor is it a near-zero-value relative-error artifact, since ||ref||_2 = 7.988 far exceeds the 0.008 denominator floor and the absolute error is ~58% of a unit-scale signal. Claim c2 (non-finite output / launch failure) was rebutted by t9: the kernel ran cleanly on Triton 3.4.0 returning 64 finite fp32 values with zero degenerate row maxima (min row absmax 2.117/2.127), so the finiteness half of the contract passes — but the E half fails decisively. No unresolved in-scope claim remains; the Skeptic reviewed the latest evidence and recorded no new claims.

Automated trace audit: []

### case_37 / single_call / neutral1

The kernel performs per-row INT4 (15-level, max-abs scale) quantization of A and B, but the inputs are deliberately constructed so each row's projection onto the unit vector x is only ~0.5 while row norms are ~sqrt(128)~11.3. Quantization step s≈max|a|/7≈0.43 gives per-row dot-product error std ≈ s/sqrt(12)≈0.12 per matrix (≈0.17 combined), versus reference entries ≈1.0, so E ≈ 0.17 > 0.1 and the contract is violated.

Stop reason: end_turn; output tokens: 2867.

Prompt ablation: removed only the explanatory sentence allowing either branch to exceed 0.1 individually; final-output contract, threshold, source and inputs are unchanged.

### case_37 / single_call / r1

The kernel applies per-row symmetric INT4 quantization (scale = max|row|/7 \u2248 0.41 for 128 standard-normal entries), giving per-element error std \u2248 s/\u221a12 \u2248 0.118 and hence per-row dot-product error std \u2248 0.118 per matrix (||x||=1), \u2248 0.167 combined for the two independent matrices; since each reference entry is \u2248 0.5+0.5 = 1.0 and ||ref|| \u2248 8, E \u2248 1.34/8 \u2248 0.17, well above the 0.1 tolerance (reaching E\u22640.1 would require max|row| \u2248 1.7, impossible here), so the contract is violated.

Stop reason: end_turn; output tokens: 3645.

### case_37 / single_call / r2

The kernel quantizes each row of A and B to 4-bit symmetric levels with scale max|row|/7 (\u2248 2.86/7 \u2248 0.41), giving per-element rounding noise of std \u2248 0.118; dotted with the unit-norm x and summed over two independent matrices, per-row error std \u2248 0.167, so ||err|| \u2248 0.167*8 \u2248 1.34 while ||ref|| \u2248 8 (each row's projection is engineered to \u22480.5+0.5=1.0), yielding E \u2248 0.17 > 0.1. The two branches use independent random matrices, so their quantization errors do not cancel.

Stop reason: end_turn; output tokens: 2722.

### case_37 / solo / r1

The contract is a finite-workload numerical compliance check: run(*make_inputs()) must produce finite outputs with E = ||out-ref||_2/max(||ref||_2, 0.001*sqrt(64)) <= 0.1, where ref = A@x + B@x in float64, and approximate intermediates are explicitly permitted. I raised c1 hypothesizing that the kernel's per-row INT4 quantization (scale = max|row|/7, round-half-up via floor(v/s+0.5), clamp to [-7,7]) would inject enough noise to exceed 0.1, since each row's true projection is deliberately constructed to be small (~0.5 per branch) relative to the ~0.4 quantization step.

Probe t6 ran the kernel on the GPU and evaluated the contract metric in float64: E = 0.033647, ||ref|| = 7.9876, ||err|| = 0.2688, all 64 outputs finite. The kernel output matched an independent NumPy per-row INT4 emulation to 2.67e-7 max abs, confirming the kernel really does implement the quantized path and that the observed error is quantization noise, not a coding bug. Probe t8 repeated the full pipeline 5x on the declared target hardware (Tesla T4): E = 0.0336471 bit-identically each run, all outputs finite. Error is ~3x below the allowed bound with no run-to-run variance.

So the only in-scope claim is rebutted by runtime evidence, and the lossy INT4 intermediate representation is precisely the deviation the contract pre-authorizes ("Approximate intermediate values are allowed if the FINAL output meets this requirement"; "not a requirement to reproduce INT4 arithmetic bit-for-bit"). Residual uncertainty: the margin depends on the fixed PCG64 seed/permutation workload, which is exactly what the contract scopes; no claim about general inputs is made or needed.

Automated trace audit: []

### case_37 / debate / r1

The contract (problem.txt, the sole operative spec here — no test.py exists) defines a finite-workload compliance check: run(*make_inputs()) on the fixed PCG64-seeded inputs, reference y = A@x + B@x in float64, metric E = ||out-ref||_2 / max(||ref||_2, 0.001*sqrt(64)), requiring all outputs finite and E <= 0.1, with approximate intermediates explicitly allowed and either branch permitted to individually exceed 0.1.

Evidence measures exactly that metric on the exact mandated workload. c1 (t8): on a Tesla T4, E = 0.033647 vs the 0.1 bound (~3x margin), all 64 outputs finite; confirmatory repeat (t12) shows E identical across 5 launches (E_spread = 0.0, bitwise-identical outputs, margin factor 2.97), so the pass is not a nondeterminism artifact. c1 is therefore rebutted, not merely unproven.

c2 (t9) is also rebutted, and its evidence closes the main way c1 could have been a lucky metric artifact: a CPU emulation of the kernel's exact quantizer (s = rowmax/7, floor(W/s+0.5), clamp ±7) reproduces the GPU output to 2.7e-7 max abs diff, confirming the decomposition is faithful, and shows errA = 0.850 (EA_only = 0.106), errB = 0.962 (EB_only = 0.120), combined only 0.269 — far below the quadrature prediction 1.283, with branch-error correlation -0.964. Each branch alone exceeding 0.1 is exactly the case the contract expressly says does not violate it; only the summed final output is judged, and the summed output passes with wide margin. The same probe shows 0 codes outside [-7,7] before clamping, so the clamp is inert, and no zero-magnitude row exists, so the sa/sb division is safe on this workload.

The residual deviations present are precisely those the contract accounts for: INT4-style quantization of intermediates ("Approximate intermediate values are allowed if the FINAL output meets this requirement"; "not a requirement to reproduce INT4 arithmetic bit-for-bit") and ~1e-7-level fp32 reduction-order effects from enable_fp_fusion=False. No unresolved claim remains; both ledger claims are settled as rebutted and the Skeptic's sign-off (t13) found no further in-scope hypothesis. Confidence is held at 0.9 rather than higher only because the debate explored a narrow claim set, but the decisive measurement is the contract's own metric applied to the contract's own fixed input.

Automated trace audit: []

## Interpretation limits

This is a selected synthetic pair testing finite-workload numerical compliance. It is not a proof over all valid inputs, not a representative dataset, and not a claim that the kernel is an upstream bug. With the correct reference and metric supplied, a fixed numerical script solves it. A tool advantage does not imply a debate advantage over a solo agent with tools. A source-only abstention is legitimate uncertainty; a token-cap exhaustion is not a wrong semantic judgment. Repeated calls on two frozen cases measure repeatability, not performance on independent held-out cases.
