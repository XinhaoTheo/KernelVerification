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

| Case | Arm | Trial | Verdict | Outcome | Confidence | API estimate |
| --- | --- | --- | --- | --- | ---: | ---: |
| case_36 | single_call | r1 | reject | correct | 0.88 | $0.0140 |
| case_36 | solo | r1 | reject | correct | 0.98 | $0.0395 |
| case_36 | debate | r1 | reject | correct | 0.97 | $0.1276 |
| case_37 | single_call | r1 | no_verdict | token_limit | None | $0.0364 |
| case_37 | solo | r1 | trust | correct | 0.97 | $0.0559 |
| case_37 | debate | r1 | trust | correct | 0.95 | $0.1947 |

Recorded API estimate: **$0.4682**. Uses the existing project model profile, not a billing invoice; Modal charges are excluded.

## Trial evidence

### case_36 / single_call / r1

The kernel replaces each row of A and B with a ~4-bit absmax/7 symmetric quantization (step sa = max|row|/7 ≈ 0.4 for N(0,1) rows with max ≈ 2.9), giving per-entry round-to-nearest noise of std ≈ sa/√12 ≈ 0.12; projected onto the unit-norm x, each branch's dot product carries error std ≈ 0.12 against a per-branch reference value of only ~0.5 (targets are 0.5 ± 0.02), so the combined output has relative error E ≈ √2·0.12/1.0 ≈ 0.17 > 0.1. The outputs are finite and the code is otherwise well-formed (clamp is a no-op since |a|/sa ≤ 7, indexing/strides are correct), but the quantization is far too coarse to meet the stated 0.1 tolerance on this workload.

API calls with recorded usage: 1; input tokens: 1210; output tokens: 12438; probes: 0.

Stop reason: stop; output tokens: 12438.

### case_36 / solo / r1

Decisive evidence from runtime probe t6 (claim c1, evidence c1.e1) on the target Tesla T4 GPU: I imported the artifact kernel, generated the exact contract workload via make_inputs() (SEED=921000, PCG64; unit-norm float32 x of shape (128,), A and B float32 (64,128) contiguous), and ran run(x, a, b) on the real GPU. Outputs were all finite (finiteness requirement satisfied), but against the contract reference (A@x + B@x on the original float32 inputs, evaluated in float64; ref_norm = 7.988, so the metric denominator is the reference norm since 7.988 >> 0.001*sqrt(64) = 0.008), the contract metric came out E = 0.2241158, more than double the required threshold of E <= 0.1 — a direct violation of the contract's binding requirement that the FINAL output satisfy E <= 0.1. To rule out a Triton arithmetic fault or probe artifact, the probe also simulated the kernel's per-row 15-level quantization (scale = max|row|/7, round-half-up, clamp to [-7,7]) in float64: the simulation matched the GPU output to 3.0e-07 (kernel_vs_sim_max_abs), and the run was deterministic across invocations. This proves the E > 0.1 failure is inherent to the kernel's quantization design (per-branch relative errors 0.211 and 0.242 combining into 0.224), not an execution error. The contract's exemptions do not rescue it: approximate intermediate values are allowed only "if the FINAL output meets this requirement", and the per-branch tolerance ("either branch may individually exceed 0.1") cannot apply to the combined final output, which is what the metric measures; nor is bit-exact INT4 arithmetic required — the metric itself is what fails. The claim is scoped in_scope with contract quotes from problem.txt, so the verdict rests on a confirmed in-scope contract violation on exactly the specified finite workload.

API calls with recorded usage: 4; input tokens: 55989; output tokens: 21631; probes: 1.

Automated trace audit: []

### case_36 / debate / r1

Reject: run(*make_inputs()) fails the stated compliance contract on the exact required workload. c1 is confirmed by direct runtime measurement on the contract-mandated inputs (make_inputs(), NumPy PCG64 seed 921000, row-permuted B, executed on a Tesla T4 via cuda:0) using exactly the metric problem.txt defines: E = ||out - ref||_2 / max(||ref||_2, 0.001*sqrt(64)) with ref = float64(A@x + B@x) on the original float32 inputs. Measured E = 0.2241 vs the contracted limit of 0.1 (err_l2 = 1.790 against the ~0.799 budget, a >2x violation); all 64 outputs are finite, so the finiteness clause passes but the accuracy clause fails. There is no test.py/get_inputs in this artifact, so problem.txt is the operative contract, and its explicit requirement "all outputs must be finite and E <= 0.1" on exactly this fixed workload is violated. c2 confirms the mechanism and attribution: a float64 emulation of the kernel's exact quantization scheme (per-row s = max|row|/7, clip(floor(v/s+0.5),-7,7), dequant code*s, both branches summed) on the same fixed inputs yields E = 0.2241, and the Triton kernel agrees with that emulation to 2.1e-8 in E (max elementwise gap 3e-7), so there is no additional indexing/scale/accumulation defect — the violation is inherent to the kernel's INT4-style quantization noise. This does not make the deviation acceptable: the contract's clause "Approximate intermediate values are allowed if the FINAL output meets this requirement" conditions intermediate approximation on final-output compliance, which fails; "not a requirement to reproduce INT4 arithmetic bit-for-bit" disclaims bit-exact INT4 reproduction, not the E <= 0.1 numeric requirement; and "Either branch may individually exceed 0.1" does not apply because the judged combined final output is the quantity that exceeds 0.1. The reference is finite (||ref||_2 = 7.988, ref rows ~1.0 +/- 0.03), so there is no reference-undefined ambiguity, and the failure margin is far beyond any fp32/fp64, tie-breaking, or run-to-run metric noise (kernel vs ideal-arithmetic emulation gap 1e-8, zero fp32-vs-fp64 code flips). The one unverified open question — whether the judging harness computes the reference exactly as problem.txt states — is immaterial at this margin and is set aside. No open claims remain, and the Skeptic recorded no_new_claims after reviewing the final evidence (t8, t9). Confidence 0.97: the decisive evidence is a direct measurement of the exact contract metric on the exact contract workload, reproduced by an independent ideal-arithmetic emulation.

API calls with recorded usage: 6; input tokens: 193910; output tokens: 66670; probes: 2.

Automated trace audit: []

### case_37 / single_call / r1

Missing or invalid JSON verdict

API calls with recorded usage: 1; input tokens: 1210; output tokens: 32768; probes: 0.

Stop reason: length; output tokens: 32768.

### case_37 / solo / r1

The contract is finite-workload compliance: run(*make_inputs()) (PCG64 seed 921000, unit-norm x of shape (128,), A/B of shape (64,128) float32, given row permutation) must produce all-finite outputs with E = ||out - (A@x + B@x)_fp64||_2 / max(||ref||_2, 0.001*sqrt(64)) <= 0.1. Probe t6 ran the exact workload on the GPU: numerator L2 = 0.26876, denominator = 7.9876, E = 0.0336 <= 0.1 (~3x margin), all 64 float32 outputs finite, out shape (64,); the GPU output matched a float64 emulation of the kernel's per-row +/-7 quantization algorithm to 2.7e-7 max abs, confirming the Triton kernel executed as written with no silent fallback. Probe t8 independently reconfirmed through an artifact-resistant path: reference computed before kernel invocation via torch fp64 AND a separate numpy fp64 matmul (references agree to 4.4e-16) both give E = 0.0336; a second run is bit-identical (deterministic); run() does not mutate inputs. Per-branch quantization error individually exceeds the bound (branch A: 0.211, branch B: 0.242 relative) but cancels in the combined output (branch error norms 0.850/0.962, combined 0.269, correlation -0.963) — precisely what the contract explicitly permits ("Approximate intermediate values are allowed if the FINAL output meets this requirement. Either branch may individually exceed 0.1 without violating the contract."), and it explicitly waives bit-exact INT4 reproduction. The only failure hypothesis raised (c1: E > 0.1 on this workload) is rebutted by both runtime probes; no other contract requirement (float32 contiguous inputs, output finiteness, metric definition) is violated. Verdict: trust.

API calls with recorded usage: 4; input tokens: 60838; output tokens: 35331; probes: 2.

Automated trace audit: []

### case_37 / debate / r1

Trust. The operative contract is problem.txt (the artifact holds only kernel.py, meta.json, problem.txt — no test.py/get_inputs — so problem.txt plus kernel.py's make_inputs define the input domain): on exactly run(*make_inputs()) (NumPy PCG64 seed 921000), all 64 outputs must be finite and E = ||out - (A@x+B@x)_fp64||_2 / max(||ref||_2, 0.001*sqrt(64)) <= 0.1, with approximate intermediate values explicitly allowed, either branch individually allowed to exceed 0.1, and no bit-exact INT4 requirement. Direct measurement on that exact workload — inputs regenerated via make_inputs and bitwise-verified against an independent NumPy PCG64 re-derivation, real Triton kernel executed on the target Tesla T4 (PyTorch 2.8.0+cu128, Triton 3.4.0) — gives all 64 outputs finite and E = 0.033647 (err norm 0.26876 / denominator 7.98760), a ~3x margin inside the 0.1 bound (c1.e1/t8). This is the kernel's true behavior, not a probe artifact: an fp32 emulation of kernel.py lines 12-18 matches the actual kernel output to 3.6e-7 max-abs, and an independent NumPy E crosscheck agrees to 2.9e-7. The per-branch quantization error is real and large (relative 0.211/0.242), but problem.txt explicitly permits this ("Approximate intermediate values are allowed if the FINAL output meets this requirement. Either branch may individually exceed 0.1 without violating the contract"), so only the combined final output decides — and it passes. The pass mechanism is verified as designed-into-the-workload and deterministic rather than a fluke: the generator's PERMUTATION is exactly an anti-sort pairing (64/64 rows), producing a -0.964 cross-branch error correlation and a 4.78x cancellation in the combined output, with the kernel's own combined error norm matching the structural emulation to 1.2e-7 relative (c2.e1/t9, t13). Alternative row pairings (identity/random, E ≈ 0.161, 0/200 within the bound) would fail, but those are not the workload the contract specifies, so they are generalization notes, not defects. Both in-scope skeptic claims are decisively rebutted on the exact contract workload; no open claims remain (claim_coverage lists none), and the Skeptic reviewed all 13 tool events including the final mechanism probe and recorded no new claims (t14). No unresolved claim is being set aside. Within the finite-workload verdict scope problem.txt declares, the kernel is compliant with ~3x margin, so trust is warranted.

API calls with recorded usage: 7; input tokens: 271768; output tokens: 107864; probes: 3.

Automated trace audit: ["tool error: finalize_probe_evidence -- tool finalize_probe_evidence missing required arg: supports"]

## Interpretation limits

This is a selected synthetic pair testing finite-workload numerical compliance. It is not a proof over all valid inputs, not a representative dataset, and not a claim that the kernel is an upstream bug. With the correct reference and metric supplied, a fixed numerical script solves it. A tool advantage does not imply a debate advantage over a solo agent with tools. A source-only abstention is legitimate uncertainty; a token-cap exhaustion is not a wrong semantic judgment. Repeated calls on two frozen cases measure repeatability, not performance on independent held-out cases.

Model: accounts/fireworks/models/glm-5p3

Incomplete attempts: []
