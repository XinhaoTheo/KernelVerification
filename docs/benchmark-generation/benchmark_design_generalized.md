# Adversarial `allclose` Benchmark Seed Set (Draft)

We have 13 cases and plan to expand the benchmark to 100 cases: 50 FN + 50 FP.

First, the rules, to make sure the benchmark stays on target:

- **FN (the half that fails when the tolerance is too loose):** The kernel really has a bug, but a verification system using a loose tolerance cannot expose it with conventional or random tests, so `allclose` reports a pass.
- **FP (the half that fails when the tolerance is too strict):** The kernel is correct, or is another equally valid implementation, but a verification system using a strict tolerance compares it against one reference implementation and reports a failure.
- Each case ultimately compares continuous outputs, such as attention results, weighted sums, or loss values, rather than directly comparing the discrete selection itself.
- Every kernel family is real and has publicly available source code or a paper; none is invented for this benchmark.

These seeds are not fixed verification recipes. For each case, the agents should first debate the operator contract, the suspected failure mechanism, and competing explanations. They should then turn the strongest hypothesis into a targeted probe, interpret the resulting evidence, and decide whether it confirms a bug, supports a valid alternative implementation, or remains inconclusive.

The named projects establish that the kernel family and numerical mechanism are real. Unless a linked source explicitly documents the defect, each **Bug** below should be read as a benchmark mutation or proposed witness, not as an allegation about the cited upstream implementation.

---

## FN Group: Real Bugs That Conventional Tests or Loose Tolerances Fail to Detect

### FN1. Incorrect handling of a deterministic tie-breaking rule

#### 1. Explanation

When several candidates receive the same score, the code needs a clear rule for choosing between them. A bug occurs when the implementation does not follow that rule.

#### 2. Example

**Concrete example:** [Native Sparse Attention (NSA)](https://arxiv.org/abs/2502.11089) is designed to reduce the cost of attention on long sequences. Instead of making each query attend to every previous token, NSA combines three sparse branches: a compression branch for summarizing global context, a selection branch for retrieving the most relevant token blocks, and a sliding-window branch for preserving recent local context. This benchmark focuses on the selection branch: it scores candidate blocks, selects the top-k blocks, and performs attention only over the selected tokens. Public implementations are available from [fla-org](https://github.com/fla-org/native-sparse-attention) and [XunhaoLai](https://github.com/XunhaoLai/native-sparse-attention-triton).

**Bug:** The benchmark requires Top-K ties to favor the lower-index block. However, the buggy implementation uses `>=` instead of `>`, allowing a later, higher-index block with the same score to replace it. Without this tie-breaking requirement, both selections would be valid and this would be an FP case rather than a bug.

**Why `allclose` misses it:** If tests generate Q and K from Gaussian random values, FP32 block scores are almost never exactly equal, so random tests rarely trigger this bug. The two implementations almost always select the same blocks, and their final outputs naturally match. Even when a tied input is constructed deliberately, the continuous output produced by the incorrectly selected block may be close enough to the correct output—especially after its effect is diluted by many other softmax terms—that the difference remains within a tolerance such as `rtol=1e-2`.

**Evaluation boundary:** This is a synthetic mutation of a real NSA structure. It does not claim that this exact bug exists in the cited fla-org or XunhaoLai implementations. Whether an exact tie occurs naturally depends on the causes listed in FP1, including symmetric inputs, identical compressed representations, and low-precision rounding collisions. To make this FN deterministic and reproducible, the benchmark directly constructs a tied input.

#### 3. How agents should investigate it

The agents should first debate whether deterministic tie-breaking is actually part of the contract and which source statement establishes it. If it is required, they can inspect the update logic, construct equal scores at the top-k boundary, and trace the selected block into the downstream attention result. The Judge should distinguish a contract violation from an equally valid selection under an unspecified contract.

#### 4. Extension directions

The same seed can extend to stable `argmax`, beam-search ranking, segmented top-k, nearest-neighbor block selection, or MoE routing whenever the API explicitly requires a deterministic winner for ties.

### FN2. Bugs can hide in code paths that are rarely triggered

#### 1. Explanation

Some code runs only when an uncommon condition is met. A bug in that code cannot be detected easily if normal tests never trigger the condition.

#### 2. Example

**Concrete example:** The rejection-sampler path used by [vLLM speculative decoding](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md). After a draft token is rejected, the kernel samples a replacement token from a corrected distribution.

**Bug:** In the standard algorithm, the corrected distribution after rejection is `max(0, target_prob - draft_prob)`: each negative difference is first clipped to zero, and the remaining nonnegative values are then summed for normalization. The buggy kernel omits the clipping step and directly sums the raw values of `target_prob - draft_prob`, which include both positive and negative terms. This makes the denominator smaller than it should be and introduces a systematic bias into the normalized correction distribution.

**Why `allclose` misses it:** This bug appears only after a draft token is rejected, but rejection is rare in conventional tests. It changes the sampling distribution rather than producing an obviously wrong single output, so detecting it requires many rejected samples and a statistical test—not a one-run `allclose` comparison.

#### 3. How agents should investigate it

The agents should reason about which branch is rare, what invariant it must preserve, and whether a one-run output can expose the defect. From that reasoning, they can construct draft and target distributions that trigger many rejections, inspect the clipping and normalization path, and compare the observed replacement-token distribution with the mathematically expected one. Evidence from only a successful common-path run should not resolve the claim.

#### 4. Extension directions

This pattern extends to fallback kernels, overflow recovery, all-masked rows, empty inputs, NaN/Inf handling, and any error or resampling branch that ordinary workloads rarely enter.

### FN3. Regular test shapes hide boundary-indexing bugs

#### 1. Explanation

A kernel may work for convenient, evenly divisible input sizes but fail when the data does not divide evenly. Testing only common, regular shapes leaves the final partial group untested.

#### 2. Example

**Concrete example:** [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)-style Triton kernels that fuse dequantization and matrix multiplication. INT4 weights are divided into groups—for example, one group per 128 columns—and each group uses its own scale. With activation ordering enabled, columns are permuted by importance, so the appropriate scale can no longer be computed by simple division and must instead be looked up through a separate mapping table, `g_idx`.

**Bug:** In the basic version, the kernel computes the number of groups using floor division, `hidden_size // group_size`, instead of ceiling division. For `hidden_size=300` and `group_size=128`, there should be three groups: the last group contains only 44 columns, but it is still a group. Floor division returns only two groups. The final 44 columns are therefore left unhandled and have no valid scale, which can cause an out-of-bounds lookup.

**Why `allclose` misses it:** This is not a case where the tolerance is insufficiently strict; the buggy branch is never exercised. Whenever `hidden_size` is divisible by `group_size`, such as `4096 / 128 = 32`, floor and ceiling division produce exactly the same group count. The kernel output can match the reference bit for bit because no numerical error is produced at all. Most benchmarks use convenient hidden sizes such as 2048, 4096, or 8192 and never try an unusual shape such as 300.

#### 3. How agents should investigate it

The agents should identify every divisibility assumption in the indexing logic and debate which dimensions the input contract allows to have a tail. They can then choose the smallest irregular shape that crosses the suspected boundary, trace the final partial group, and compare it with an independent dequantization path. The test shape should follow from the suspected indexing bug rather than from a fixed list of odd sizes.

#### 4. Extension directions

Related seeds include partial GEMM tiles, softmax tails, ragged sequences, non-power-of-two feature dimensions, odd attention head sizes, and quantization groups that do not divide the tensor exactly.

### FN4. Homogeneous test data hides routing and indexing bugs

#### 1. Explanation

A routing bug can be invisible when the values being routed look the same. The test needs distinct data at each destination so that an incorrect mapping changes the result.

#### 2. Example

**Concrete example:** A [Grouped Query Attention (GQA)](https://arxiv.org/abs/2305.13245) kernel in which query heads are mapped in groups to shared key/value heads.

**Bug:** The mapping should divide the query-head index by the repetition factor `n_rep`. For example, with 32 query heads and 8 KV heads, `n_rep=4`, so query heads 0–3 all map to KV head 0. The buggy kernel instead takes the query-head index modulo the number of KV heads, so query heads 0, 8, 16, and 24 map to KV head 0. These are completely different groupings.

**An analogy for the mechanism:** Suppose there are eight mailboxes and each should receive a specific letter. If all eight letters contain different text, placing a letter in the wrong mailbox is immediately visible. If all eight letters are identical flyers, however, opening any mailbox reveals the same content and the routing mistake is invisible. The correct and incorrect results look identical because the data lacks distinguishing information.

**Why `allclose` may still miss it:** A test may lazily provide identical or highly similar values to different KV heads instead of independently generating distinguishable data for each head. In that case, reading the wrong head produces nearly the same result. Another weak test may inspect only a coarse statistic such as the average number of query heads assigned to each KV head. Both the correct and buggy mappings assign four query heads to each KV head; they disagree only on which four. Such an aggregate check therefore cannot detect the error.

#### 3. How agents should investigate it

The agents should debate which axis is being routed, what mapping formula is required, and whether the current inputs make destinations distinguishable. They can then encode a recognizable signature in each KV head and trace each query head's output back to its source. Aggregate counts alone are insufficient evidence because they can remain correct under a wrong permutation.

#### 4. Extension directions

The same mechanism appears in MoE expert routing, batch or beam reordering, KV-cache page tables, channel permutations, stride calculations, and any gather/scatter kernel tested with duplicated values.

### FN5. Synthetic tests miss quantization bugs triggered by real input distributions

#### 1. Explanation

Random test data may not resemble real model data. A method that appears correct on mild synthetic values can fail when real weights or activations contain large outliers.

#### 2. Example

**Concrete example:** INT8 matrix multiplication, a common inference optimization in which weights and activations are represented as INT8 for faster execution.

**Bug:** The scale should be computed from the actual maximum value in the current batch, but the kernel instead uses an exponential moving average (EMA) of historical maximum values.

**Why `allclose` misses it:** This is not primarily a missing-configuration problem. The distribution of synthetic test data differs from the distribution of real model data. Randomly initialized tensors, such as standard Gaussian samples, usually lack the structured, channel-specific extreme values seen in real model activations, so the EMA scale remains close to the scale computed from the current batch. However, real trained LLM weights and activations often contain a few channels with unusually large values. These outlier features are discussed by methods such as [LLM.int8()](https://arxiv.org/abs/2208.07339) and [SmoothQuant](https://arxiv.org/abs/2211.10438). On realistic distributions, the EMA cannot track rapidly changing outliers, and the genuinely large values in the current batch are severely clipped.

#### 3. How agents should investigate it

The agents should debate which statistics control quantization and whether the synthetic input distribution exercises their failure range. They can inspect the scale update rule, study real activation or weight statistics when available, and construct controlled outlier patterns that isolate the proposed mechanism. A failure on arbitrary extreme data is not enough; the evidence should connect the tested distribution to realistic or contract-permitted inputs.

#### 4. Extension directions

This seed can cover per-channel activation outliers, calibration-set mismatch, asymmetric zero-points, delayed scale updates, heavy-tailed gradients, and quantizers whose accuracy depends on correlations absent from IID random tensors.

### FN6. Tests covering short sequences hide accumulated numerical errors

#### 1. Explanation

A tiny error may be harmless in one step but grow after the same operation is repeated many times. A short test can pass even though a long run becomes inaccurate.

#### 2. Example

**Concrete example:** A low-bit KV-cache compression method such as [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/), which compresses the KV cache to three bits and reports almost no accuracy loss.

**Bug:** There is a small error in the computation of the rotation matrix or codebook used by quantization. Its effect is tiny in a single decoding step.

**Why `allclose` misses it:** The limitation here is not inherent to `allclose`. If the outputs of sufficiently long generation were compared, the deviation would eventually become large enough to fail any reasonable tolerance. The real problem is that a conventional kernel unit test normally runs one forward pass and compares one output; it does not test long autoregressive generation. This therefore belongs to the broader category of inadequate scenario coverage, like FN3. FN3 omits irregular configurations, whereas this case omits long-horizon accumulation.

#### 3. How agents should investigate it

The agents should first debate whether the suspected error is local or accumulative and predict how it should grow with sequence length. They can then compare short and progressively longer executions, inspect the state update or quantization transform, and check whether the observed growth matches the hypothesis. The Judge should use the trend and downstream impact, not only one endpoint or one fixed sequence length.

#### 4. Extension directions

Other instances include recurrent state-space kernels, optimizer updates, iterative solvers, autoregressive state updates, repeated low-bit requantization, and normalization errors that compound over layers or time steps.

### FN7. Random inputs miss near-zero numerical edge cases

#### 1. Explanation

Random inputs usually avoid special values such as zero. Bugs that matter only near those values can remain invisible in ordinary random tests.

#### 2. Example

**Concrete example:** [RMSNorm](https://arxiv.org/abs/1910.07467), for which many LLM training and inference frameworks have a custom GPU implementation.

**Bug:** The intended formula is `sqrt(mean(x^2) + eps)`, but the kernel computes `sqrt(mean(x^2)) + eps`.

**Why `allclose` misses it:** Conventional tests generate inputs from a Gaussian distribution, whose mean square is extremely unlikely to be near zero. At normal magnitudes, placing `eps` inside or outside the square root creates only a tiny numerical difference that easily falls within tolerance. The bug becomes dramatic only when an input row is all zero or nearly zero, as may happen at padded embedding positions or when pruning or dropout zeros an entire row.

#### 3. How agents should investigate it

The agents should derive the expected behavior as the row norm approaches zero and inspect where `eps` enters the implementation. Based on that reasoning, they can probe zero and progressively smaller inputs and compare the observed scaling with the formula. This makes the experiment test a specific numerical claim instead of merely adding a special value to a generic test suite.

#### 4. Extension directions

The mechanism extends to LayerNorm variance, reciprocal and square-root kernels, cosine normalization, fully masked softmax rows, gradient underflow, and any operation whose formula changes character near zero.

---

## FP Group: Correct Implementations Misclassified by a Strict Tolerance and a Single Reference

### FP1. Unspecified tie-breaking allows multiple valid outputs

#### 1. Explanation

The shared mechanism is that a selection contains a tie, the specification does not define a unique correct answer, and different reasonable rules choose different results that cause downstream continuous outputs to diverge.

#### 2. Example

##### Example 1: The NSA top-k specification does not define tie-breaking

The same NSA structure as FN1. The difference is that FN1 assumes a contract requiring the lower index to win a tie. Here, the specification says nothing about tie-breaking. If two candidate blocks have equal scores, either selection satisfies the abstract requirement to select the highest-scoring blocks.

**Difference:** Implementation A selects the lower-index block when scores are tied, while implementation B selects the higher-index block. Both satisfy the abstract top-k contract, but the selected blocks contain different original K/V values and therefore produce different continuous attention outputs.

**Why `allclose` produces a false positive:** If implementation A is treated as the reference and implementation B is checked with `allclose`, the outputs may disagree substantially. Implementation B has nevertheless violated no declared requirement.

**How exact ties arise:** They can result from symmetric inputs, such as different compressed keys having equal dot products with the current query; identical compressed representations from repeated or constant input regions; distinct scores rounding to the same representable BF16, FP16, or quantized value; collisions after aggregating multiple score components; or deliberately degenerate inputs such as a zero query. Only ties between valid candidate blocks at the top-k boundary count. Padding blocks or causally masked blocks that share `-inf` should already be excluded and do not constitute a meaningful tie-breaking ambiguity.

> **Evaluation boundary:** This case does not claim that ties are common in ordinary NSA training or inference. It is a deliberately constructed boundary case that tests whether the verifier recognizes multiple valid outputs when the specification leaves tie-breaking undefined.

##### Example 2: Valid tie-breaking differences in quantized MoE routing

A quantized Mixture-of-Experts router that computes expert scores in FP16 or INT8, selects the top two experts, and returns a weighted sum of their outputs. This structure is used by inference systems such as [vLLM](https://github.com/vllm-project/vllm).

**Difference:** Both implementations follow the algorithm correctly, but when two expert scores tie—because two logits round to the same FP16 value—they use different yet reasonable tie-breaking rules and select different second experts.

**Why `allclose` produces a false positive:** Different experts are independently trained and have completely different weights, so choosing a different expert can easily change the final weighted continuous output, far beyond a conventional tolerance. Comparing implementation B against implementation A therefore reports B as wrong even though the paper does not specify which expert should win a tie.

##### Example 3: Valid tie-breaking differences in Top-P sampling

**Concrete example:** [vLLM's Triton top-p/top-k sampling kernel](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py).

**Difference:** When probabilities tie near the cutoff, as can happen in a smooth tail distribution, two implementations may use different but valid ordering rules and produce slightly different candidate sets. Both sets satisfy the constraint that cumulative probability is at least `p`.

**Why `allclose` produces a false positive:** The different candidate sets can produce different continuous downstream results, such as probability-weighted embeddings or expected outputs. Comparing one such result against the other can fail even though both candidate sets satisfy the unspecified tie contract.

#### 3. How agents should investigate it

The agents should debate whether the contract defines one exact winner or a set of valid tied outcomes. They can then construct a boundary tie, enumerate the permitted selections, and test whether each implementation stays inside that set and preserves the required downstream behavior. The Judge should reject a result only for violating the contract, not merely for differing from one reference's private tie-breaking convention.

#### 4. Extension directions

This seed extends to MoE routing, top-p sampling, beam search, `argmax`, sorting, nearest-neighbor retrieval, and sparse block selection whenever ties are possible and the contract leaves their ordering unspecified.

### FP2. Equivalent representation conventions produce different tensors

#### 1. Explanation

Two implementations can arrange the same mathematical information differently. Their tensors may differ element by element even when both produce the same behavior when used consistently.

#### 2. Example

**Concrete example:** [Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2104.09864), an essential component of many LLMs with multiple GPU implementations.

**Difference:** RoPE has two widely used layouts. GPT-NeoX style splits the feature dimension into two halves and rotates them, while GPT-J style rotates adjacent interleaved dimensions. The two formulations are mathematically equivalent up to a dimension permutation, but their intermediate tensors do not match element by element for the same input. Mixing the conventions without transforming the associated dimensions is incorrect; using either convention consistently is valid.

**Why `allclose` produces a false positive:** Taking the output of one convention as the reference and checking the other with `allclose` yields large differences across nearly every element, even though both conventions are valid and provide equivalent behavior when used consistently by the downstream model.

#### 3. How agents should investigate it

The agents should debate whether the mismatching tensors encode the same mathematical object under a known transformation. They can inspect each layout convention, propose the permutation that maps one representation to the other, and test both the mapped tensors and downstream attention behavior. A claim of equivalence needs an explicit mapping or behavioral witness; visual similarity or a loose tolerance is not enough.

#### 4. Extension directions

Related cases include NCHW versus NHWC layouts, packed versus separate QKV tensors, real/imaginary complex packing, transposed quantization scales, blocked weight formats, and any representation pair connected by a documented permutation or invertible transform.

### FP3. Equivalent reduction orders produce expected numerical drift

#### 1. Explanation

Both methods are mathematically equivalent, but their different addition or reduction orders introduce numerical drift that grows with problem size.

#### 2. Example

##### Example 1: One-pass online versus two-pass cross-entropy

A fused cross-entropy kernel such as [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), commonly used in training with vocabularies that may contain hundreds of thousands of tokens.

**Difference:** One implementation uses a one-pass online algorithm that updates the running maximum while scanning. Another uses a conventional two-pass algorithm. They are mathematically equivalent and differ only in the order of floating-point operations.

**Why `allclose` produces a false positive:** With a large vocabulary, cross-entropy must sum many FP16 values. Different summation orders introduce different rounding errors. A few per-token losses may therefore fail a strict elementwise comparison, even when the average loss over the batch is nearly identical.

##### Example 2: Mamba2 parallel chunked scan versus serial recurrence

The Mamba2/SSD chunked scan, including the [Triton implementation in the official state-spaces/mamba repository](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py), used by modern state-space and linear-attention models.

**Difference:** A chunked parallel scan and a simple serial recurrence are both exact mathematical algorithms rather than approximations, but they accumulate values in completely different orders. The parallel implementation uses tree-shaped reductions, whereas the serial implementation adds terms one step at a time.

**Why `allclose` produces a false positive:** As sequence length grows, the floating-point drift caused by the two accumulation orders becomes more visible. Elementwise differences on long sequences may exceed a tolerance calibrated on short sequences, even though the implementation is correct and the discrepancy has the expected signature of floating-point accumulation.

#### 3. How agents should investigate it

The agents should compare the mathematical algorithms and debate whether their only semantic difference is operation order. They can then vary vocabulary or sequence length, precision, and reduction structure to test whether the error grows with the expected floating-point signature while aggregate behavior remains stable. An unexplained discontinuity, directional bias, or invariant violation should remain an open bug claim rather than being dismissed as ordinary drift.

#### 4. Extension directions

The same reasoning applies to softmax, normalization, attention reductions, histograms, scatter/atomic accumulation, distributed all-reduce, prefix scans, and any parallel reduction compared with a serial reference.

### FP4. Different precision levels require different error models

#### 1. Explanation

Lower precision deliberately trades some numerical accuracy for speed and memory savings. A tolerance suitable for FP32 may therefore be too strict for a correct low-bit kernel.

#### 2. Example

**Concrete example:** Low-bit attention such as [Attn-QAT](https://arxiv.org/abs/2603.00040), a four-bit attention method with Triton kernels for training and inference.

**Difference:** Low-bit quantization inherently introduces more quantization noise. That error is part of the intended design, not a bug.

**Why `allclose` produces a false positive:** A verification system may apply a fixed tolerance calibrated for FP32 or FP16 kernels to a correct FP4 implementation. The expected quantization error then exceeds the tolerance and causes a failure, even though the kernel behaves correctly for its precision level.

#### 3. How agents should investigate it

The agents should debate what error is intentionally introduced by the declared numeric format and what behavior would indicate a real implementation defect. They can inspect scale, rounding, clipping, and accumulator semantics, then construct probes around representable boundaries and compare against a reference that emulates the same format. The resulting threshold or metric should be justified by that error model, not selected merely to make the kernel pass.

#### 4. Extension directions

This category includes FP8 or FP4 GEMM, INT4 dequantization, low-bit attention, mixed-precision accumulators, approximate `exp` or reciprocal instructions, and kernels that use format-specific saturation or flush-to-zero behavior.

### FP5. Randomized kernels can produce different correct outputs

#### 1. Explanation

Some kernels deliberately use randomness. Two correct runs do not need to return exactly the same values, so one run cannot serve as the only valid reference.

#### 2. Example

**Concrete example:** Gradient-accumulation or weight-update kernels used in low-precision BF16 or FP8 training. Some frameworks deliberately use stochastic rounding when converting FP32 intermediate results back to low-precision storage. The kernel randomly rounds up or down based on how close the FP32 value is to each representable low-precision value. Over many updates, this avoids a consistent rounding bias.

**Difference:** The same correct implementation can produce different elementwise weight updates on two runs of the same batch because the rounding decisions use different random values. This nondeterminism is intentional rather than erroneous.

**Why `allclose` produces a false positive:** If the output from one run is treated as the reference for another—even when both runs use the same code—many elements may differ. A deterministic elementwise comparison misclassifies the intended stochastic behavior as an uncontrolled implementation error.

#### 3. How agents should investigate it

The agents should first establish whether randomness is part of the contract, which distribution it should follow, and which properties must hold across runs. They can then design repeated trials, control seeds where useful, and test expected mean, bias, variance, or downstream convergence rather than requiring two samples to match. Randomness should not excuse malformed distributions or broken reproducibility guarantees.

#### 4. Extension directions

Related seeds include stochastic rounding, dropout, token sampling, randomized quantization or dithering, and kernels whose contract explicitly permits nondeterministic but statistically constrained outputs.

### FP6. Comparison metrics can become unstable in edge regimes

#### 1. Explanation

A comparison rule can exaggerate a harmless numerical difference. Near zero, a tiny absolute difference can appear very large when it is divided by an equally tiny reference value.

#### 2. Example

**Concrete example:** Any continuous kernel whose output may approach zero, such as heavily masked attention positions with very small remaining weights or gradients near convergence.

**Difference:** Both implementations are correct, but a particular true value is extremely small. One implementation may produce `1.0e-8`, while ordinary floating-point rounding makes the other produce `1.3e-8`. The absolute error is negligible.

**Why `allclose` produces a false positive:** Relative error divides `|a-b|` by the magnitude of the reference. As the reference approaches zero, even a negligible absolute difference is divided by a tiny number and appears disproportionately large. A verifier using only a strict relative tolerance can therefore report a severe mismatch even though the metric itself is unstable in this region. This is why [`torch.allclose`](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html) exposes both `atol` and `rtol` rather than relying on relative tolerance alone.

#### 3. How agents should investigate it

The agents should debate whether the apparent failure comes from the kernel or from a metric that becomes ill-conditioned in the observed value range. They can partition results by magnitude, inspect both absolute and relative error, and test whether the discrepancy changes any downstream invariant. The Judge should adopt a scale-aware metric only after the evidence identifies the unstable regime.

#### 4. Extension directions

This mechanism appears in cosine similarity near zero norm, relative error around zero, saturated activations, sparse outputs dominated by zeros, tiny probability tails, and gradients close to convergence.

---

## Summary

The seed set contains 13 cases: 7 FNs and 6 FPs. The cases cover real kernel families including sparse attention (NSA), speculative decoding, GPTQ INT4, GQA, INT8 quantization, KV-cache quantization, RMSNorm, MoE routing, RoPE, cross-entropy, FP8/FP4 attention, Mamba2, and top-p sampling.

The FN group contains 4 underlying ways to evade conventional testing:

1. **Insufficient scenario coverage (FN3 and FN6):** Tests use only conventional configurations, shapes, or sequence lengths, so the code path containing the bug is never tested. The numerical difference is not hidden; it is never produced. This mechanism can be reproduced in almost any kernel by finding a configuration dimension that conventional tests omit.
2. **Random test data misses special numerical values (FN1 and FN7):** Randomly generated inputs naturally avoid exact ties and near-zero values, so bugs triggered only at those boundary points remain dormant.
3. **Statistical evidence or realistic data distributions are required (FN2 and FN5):** The bug creates a distribution-level bias or appears only on real data distributions. A one-shot comparison with randomly initialized inputs does not provide the necessary evidence.
4. **Test data lacks variation along the misrouted dimension (FN4):** The bug itself can create a large numerical error, but homogeneous values across the confused dimension—or a test that checks only coarse aggregate statistics—make incorrect routing or indexing invisible. This differs from insufficient scenario coverage: the configuration is exercised, but the data contains too little distinguishing information to reveal the mistake. When expanding to 100 cases, this mechanism deserves more examples, such as misaligned MoE expert weights or batch elements.

The FP group contains six mechanisms:

1. **Valid ambiguity in tie-breaking (FP1):** A selection contains a tie and the specification does not define a unique answer. MoE routing and top-p sampling are two extensions of this mechanism.
2. **Different floating-point reduction orders (FP3):** The algorithms are mathematically equivalent, but different addition orders create numerical drift that grows with scale. Cross-entropy and Mamba2 are two variants.
3. **The precision level determines the appropriate tolerance (FP4):** Lower-bit quantization has a wider inherent noise floor, so one fixed tolerance should not be applied across precision levels.
4. **Different numerical representation conventions (FP2):** Parameterizations differ elementwise while remaining mathematically equivalent and producing the same downstream behavior.
5. **The kernel is intentionally stochastic (FP5):** A single deterministic comparison is incompatible with the kernel's design.
6. **The comparison metric is unstable in an extreme-value region (FP6):** The problem lies in the chosen metric, not in the kernel implementation.

Together, the FN group has four mechanisms and the FP group has six, with two mechanisms represented by multiple kernel variants. This taxonomy provides a useful path for expanding to 100 cases: reproduce each mechanism across several kernel families while continuing to add genuinely distinct mechanisms, rather than repeatedly relabeling the same pattern as a different bug.
