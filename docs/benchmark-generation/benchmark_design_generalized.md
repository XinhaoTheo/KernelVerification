# Mechanism-Oriented Adversarial `allclose` Benchmark Seed Set (Draft Copy)

We have 13 cases and plan to expand the benchmark to 100 cases: 50 FN + 50 FP.

> In this draft, case titles describe reusable verification-failure mechanisms. Named kernels appear in the body as concrete examples.

First, the rules, to make sure the benchmark stays on target:

- **FN (the half that fails when the tolerance is too loose):** The kernel really has a bug, but a verification system using a loose tolerance cannot expose it with conventional or random tests, so `allclose` reports a pass.
- **FP (the half that fails when the tolerance is too strict):** The kernel is correct, or is another equally valid implementation, but a verification system using a strict tolerance compares it against one reference implementation and reports a failure.
- Each case ultimately compares continuous outputs, such as attention results, weighted sums, or loss values, rather than directly comparing the discrete selection itself.
- Every kernel family is real and has publicly available source code or a paper; none is invented for this benchmark.

---

## FN Group: Real Bugs That Conventional Tests or Loose Tolerances Fail to Detect

### FN1. Incorrect handling of a deterministic tie-breaking rules

#### 1. Explanation

When several candidates receive the same score, the code needs a clear rule for choosing between them. A bug occurs when the implementation does not follow that rule.

#### 2. Example

**Concrete example:** Native Sparse Attention (NSA) is designed to reduce the cost of attention on long sequences. Instead of making each query attend to every previous token, NSA combines three sparse branches: a compression branch for summarizing global context, a selection branch for retrieving the most relevant token blocks, and a sliding-window branch for preserving recent local context. This benchmark focuses on the selection branch: it scores candidate blocks, selects the top-k blocks, and performs attention only over the selected tokens. Public Triton implementations are available in projects such as fla-org and XunhaoLai.

**Bug:** The benchmark requires Top-K ties to favor the lower-index block. However, the buggy implementation uses `>=` instead of `>`, allowing a later, higher-index block with the same score to replace it. Without this tie-breaking requirement, both selections would be valid and this would be an FP case rather than a bug.

**Why `allclose` misses it:** If tests generate Q and K from Gaussian random values, In FP32, randomly generated block scores are almost never exactly equal, so random tests rarely trigger this bug. the two implementations almost always select the same blocks, and their final outputs naturally match. Even when a tied input is constructed deliberately, the continuous output produced by the incorrectly selected block may be close enough to the correct output—especially after its effect is diluted by many other softmax terms—that the difference remains within a tolerance such as `rtol=1e-2`.

**Evaluation boundary:** This is a synthetic mutation of a real NSA structure. It does not claim that this exact bug exists in the cited fla-org or XunhaoLai implementations. Whether an exact tie occurs naturally depends on the causes listed in FP1, including symmetric inputs, identical compressed representations, and low-precision rounding collisions. To make this FN deterministic and reproducible, the benchmark directly constructs a tied input.

#### 3. How it should be verified

Review whether the comparison in the top-k update code is `>` or `>=`. Then construct an input that forces two valid block scores to be equal and check whether the selected index follows the lower-index tie-breaking contract, rather than looking only at the numerical error in the continuous output.

### FN2. Bugs can hide in code paths that are rarely triggered

#### 1. Explanation

Some code runs only when an uncommon condition is met. A bug in that code can not be easily detected if normal tests never trigger the condition.

#### 2. Example

**Concrete example:** The Triton rejection-sampler kernel used by vLLM for speculative decoding. After a draft token is rejected, the kernel samples a replacement token from a corrected distribution.

**Bug:** In the standard algorithm, the corrected distribution after rejection is `max(0, target_prob - draft_prob)`: each negative difference is first clipped to zero, and the remaining nonnegative values are then summed for normalization. The buggy kernel omits the clipping step and directly sums the raw values of `target_prob - draft_prob`, which include both positive and negative terms. This makes the denominator smaller than it should be and introduces a systematic bias into the normalized correction distribution.

**Why `allclose` misses it:** This bug appears only after a draft token is rejected, but rejection is rare in conventional tests. It changes the sampling distribution rather than producing an obviously wrong single output, so detecting it requires many rejected samples and a statistical test—not a one-run `allclose` comparison.

#### 3. How it should be verified

Review the code to confirm that `max(0, ...)` clipping is applied before computing the normalization denominator. For dynamic validation, deliberately make the draft and target distributions different enough to trigger many rejections, then perform a statistical distribution test rather than inspecting the numerical error of a single run.

### FN3. Regular test shapes hide boundary-indexing bugs

#### 1. Explanation

A kernel may work for convenient input sizes but fail when the data does not divide evenly. Testing only common, regular shapes leaves the final partial group untested.

#### 2. Example

**Concrete example:** AutoGPTQ-style Triton kernels that fuse dequantization and matrix multiplication. INT4 weights are divided into groups—for example, one group per 128 columns—and each group uses its own scale. With activation ordering enabled, columns are permuted by importance, so the appropriate scale can no longer be computed by simple division and must instead be looked up through a separate mapping table, `g_idx`.

**Bug:** In the basic version, the kernel computes the number of groups using floor division, `hidden_size // group_size`, instead of ceiling division. For `hidden_size=300` and `group_size=128`, there should be three groups: the last group contains only 44 columns, but it is still a group. Floor division returns only two groups. The final 44 columns therefore have no valid scale and either cause an out-of-bounds lookup or are incorrectly clamped to the preceding group and dequantized with the wrong scale. Misaligned `g_idx` handling under activation ordering is a more advanced real-world variant of the same problem.

**Why `allclose` misses it:** This is not a case where the tolerance is insufficiently strict; the buggy branch is never exercised. Whenever `hidden_size` is divisible by `group_size`, such as `4096 / 128 = 32`, floor and ceiling division produce exactly the same group count. The kernel output can match the reference bit for bit because no numerical error is produced at all. Most benchmarks use convenient hidden sizes such as 2048, 4096, or 8192 and never try a remainder-bearing shape such as 300.

#### 3. How it should be verified

Inspect the line that computes the number of groups and confirm that it performs ceiling rather than floor division. Then test a shape that is not divisible by the group size, such as `(300, 128)`, instead of validating only the default regular shapes.

### FN4. Homogeneous test data hides routing and indexing bugs

#### 1. Explanation

A routing bug can be invisible when the values being routed look the same. The test needs distinct data at each destination so that an incorrect mapping changes the result.

#### 2. Example

**Concrete example:** A Grouped Query Attention (GQA) kernel in which query heads are mapped in groups to shared key/value heads.

**Bug:** The mapping should divide the query-head index by the repetition factor `n_rep`. For example, with 32 query heads and 8 KV heads, `n_rep=4`, so query heads 0–3 all map to KV head 0. The buggy kernel instead takes the query-head index modulo the number of KV heads, so query heads 0, 8, 16, and 24 map to KV head 0. These are completely different groupings.

**An analogy for the mechanism:** Suppose there are eight mailboxes and each should receive a specific letter. If all eight letters contain different text, placing a letter in the wrong mailbox is immediately visible. If all eight letters are identical flyers, however, opening any mailbox reveals the same content and the routing mistake is invisible. The correct and incorrect results look identical because the data lacks distinguishing information.

**Why the original classification was wrong:** This case was initially grouped with GPTQ as another example of a regular configuration hiding an irregular boundary. That explanation does not hold. Unlike GPTQ, modulo and division produce different head groupings regardless of whether the head counts divide evenly. As long as different KV heads contain genuinely different data, reading the wrong head should change the attention output substantially and `allclose` should detect it easily.

**Why `allclose` may still miss it:** The actual masking mechanism is homogeneous test data—the equivalent of identical letters in every mailbox. A test may lazily provide identical or highly similar values to different KV heads instead of independently generating distinguishable data for each head. In that case, reading the wrong head produces nearly the same result. Another weak test may inspect only a coarse statistic such as the average number of query heads assigned to each KV head. Both the correct and buggy mappings assign four query heads to each KV head; they disagree only on which four. Such an aggregate check therefore cannot detect the error.

#### 3. How it should be verified

Ensure that every KV head in the test data is genuinely distinct rather than copied or duplicated. Then check, query head by query head, whether it reads the KV-head contents required by the mapping formula, rather than checking only an aggregate statistic. This mechanism generalizes to other routing and indexing kernels. For example, a misalignment of MoE expert weights or elements within a batch can remain hidden whenever test data lacks variation along the dimension being permuted.

### FN5. Synthetic data hides distribution-dependent quantization bugs

#### 1. Explanation

Random test data may not resemble real model data. A method that appears correct on mild synthetic values can fail when real weights or activations contain large outliers.

#### 2. Example

**Concrete example:** INT8 matrix multiplication in which weights and activations are represented as INT8 for faster inference, a common inference optimization.

**Bug:** The scale should be computed from the actual maximum value in the current batch, but the kernel instead uses an exponential moving average (EMA) of historical maximum values.

**Why `allclose` misses it:** This is not primarily a missing-configuration problem. The distribution of synthetic test data differs from the distribution of real model data. Randomly initialized tensors, such as standard Gaussian samples, usually contain relatively mild values and no strong outliers, so the EMA scale remains close to the scale computed from the current batch. Real trained LLM weights and activations, however, often contain a few channels with unusually large values. These outlier features are a real phenomenon discussed specifically by work such as LLM.int8() and SmoothQuant. On realistic distributions, the EMA cannot track rapidly changing outliers, and the genuinely large values in the current batch are severely clipped.

#### 3. How it should be verified

Do not test only randomly initialized tensors. Use distributions from real trained model weights and activations, or at least deliberately construct data with outliers. Also review whether the code computes the scale from the current batch or from some historical average.

### FN6. Short-horizon tests hide accumulated numerical errors

#### 1. Explanation

A tiny error may be harmless in one step but grow after the same operation is repeated many times. A short test can pass even though a long run becomes inaccurate.

#### 2. Example

**Concrete example:** A low-bit KV-cache compression method such as TurboQuant, which compresses the KV cache to three bits and reports almost no accuracy loss.

**Bug:** There is a small error in the computation of the rotation matrix or codebook used by quantization. Its effect is tiny in a single decoding step.

**Why `allclose` misses it:** The limitation here is not inherent to `allclose`. If the outputs of sufficiently long generation were compared, the deviation would eventually become large enough to fail any reasonable tolerance. The real problem is that a conventional kernel unit test normally runs one forward pass and compares one output; it does not test long autoregressive generation. This therefore belongs to the broader category of inadequate scenario coverage, like FN3. FN3 omits irregular configurations, whereas this case omits long-horizon accumulation.

#### 3. How it should be verified

Do not test only a single forward step. Measure end-to-end quality degradation after long-sequence generation, or review the mathematical derivation and implementation of the quantization rotation matrix or codebook directly.

### FN7. Random inputs miss near-zero numerical edge cases

#### 1. Explanation

Random inputs usually avoid special values such as zero. Bugs that matter only near those values can remain invisible in ordinary random tests.

#### 2. Example

**Concrete example:** RMSNorm, for which nearly every LLM training and inference framework has a custom Triton implementation.

**Bug:** The intended formula is `sqrt(mean(x^2) + eps)`, but the kernel computes `sqrt(mean(x^2)) + eps`.

**Why `allclose` misses it:** Conventional tests generate inputs from a Gaussian distribution, whose mean square is extremely unlikely to be near zero. At normal magnitudes, placing `eps` inside or outside the square root creates only a tiny numerical difference that easily falls within tolerance. The bug becomes dramatic only when an input row is all zero or nearly zero, as may happen at padded embedding positions or when pruning or dropout zeros an entire row.

#### 3. How it should be verified

Review where `eps` is added and construct an all-zero or near-zero input row, rather than relying exclusively on ordinary random inputs.

---

## FP Group: Correct Implementations Misclassified by a Strict Tolerance and a Single Reference

### FP1. Unspecified tie-breaking permits multiple valid outputs

#### 1. Explanation

The shared mechanism is that a selection contains a tie, the specification does not define a unique correct answer, and different reasonable rules choose different results that cause downstream continuous outputs to diverge. NSA is the primary example because it is the direct counterpart of FN1: whether the specification defines a tie-breaking rule is the single variable that determines whether the difference is a bug or a valid implementation choice. MoE routing and top-p sampling are extensions of the same mechanism.

#### 2. Example

##### Primary example: The NSA Top-K specification does not define tie-breaking

**Concrete example:** The same NSA structure as FN1. The difference is that FN1 assumes a contract requiring the lower index to win a tie. Here, the specification says nothing about tie-breaking. If two candidate blocks have equal scores, either selection satisfies the abstract requirement to select the highest-scoring blocks.

**Difference:** Implementation A selects the lower-index block when scores are tied, while implementation B selects the higher-index block. Both satisfy the abstract top-k contract, but the selected blocks contain different original K/V values and therefore produce different continuous attention outputs.

**Why `allclose` produces a false positive:** If implementation A is treated as the reference and implementation B is checked with `allclose`, the outputs may disagree substantially. B has nevertheless violated no declared requirement.

**How exact ties arise:** They can result from symmetric inputs, such as different compressed keys having equal dot products with the current query; identical compressed representations from repeated or constant input regions; distinct scores rounding to the same representable BF16, FP16, or quantized value; collisions after aggregating multiple score components; or deliberately degenerate inputs such as a zero query. Only ties between valid candidate blocks at the top-k boundary count. Padding blocks or causally masked blocks that share `-inf` should already be excluded and do not constitute a meaningful tie-breaking ambiguity.

**Evaluation boundary:** This case does not claim that ties are common in ordinary NSA training or inference. It is a deliberately constructed boundary case that tests whether the verifier recognizes multiple valid outputs when the specification leaves tie-breaking undefined. It is not an estimate of the real-world tie frequency. The MoE extension below has stronger evidence that close or tied scores occur in practice.

##### Extension: Valid tie-breaking differences in quantized MoE routing

**Concrete example:** A quantized Mixture-of-Experts router that computes expert scores in FP16 or INT8, selects the top two experts, and returns a weighted sum of their outputs. This structure is used by vLLM.

**Difference:** Both implementations follow the algorithm correctly, but when two expert scores tie—because two logits round to the same FP16 value—they use different yet defensible tie-breaking rules and select different second experts.

**Why `allclose` produces a false positive:** Different experts are independently trained and have completely different weights, so choosing a different expert can easily change the final weighted continuous output by more than 10%, far beyond a conventional tolerance. Comparing implementation B against implementation A therefore reports B as wrong even though the paper does not specify which expert should win a tie. Unlike the deliberately constructed NSA boundary case, this situation has support from actual training practice: noisy top-k routing is used specifically to break close or tied gating scores, indicating that this is a real phenomenon rather than a purely artificial input.

##### Extension: Valid tie-breaking differences in Top-P sampling

**Concrete example:** vLLM's Triton top-p/top-k sampling kernel, enabled through `VLLM_USE_TRITON_SAMPLER`.

**Difference:** When probabilities tie near the cutoff, as can happen in a smooth tail distribution, two implementations may use different but valid ordering rules and produce slightly different candidate sets. Both sets satisfy the constraint that cumulative probability is at least `p`.

**Why `allclose` produces a false positive:** Directly comparing the masks representing the two selected sets reveals many different zero/one entries, even though both selections are valid. This is analogous to two equally ranked entries being ordered differently without either ordering being incorrect.

#### 3. How it should be verified

Determine first whether the specification defines a tie-breaking rule. If it does not, a single reference output must not be treated as the only valid answer. Compare the overlap or recall of the selected sets, or validate the reasonableness of the downstream continuous result, instead of requiring elementwise equality.

### FP2. Equivalent representation conventions produce different tensors

#### 1. Explanation

Two implementations can arrange the same mathematical information differently. Their tensors may differ element by element even when both produce the same behavior when used consistently.

#### 2. Example

**Concrete example:** Rotary Position Embeddings (RoPE), an essential component of most LLMs with many Triton implementations.

**Difference:** RoPE has two widely used layouts. GPT-NeoX style splits the feature dimension into two halves and rotates them, while GPT-J style rotates adjacent interleaved dimensions. The two formulations are mathematically equivalent up to a dimension permutation, but their intermediate tensors do not match element by element for the same input. This is not hypothetical: Hugging Face Transformers has historically encountered real weight-conversion bugs caused by mixing these conventions.

**Why `allclose` produces a false positive:** Taking the output of one convention as the reference and checking the other with `allclose` yields large differences across nearly every element, even though both conventions are valid and provide equivalent behavior when used consistently by the downstream model.

#### 3. How it should be verified

Determine whether the two implementations use the same layout convention. If they do not, align the layouts explicitly or compare the resulting downstream attention behavior instead of directly comparing the intermediate RoPE tensors.

### FP3. Equivalent reduction orders produce expected numerical drift

#### 1. Explanation

This mechanism has two variants: cross-entropy and Mamba2 chunked scan. Both are mathematically equivalent implementations whose numerical differences arise solely from the order of addition or reduction, with the difference growing as the problem size increases. Cross-entropy is the primary example and Mamba2 is an extension.

#### 2. Example

##### Primary example: One-pass online versus two-pass cross-entropy

**Concrete example:** A fused cross-entropy kernel such as Liger-Kernel, commonly used in training with vocabularies that may contain hundreds of thousands of tokens.

**Difference:** One implementation uses a one-pass online algorithm that updates the running maximum while scanning. Another uses a conventional two-pass algorithm. They are mathematically equivalent and differ only in the order of floating-point operations.

**Why `allclose` produces a false positive:** With a very large vocabulary and FP16 arithmetic, non-associativity of floating-point addition can make the loss for an individual token differ by several percent, even though the difference nearly disappears after averaging over an entire batch. A strict per-token comparison can therefore report an inconsistency between two correct algorithms.

##### Extension: Mamba2 parallel chunked scan versus serial recurrence

**Concrete example:** The Mamba2/SSD chunked scan, including the Triton implementation in the official state-spaces/mamba repository, used by modern state-space and linear-attention models.

**Difference:** A chunked parallel scan and a simple serial recurrence are both exact mathematical algorithms rather than approximations, but they accumulate values in completely different orders. The parallel implementation uses tree-shaped reductions, whereas the serial implementation adds terms one step at a time.

**Why `allclose` produces a false positive:** As sequence length grows, the floating-point drift caused by the two accumulation orders becomes more visible. Elementwise differences on long sequences may exceed a tolerance calibrated on short sequences, even though the implementation is correct and the discrepancy has the expected signature of floating-point accumulation.

#### 3. How it should be verified

Confirm that the difference comes only from floating-point reduction order rather than an algorithmic error. Evaluate aggregate behavior and whether error growth follows the expected floating-point drift pattern as the problem size increases, instead of requiring every individual value to match tightly.

### FP4. Different precision levels require different error models

#### 1. Explanation

Lower precision deliberately trades some numerical accuracy for speed and memory savings. A tolerance suitable for FP32 may therefore be too strict for a correct low-bit kernel.

#### 2. Example

**Concrete example:** Low-bit attention such as Attn-QAT, a four-bit attention method with Triton kernels for training and inference.

**Difference:** Low-bit quantization inherently introduces more quantization noise. That error is part of the intended design, not a bug.

**Why `allclose` produces a false positive:** A verification system may apply a fixed tolerance calibrated for FP32 or FP16 kernels to a correct FP4 implementation. The expected quantization error then exceeds the tolerance and causes a failure, even though the kernel behaves correctly for its precision level.

#### 3. How it should be verified

Determine the acceptable error range for the specific bit width and quantization scheme, using the method and error levels reported by its design or paper, rather than applying a precision-independent fixed threshold.

### FP5. Intentional stochasticity invalidates one-run comparisons

#### 1. Explanation

Some kernels deliberately use randomness. Two correct runs do not need to return exactly the same values, so one run cannot serve as the only valid reference.

#### 2. Example

**Concrete example:** Gradient-accumulation or weight-update kernels used in low-precision BF16 or FP8 training. Some frameworks deliberately use stochastic rounding when converting FP32 intermediate results back to low-precision storage. The direction of rounding is sampled in proportion to the discarded remainder, avoiding the systematic bias of always rounding to the nearest value.

**Difference:** The same correct implementation can produce different elementwise weight updates on two runs of the same batch because the rounding decisions use different random values. This nondeterminism is intentional rather than erroneous.

**Why `allclose` produces a false positive:** If the output from one run is treated as the reference for another—even when both runs use the same code—many elements may differ. A deterministic elementwise comparison misclassifies the intended stochastic behavior as an uncontrolled implementation error.

#### 3. How it should be verified

First confirm from the design that stochastic rounding is intended. Then run the kernel repeatedly and validate the statistical properties of the output, including its mean and variance, rather than comparing one realization element by element with another.

### FP6. Comparison metrics can become unstable in edge regimes

#### 1. Explanation

A comparison rule can exaggerate a harmless numerical difference. Near zero, a tiny absolute difference can appear very large when it is divided by an equally tiny reference value.

#### 2. Example

**Concrete example:** Any continuous kernel whose output may approach zero, such as heavily masked attention positions with very small remaining weights or gradients near convergence.

**Difference:** Both implementations are correct, but a particular true value is extremely small. One implementation may produce `1.0e-8`, while ordinary floating-point rounding makes the other produce `1.3e-8`. The absolute error is negligible.

**Why `allclose` produces a false positive:** Relative error divides `|a-b|` by the magnitude of the reference. As the reference approaches zero, even a negligible absolute difference is divided by a tiny number and appears disproportionately large. A verifier using only a strict relative tolerance can therefore report a severe mismatch even though the metric itself is unstable in this region. This is why `torch.allclose` exposes both `atol` and `rtol` rather than relying on relative tolerance alone.

#### 3. How it should be verified

For reference values near zero, inspect absolute error and select an appropriate `atol` instead of applying a pure relative-error threshold uniformly to every element.

---

## Summary

The seed set contains 13 cases: seven FNs and six FPs. It is intentionally no longer a symmetric 6+6 set; preserving sound mechanisms is more important than forcing the seed count to be balanced. The cases cover real kernel families including sparse attention (NSA), speculative decoding, GPTQ INT4, GQA, INT8 quantization, KV-cache quantization, RMSNorm, MoE routing, RoPE, cross-entropy, FP8/FP4 attention, Mamba2, and top-p sampling.

The FN group contains four underlying ways to evade conventional testing:

1. **Insufficient scenario coverage (FN3 and FN6):** Tests use only conventional configurations, shapes, or sequence lengths, so the code path containing the bug is never exercised. The numerical difference is not hidden; it is never produced. This mechanism can be reproduced in almost any kernel by finding a configuration dimension that conventional tests omit.
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

## Questions for Review

1. Is this the right level of detail for the 13 cases—10 distinct mechanisms after consolidation, with three cases containing extension variants?
2. FN4, the GQA example, is currently the only case in which homogeneous test data hides a routing error. Should we add another variant now, such as misaligned MoE expert weights, or defer it until the expansion to 100 cases?
3. Once this seed set is approved, should the path to 100 cases primarily replicate each mechanism across additional kernel families, or should it also prioritize introducing new kernel families and new mechanism categories?
