# Adversarial Precision Verification

Use this skill when the kernel involves lossy, saturating, quantized, or
discrete-selection behavior where plain final-output allclose can be a weak
correctness proxy. This is guidance for claim and probe design, not a fixed
checklist. Only investigate a mode when the problem contract, kernel source, or
artifact context makes it relevant.

## Softmax And Activations

Softmax, ReLU, GELU, SiLU, and similar functions can hide errors in tails or
dead/saturated regions. A small pre-activation mistake may disappear after the
operation, while a small dominant-logit mistake can be amplified.

When relevant, prefer claims and probes that separate:

- pre-operation quantities from final output when source access makes that
  possible;
- dominant entries from tail entries;
- dead/saturated regions from active regions.

Do not confirm or reject from one benign random input if the suspected failure
only appears under tail-heavy or dominant-entry distributions.

## Sort And Top-k

Sort/top-k correctness is not a single elementwise numeric distance. Index
recall can falsely reject valid tie-breaking, while value-only comparison can
hide harmful misses.

When relevant, design evidence around the selected values and their gaps:

- distinguish harmless boundary swaps where selected and missed values are tied
  or nearly tied;
- treat missing a value far above the cutoff as a stronger bug signal;
- include tied, quantized, and cutoff-clustered inputs, not only `torch.randn`;
- when the top-k is used inside attention, consider a downstream softmax-value
  output comparison as better evidence than raw index overlap alone.

A claim about top-k should state which metric decides it: exact index order, set
overlap, selected values, value-gap-weighted misses, or downstream output.

## FP8 With Softmax Or Activations

FP8 can snap many small probabilities or activations to the same code or zero.
That can hide tail errors, and it can also make a correct lossy kernel look far
from an FP32 reference under a tight tolerance.

When relevant, compare against the contractually correct quantized behavior, not
only an FP32 reference with a magic tolerance. Probe tail probabilities, flush-to
zero cases, and dominant entries separately when those cases are in scope.

## FP4 And INT4

Very low bit formats can erase differences smaller than the quantization step.
Elementwise allclose against an FP32 reference is often not decisive by itself.

When relevant, prefer evidence based on quantized-reference behavior, downstream
impact, or task-level invariants. A probe should report quantization step, value
ranges, saturation/zero counts, and whether an observed mismatch is representable
in the target format.

## Verdict Discipline

Do not reject a kernel merely because an adversarial case is interesting. The
claim still needs scope evidence tying that case to the problem/test contract.
Do not trust a kernel merely because benign random probes pass when the operator
class is known to hide distribution-dependent failures.
