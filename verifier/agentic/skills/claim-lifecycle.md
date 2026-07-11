# Claim Lifecycle

Use claim statuses consistently:

- `open`: the hypothesis is recorded but not yet tested.
- `confirmed`: evidence supports the claim.
- `rebutted`: evidence contradicts the claim.
- `inconclusive`: available evidence cannot decide the claim.

Do not move a claim directly between `confirmed` and `rebutted`. If later
evidence invalidates the previous status, mark it `inconclusive` first or refine
the claim into a narrower new claim.

