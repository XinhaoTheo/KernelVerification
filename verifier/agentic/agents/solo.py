"""Solo agent: one role holding the union of the working roles' tools.

This exists as the middle arm of an ablation. The debate system differs from a
single model call in two ways at once -- it can execute code, and it argues with
itself -- so a difference between those two cannot be attributed to either. The
solo agent has execution and iteration but no adversarial structure: it forms a
hypothesis, probes it, reads the result, probes again, and records its own
verdict, with nobody to challenge it.

    single call      no execution, no iteration, no critique
    solo (this)      execution + iteration, no critique
    debate           execution + iteration + critique

The gap from the first to this one measures what running code is worth. The gap
from this one to the debate measures what the adversarial structure is worth.

It runs through `run_agents_sequential`, not the debate workflow: there is no
Skeptic to sign off and no Describer to answer a clarification, so the workflow's
claim-coverage and sign-off gates would deadlock it.
"""

from __future__ import annotations

from verifier.agentic.agents.base import LLMAgent
from verifier.agentic.llm import LLMClient
from verifier.agentic.state import Role

_SOLO_INSTRUCTIONS = """
You are the sole verification agent for one GPU kernel. You decide whether the
kernel satisfies the contract in its problem statement, and you have a real GPU
available to run experiments on.

Nobody will review your reasoning, ask you for more evidence, or challenge a
conclusion you reach. Whatever you record is the final answer.

How to work:
- Read the problem statement and the kernel source first. The problem statement
  is the operative contract: what it requires is what the kernel must do, and
  what it leaves unspecified cannot be the basis for a reject.
- Form a specific, testable hypothesis about where the implementation could fail
  the contract, and record it with record_claim so your own later turns can see
  what you are testing.
- Run it. Use run_claim_probe to execute Python against the real kernel, and put
  the decisive numbers in the evidence `data` object with finalize_probe_evidence
  or append_evidence, not only in prose. Probes are how you find out what the
  code actually does rather than what it appears to do; a hypothesis you could
  have tested and did not is a weaker basis for a verdict than one you ran.
- Read what came back before deciding the next step. A probe that timed out, hit
  an error, or produced output you cannot interpret has not confirmed anything.
- Prefer the input domain the contract actually declares. A failure on an input
  the contract excludes is not a defect; a failure on one it admits is.
- Resolve every claim you open with update_claim_status before finishing.

Deciding:
- record_verdict exactly once, with "reject" if the implementation violates a
  behaviour the contract requires, "trust" if it satisfies the contract or is an
  equally valid alternative implementation, and "needs_more_evidence" if you
  genuinely cannot tell from what you were able to establish.
- Deviation the contract already accounts for is not a defect: quantization or
  rounding error consistent with a declared low-precision format, a different but
  internally consistent layout convention, a different floating-point reduction
  order, run-to-run variation in a kernel the contract declares randomized, a
  different choice among outcomes the contract leaves unspecified, or a large
  relative error on values so near zero the absolute difference is negligible.
- Conversely, when the contract explicitly requires a behaviour -- a formula, an
  invariant, a tie-break rule, a numeric format -- and evidence shows the kernel
  does not implement it, that is a defect however small it looks on typical
  inputs.
- Say in the reason what you ran and what it showed, citing the tool event ids,
  so the verdict can be checked rather than taken on trust.
""".strip()


def build_solo_agent(llm_client: LLMClient, *, max_tokens: int = 4096) -> LLMAgent:
    return LLMAgent(
        role=Role.SOLO,
        instructions=_SOLO_INSTRUCTIONS,
        llm_client=llm_client,
        skill_names=[
            "kernel-verification.md",
            "evidence-driven-review.md",
            "claim-lifecycle.md",
            "adversarial-precision.md",
            "metric-selection.md",
            "scope-policy.md",
            "experiment-design.md",
        ],
        max_tokens=max_tokens,
    )
