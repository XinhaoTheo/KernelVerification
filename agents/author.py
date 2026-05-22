"""Author: structured witness.

Reads the final kernel AND session_dir/seed_*.py (KernelAgent's evolution trace)
and produces a grounded "what does this kernel do + how did it evolve"
description. The prompt forbids quality judgment — describe only what is
observably present.

This asymmetry is intentional:
- Author sees seeds (specialist reader of evolution history).
- Skeptic and judge see only the final kernel + author's testimony.
- Skeptic challenges author's description against the final kernel they share;
  on seed-evolution claims, skeptic must ask author to point at specific lines.
"""

from __future__ import annotations

from pathlib import Path

from verifier import llm_client

from .types import Turn

SYSTEM = """You are the AUTHOR/WITNESS in a structured review of a Triton GPU kernel.

You are NOT defending the kernel. You are a structured witness:
- Describe precisely what the kernel does, in steps a reader can verify against the code.
- Describe how it evolved: what early seeds looked like, what changed between
  seeds and the final kernel, what bugs were apparently caught and fixed.
- Do NOT claim the kernel is correct, fast, or well-tested. Do NOT speculate
  about intent beyond what the code/seeds show.
- Cite specific lines/files (e.g. "final_kernel.py:42" or "seed_2.py:8-15").
- If the skeptic asked questions in prior turns, answer them directly with
  observable facts only. If you cannot answer from the artifacts, say so.

Structure your response (~400 words max):
  1. What this kernel does (line-cited walkthrough)
  2. Notable changes from seeds to final
  3. Direct answers to skeptic questions (if any in prior turns)

CONVERGENCE SIGNAL: If you have nothing new to add — i.e., you would only
restate prior observations, re-cite lines you already cited, or repeat the
same answers to skeptic questions — START your response with the exact
sentinel:

    NO_NEW_OBSERVATIONS.

Then in 1-2 sentences explain why (e.g., "All concerns have been confirmed
with line citations in prior rounds and no new skeptic question was raised.").
Do NOT include sections 1-3 in this case. The judge uses this sentinel to
detect convergence and terminate the loop.
"""


def respond(history: list[Turn], artifact: dict, tools=None) -> Turn:
    extra = _read_seeds(artifact.get("session_dir"))
    text = llm_client.call(
        system=SYSTEM,
        artifact=artifact,
        history=history,
        extra_context=extra,
    )
    round_idx = sum(1 for t in history if t.get("by") == "author")
    return Turn(by="author", round=round_idx, text=text)


def _read_seeds(session_dir: str | None) -> dict[str, str]:
    """Return {filename: contents} for all seed_*.py and final_kernel.py."""
    if not session_dir:
        return {}
    sd = Path(session_dir)
    if not sd.exists():
        return {}
    out: dict[str, str] = {}
    for f in sorted(sd.glob("seed_*.py")):
        out[f.name] = f.read_text()
    final = sd / "final_kernel.py"
    if final.exists():
        out[final.name] = final.read_text()
    return out
