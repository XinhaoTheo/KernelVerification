"""Judge: neutral arbiter. Decides convergence and renders a final verdict.

MVP convergence: "no new arguments this round" — judge re-reads the latest
author + skeptic turns and decides if either side introduced something
substantively new. Phase 2: switch to "all claims confirmed/rebutted".
"""

from __future__ import annotations

import json
import re

from verifier import llm_client

from .types import Turn

SYSTEM_NO_NEW = """You are the JUDGE in a structured review of a Triton GPU kernel.

You will be shown the full review so far (author + skeptic turns). Decide ONLY
this question: did the most recent author turn AND the most recent skeptic
turn each introduce substantively new information or challenge (relative to
everything that came before)?

Reply with exactly one JSON object on a single line:
  {"new_author": true|false, "new_skeptic": true|false, "reason": "<one short sentence>"}

Do not include any other text.
"""

SYSTEM_VERDICT = """You are the JUDGE in a structured review of a Triton GPU kernel.

You will be shown the full review. Render a final verdict on whether the
kernel should be trusted.

The author is a witness, not a defender: their job is to describe what the
kernel does and how it evolved. The skeptic raises challenges. Your job is
to weigh skeptic challenges against (a) the kernel/test code you can read
directly and (b) whether the author was able to ground claims in observable
artifacts.

Reply with exactly one JSON object:
  {
    "verdict": "trust" | "reject" | "needs_more_evidence",
    "confidence": 0.0-1.0,
    "key_author_points": ["...", "..."],
    "key_skeptic_points": ["...", "..."],
    "reason": "<2-4 sentences explaining the verdict>"
  }

Weigh evidence quality, not rhetorical force. A specific concrete concern
outweighs a vague reassurance. If the skeptic raised a real defect that the
author could not or did not address with observable evidence, prefer "reject"
or "needs_more_evidence".
"""


def no_new_arguments(history: list[Turn], artifact: dict, round_idx: int) -> bool:
    """Return True if neither side introduced anything new this round."""
    if round_idx == 0:
        return False

    raw = llm_client.call(
        system=SYSTEM_NO_NEW,
        artifact=artifact,
        history=history,
        max_tokens=256,
    )
    parsed = _extract_json(raw)
    if not parsed:
        return False
    return not parsed.get("new_author", True) and not parsed.get("new_skeptic", True)


def final_verdict(history: list[Turn], artifact: dict) -> dict:
    raw = llm_client.call(
        system=SYSTEM_VERDICT,
        artifact=artifact,
        history=history,
        max_tokens=1024,
    )
    parsed = _extract_json(raw)
    if parsed is None:
        return {
            "verdict": "needs_more_evidence",
            "confidence": 0.0,
            "reason": "judge response could not be parsed",
            "raw": raw,
        }
    return parsed


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
