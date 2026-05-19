"""Three-agent debate loop over a kernel artifact."""

from __future__ import annotations

import os
from typing import Any

from agents import author, judge, skeptic
from agents.types import Turn


def run_debate(
    artifact: dict[str, Any],
    *,
    max_rounds: int | None = None,
    tools=None,
) -> tuple[dict, list[Turn]]:
    """Run author -> skeptic -> judge review loop.

    Author describes what the kernel does + how it evolved.
    Skeptic challenges that description against the kernel they share.
    Judge decides convergence (MVP: "no new arguments this round") and
    eventually renders the verdict. Stops at max_rounds regardless.

    Returns (verdict_dict, full_history).
    """
    if max_rounds is None:
        max_rounds = int(os.getenv("DEBATE_MAX_ROUNDS", "4"))

    history: list[Turn] = []
    for round_idx in range(max_rounds):
        auth_turn = author.respond(history, artifact, tools)
        history.append(auth_turn)

        skp_turn = skeptic.respond(history, artifact, tools)
        history.append(skp_turn)

        if judge.no_new_arguments(history, artifact, round_idx):
            break

    verdict = judge.final_verdict(history, artifact)
    return verdict, history
