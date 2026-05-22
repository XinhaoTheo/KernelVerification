"""Shared types for the debate layer."""

from __future__ import annotations

from typing import Literal, TypedDict

Role = Literal["author", "skeptic", "verifier", "judge"]


class Claim(TypedDict, total=False):
    id: str  # "c1", "c2", ... assigned by debate loop when registered
    type: str  # "non_contiguous_bug" / "numerical_drift" / "fake_triton" / ...
    statement: str  # what the skeptic asserts might be wrong
    raised_by: Role
    round: int
    evidence: list[dict]  # filled by verifier: probe input + measured diff
    status: Literal["open", "confirmed", "rebutted", "inconclusive"]
    judge_note: str  # judge's optional override / severity note at final verdict


class Turn(TypedDict, total=False):
    by: Role
    round: int
    text: str
    claims: list[Claim] | None
    tool_calls: list[dict] | None
