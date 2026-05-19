"""Shared types for the debate layer."""

from __future__ import annotations

from typing import Literal, TypedDict

Role = Literal["author", "skeptic", "judge"]


class Claim(TypedDict, total=False):
    type: str
    statement: str
    evidence: list[dict]
    status: Literal["open", "rebutted", "confirmed"]


class Turn(TypedDict, total=False):
    by: Role
    round: int
    text: str
    claims: list[Claim] | None
    tool_calls: list[dict] | None
