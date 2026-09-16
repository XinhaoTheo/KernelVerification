"""One row per model: everything that has to change when the model changes.

These settings used to live in three files and had no way to disagree loudly.
`max_tokens` was a single flat default shared by every model, the provider ->
model map lived in the runner, and prices lived in the summarizer keyed by a
slug that was derived independently. Adding a model meant remembering three
edits across two files, and forgetting one of them is exactly how a full 32-case
debate run went out at `max_tokens=4096` and came back with three cases holding
zero claims, zero probes, and about $33 of nothing.

The values here are measured, not guessed. Each one is justified at its row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    model: str
    provider: str
    max_tokens: int
    timeout_s: int
    # USD per million tokens. cache_write/cache_read are multipliers on price_in:
    # a provider with no prompt caching bills cached tokens as ordinary input,
    # so both are 1.0 there.
    price_in: float
    price_out: float
    cache_write: float
    cache_read: float
    # Top-level directory this model's traces live in, one tree per model:
    # benchmark_fn_fp/<traces_dir>/<case_id>/<arm>/.
    traces_dir: str
    known: bool = True


PROFILES: dict[str, ModelProfile] = {
    "claude-opus-5": ModelProfile(
        model="claude-opus-5",
        provider="anthropic",
        # Measured peak output per role is 6.3k-8.5k, so this leaves about half
        # in reserve. Adaptive thinking is billed against it: too low and a turn
        # spends its whole budget thinking and returns no text and no tool call.
        max_tokens=16384,
        # Turns measured at 45-52s; the client default of 60 was close enough
        # that staying under it was luck, and one run died on APITimeoutError.
        timeout_s=600,
        price_in=5.0,
        price_out=25.0,
        cache_write=2.0,
        cache_read=0.1,
        traces_dir="traces_opus5",
    ),
    "z-ai/glm-5.3-flash": ModelProfile(
        model="z-ai/glm-5.3-flash",
        provider="openrouter",
        # Double Opus's. This model writes its whole analysis as ordinary output
        # text rather than in a thinking block: one measured turn emitted 15,834
        # tokens, 97% of a 16,384 budget, and only produced its tool call because
        # it happened to fit. The model's own ceiling is 131,072, so this is not
        # near any hard limit -- it is bounded by time, at 20-55 tok/s.
        max_tokens=32768,
        # That same turn took 769 seconds, past the 600 this used to be set to.
        timeout_s=1800,
        price_in=0.075,
        price_out=0.25,
        # No prompt caching was observed on the measured runs: cache_read came
        # back 0 on every turn, so cached tokens bill as ordinary input.
        cache_write=1.0,
        cache_read=1.0,
        traces_dir="traces_glm",
    ),
}

# Used when only the provider is known, e.g. `--provider openrouter` with no
# --model. Kept explicit rather than "first profile with this provider" so
# adding a second model for a provider cannot silently change what runs.
DEFAULT_MODEL_FOR_PROVIDER = {
    "anthropic": "claude-opus-5",
    "openrouter": "z-ai/glm-5.3-flash",
}


def _slugify(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")


def profile_for(model: str) -> ModelProfile:
    """The profile for `model`, or a conservative unknown one.

    An unknown model still runs -- trying a new model must not require editing
    this file first -- but it carries no prices, so summarize_traces reports its
    cost as unknown rather than inventing a number. Add a row here once a model
    is worth keeping.
    """
    if model in PROFILES:
        return PROFILES[model]
    return ModelProfile(
        model=model,
        provider="",
        max_tokens=16384,
        timeout_s=1800,
        price_in=0.0,
        price_out=0.0,
        cache_write=1.0,
        cache_read=1.0,
        traces_dir=f"traces_{_slugify(model)}",
        known=False,
    )


def traces_dir_for(model: str) -> str:
    """The tree this model's traces live in.

    One tree per model rather than one tree with model-suffixed arm names: a $1
    run on an open model must not land anywhere near $88 of Opus traces, and
    within a tree the arm is just `solo` or `debate`, so the same case reads the
    same way whichever model produced it.
    """
    return profile_for(model).traces_dir


def label_for_traces_dir(name: str) -> str:
    """`traces_glm` -> `glm`; the name an arm carries on the scoreboard."""
    return name[len("traces_"):] if name.startswith("traces_") else name
