"""Robust JSON extraction from LLM responses.

Agents emit prose plus a structured block — and the prose often ALSO contains
fenced code blocks (e.g. the skeptic quoting ```python kernel lines). So we
can't just grab the first fence. Strategy:
  1. Collect every fenced block; try json-tagged ones first, then untagged.
  2. Return the first that parses to a dict.
  3. Fall back to the last balanced {...} in the raw text.
Returns None on failure so callers degrade gracefully ("no new claims").
"""

from __future__ import annotations

import json
import re
from typing import Any

# Capture the fence tag + body so we can prefer ```json over ```python.
_FENCE = re.compile(r"```([a-zA-Z0-9_-]*)\s*\n(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the most plausible JSON object in `text`, or None."""
    fences = _FENCE.findall(text)  # list of (tag, body)
    # json-tagged first, then everything else, preserving order within each group.
    ordered = [b for tag, b in fences if tag.lower() == "json"] + [
        b for tag, b in fences if tag.lower() != "json"
    ]
    for body in ordered:
        obj = _try_load_object(body)
        if obj is not None:
            return obj

    # No fenced block parsed — try the last balanced {...} in the raw text.
    return _try_load_object(text)


def _try_load_object(s: str) -> dict[str, Any] | None:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
