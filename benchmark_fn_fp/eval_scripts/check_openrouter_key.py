"""Print what the OpenRouter key in .env can still spend.

Two limits look alike and fail differently, which cost a run to learn: the
workspace has a credit balance, and each key has its own cap on top of it. A key
whose own cap is spent still authenticates -- GET /credits returns 200 -- and
then every inference call comes back 403 "Key limit exceeded". So check the key,
not the balance.

There is no web page for this. The key was issued from someone else's
workspace, so its dashboard needs their login; this endpoint answers to anyone
holding the key.

Usage:  python benchmark_fn_fp/eval_scripts/check_openrouter_key.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def key_from_env() -> str | None:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = REPO / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip() or None
    return None


def fetch(key: str) -> dict:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())["data"]


def main() -> int:
    key = key_from_env()
    if not key:
        print("OPENROUTER_API_KEY is not set in the environment or .env", file=sys.stderr)
        return 2
    try:
        data = fetch(key)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        print(f"HTTP {exc.code}: {body}", file=sys.stderr)
        # 401 is an expired or revoked key; 403 here would be a key that cannot
        # even read its own record. Both mean no run will start.
        return 1

    limit = data.get("limit")
    usage = data.get("usage") or 0.0
    remaining = data.get("limit_remaining")

    print(f"key       {data.get('label')}")
    print(f"expires   {data.get('expires_at') or 'never'}")
    if limit is None:
        print(f"limit     none (bounded only by the workspace balance)")
        print(f"used      ${usage:.2f}")
    else:
        print(f"limit     ${limit:.2f}")
        print(f"used      ${usage:.2f}")
        print(f"left      ${(remaining or 0):.2f}")

    # Measured on this benchmark: solo ~$0.011/case, debate ~$0.022/case.
    left = remaining if limit is not None else None
    if left is not None:
        if left <= 0:
            print("\nSPENT. Inference will return 403 'Key limit exceeded' even though "
                  "the key still authenticates. Ask the key's owner to raise its limit.")
            return 1
        print(f"\nenough for roughly {int(left / 0.011)} solo runs "
              f"or {int(left / 0.022)} debate runs "
              f"(32 cases: ~$0.35 solo, ~$0.70 debate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
