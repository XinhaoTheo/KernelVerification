"""Baseline 2: a single LLM call, no tools, no debate, no execution.

Sends each case's problem.txt + kernel.py to one Claude call and asks for a
verdict in the same vocabulary the debate system's Judge uses. This is the
"just ask a strong model once" baseline -- it sees the same source the debate
system starts from, but cannot run anything, probe anything, or argue with
itself.

The verdict comes back through structured outputs (`output_config` with a JSON
schema), so constrained decoding guarantees the reply matches the schema and the
verdict is one of the three allowed strings. An earlier version parsed a JSON
object out of free text with a regex and fell back to scanning for the words
"trust"/"reject"; that fallback would score a sentence like "I would not reject
this" as a reject. A benchmark whose scoring can be flipped by phrasing measures
the parser, not the model.

Usage (from repo root, with ANTHROPIC_API_KEY set):
    python benchmark_fn_fp/eval/baseline2_single_llm.py --all
    python benchmark_fn_fp/eval/baseline2_single_llm.py --cases fn7_liger_rmsnorm_eps_placement
    python benchmark_fn_fp/eval/baseline2_single_llm.py --limit 2      # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traces import write_trace  # noqa: E402
from common import (  # noqa: E402
    INCONCLUSIVE,
    NO_VERDICT,
    REJECT,
    TRUST,
    load_cases,
    print_report,
    score,
)

# Claude Opus 5 list price, USD per million tokens. Only used for the cost line
# in the report; scoring never depends on it.
USD_PER_MTOK_IN = 5.0
USD_PER_MTOK_OUT = 25.0

MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0

SYSTEM_PROMPT = """You are verifying whether a GPU (Triton) kernel implementation is correct.

You will be given:
1. A problem statement describing the operation's contract and its reference behavior.
2. The kernel implementation under test.

Decide one of:
- "reject": the implementation has a real defect / violates the stated contract.
- "trust": the implementation is correct, or is an equally valid alternative implementation.
- "needs_more_evidence": you genuinely cannot tell.

Answer with the JSON object required by the output schema."""

USER_TEMPLATE = """## Problem statement (contract + reference behavior)

{problem}

## Kernel implementation under test (kernel.py)

```python
{kernel}
```

Is this implementation correct?"""

# Constrained decoding enforces this schema, so `verdict` is always one of the
# three strings and the reply is always parseable JSON. Numeric range and string
# length constraints are not part of the supported schema subset, so `confidence`
# is only declared as a number.
VERDICT_SCHEMA = {
    "type": "object",
    "required": ["verdict", "confidence", "reason"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [TRUST, REJECT, INCONCLUSIVE],
            "description": (
                "'reject' if the implementation has a real defect, 'trust' if it is "
                "correct or an equally valid alternative, 'needs_more_evidence' if "
                "you genuinely cannot tell."
            ),
        },
        "confidence": {"type": "number", "description": "0.0 to 1.0."},
        "reason": {"type": "string", "description": "One or two sentences."},
    },
    "additionalProperties": False,
}


# Opus 5 thinks adaptively before answering, and that thinking is billed against
# max_tokens. At 1024 the harder cases spent the whole budget inside the thinking
# block and returned stop_reason=max_tokens with no text block at all -- which
# looks exactly like a parse failure. Give the model room to finish.
_DEFAULT_MAX_TOKENS = 8192


# Adaptive thinking makes the needed budget case-dependent: fn7 finished in 290
# output tokens, the harder NSA cases spent 8192 without emitting the answer.
# Rather than guess one constant, escalate when the model says it ran out.
_MAX_TOKEN_ESCALATIONS = 3


def judge_one(client, model: str, case, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> dict:
    """One case, one call. Retries transient API failures; never guesses a verdict."""
    last_error: str | None = None
    escalations = 0
    user_prompt = USER_TEMPLATE.format(problem=case.problem_txt, kernel=case.kernel_py)
    for attempt in range(1, MAX_ATTEMPTS + _MAX_TOKEN_ESCALATIONS + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:  # transient 429/5xx, connection resets
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt - escalations >= MAX_ATTEMPTS:
                break
            time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))
            continue

        usage = getattr(resp, "usage", None)
        base = {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "attempts": attempt,
            "max_tokens": max_tokens,
            "stop_reason": getattr(resp, "stop_reason", None),
        }
        if base["stop_reason"] == "max_tokens" and escalations < _MAX_TOKEN_ESCALATIONS:
            # The whole budget went into the thinking block; the answer never
            # started. Give it more room rather than scoring a thinking-length
            # accident as a failure to answer.
            escalations += 1
            max_tokens *= 2
            continue

        # The schema-conforming JSON arrives in a text block. Opus 5 may emit a
        # thinking block first, so select by type rather than taking content[0].
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        # This arm has no orchestrator run directory, so without writing them here
        # the exact prompt and the raw reply -- the only record of what was asked
        # and what came back -- would exist nowhere once the process exits.
        thinking = "".join(getattr(b, "thinking", "") or ""
                           for b in resp.content if getattr(b, "type", None) == "thinking")
        write_trace(
            case.name, "single_call",
            files={
                "system_prompt.txt": SYSTEM_PROMPT,
                "user_prompt.txt": user_prompt,
                "response_text.json": text,
                "response_thinking.txt": thinking,
                "usage.json": json.dumps({**base, "model": model}, indent=2),
            },
        )
        try:
            args = json.loads(text)
            verdict = str(args["verdict"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            # Constrained decoding should make this unreachable; if it happens,
            # it is a failure to answer, not an "inconclusive" judgement, and is
            # scored separately rather than silently counted as a verdict.
            return {**base, "verdict": NO_VERDICT, "confidence": None,
                    "reason": (f"unparseable reply ({type(exc).__name__}, "
                               f"stop_reason={base['stop_reason']}): {text[:200]}")}
        return {**base, "verdict": verdict,
                "confidence": args.get("confidence"),
                "reason": str(args.get("reason", ""))}

    return {"verdict": NO_VERDICT, "confidence": None,
            "reason": f"API failed after {MAX_ATTEMPTS} attempts: {last_error}",
            "input_tokens": 0, "output_tokens": 0, "attempts": MAX_ATTEMPTS,
            "stop_reason": None}


def _write_results(out_path: Path, *, model: str, cases, results: dict[str, str],
                   details: dict[str, dict]) -> None:
    by_name = {c.name: c for c in cases}
    scored = score(results, [by_name[n] for n in results])
    tok_in = sum(d.get("input_tokens") or 0 for d in details.values())
    tok_out = sum(d.get("output_tokens") or 0 for d in details.values())
    payload = {
        "baseline": "single_llm_no_debate",
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            **scored,
            "tokens_in": tok_in,
            "tokens_out": tok_out,
            "usd": round(tok_in / 1e6 * USD_PER_MTOK_IN + tok_out / 1e6 * USD_PER_MTOK_OUT, 4),
        },
        "cases": {
            name: {
                "group": by_name[name].group,
                "seed_class": by_name[name].seed_class,
                "real_name": by_name[name].real_name,
                "kernel_family": by_name[name].kernel_family,
                "verdict": info["verdict"],
                "ground_truth": by_name[name].ground_truth,
                "correct": info["verdict"] == by_name[name].ground_truth,
                "confidence": info.get("confidence"),
                "reason": info.get("reason"),
                "input_tokens": info.get("input_tokens"),
                "output_tokens": info.get("output_tokens"),
                "attempts": info.get("attempts"),
            }
            for name, info in details.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cases", default="", help="comma-separated case names")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    parser.add_argument("--model", default=os.getenv("AGENTIC_MODEL", "claude-opus-5"))
    parser.add_argument("--out", default="benchmark_fn_fp/eval/results_baseline2.json")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1

    cases = load_cases()
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.name in wanted]
    elif not args.all and args.limit is None:
        parser.error("pass --all, --cases, or --limit")
    if args.limit is not None:
        cases = cases[: args.limit]

    from anthropic import Anthropic

    # Anaconda ships brotli 1.0.9, whose Decompressor.process() takes only
    # positional arguments, while httpx>=0.28 calls it with keywords. Any
    # brotli-encoded response then dies as a bare "APIConnectionError:
    # Connection error", which looks like a network fault and is not one.
    # Asking for encodings that do not go through that decoder sidesteps it
    # without requiring the caller to repair their interpreter's packages.
    client = Anthropic(default_headers={"accept-encoding": "gzip, deflate"})
    out_path = Path(args.out)
    results: dict[str, str] = {}
    details: dict[str, dict] = {}

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name} ...", flush=True)
        info = judge_one(client, args.model, case)
        results[case.name] = info["verdict"]
        details[case.name] = info
        mark = "OK " if info["verdict"] == case.ground_truth else "XX "
        print(f"  {mark} got={info['verdict']} gt={case.ground_truth} "
              f"conf={info.get('confidence')} | {str(info.get('reason'))[:100]}")
        # Written after every case so an interrupted run keeps what it earned.
        _write_results(out_path, model=args.model, cases=cases,
                       results=results, details=details)

    print_report(f"Baseline 2: single {args.model} call, no tools", results, cases)
    summary = json.loads(out_path.read_text())["summary"]
    print(f"  tokens_in={summary['tokens_in']} tokens_out={summary['tokens_out']} "
          f"cost=${summary['usd']}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
