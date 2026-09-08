"""One API request per case; no debate, execution tools, retries, or token escalation.

Usage: /opt/anaconda3/bin/python benchmark_fn_fp/numerical_pilot/run_single.py
Resumes completed cases without paying again. Never passes the answer key to LLM.
"""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "eval"))
from baseline2_single_llm import SYSTEM_PROMPT, USER_TEMPLATE, VERDICT_SCHEMA


def main():
    load_dotenv(ROOT.parents[1] / ".env")
    model = os.getenv("AGENTIC_MODEL", "claude-opus-5")
    key = json.loads((ROOT / "answer_key.json").read_text())
    output = ROOT / "results_single.json"
    result = json.loads(output.read_text()) if output.exists() else {
        "model": model, "max_tokens": 8192, "protocol": "one request per case; no tools; no retries",
        "created_at": datetime.now(timezone.utc).isoformat(), "cases": {}}
    assert result["model"] == model
    jobs = []
    for name, info in key["cases"].items():
        d = ROOT / "eval_cases" / name
        code, problem = (d / "kernel.py").read_text(), (d / "problem.txt").read_text()
        assert hashlib.sha256(code.encode()).hexdigest() == info["kernel_sha256"]
        assert hashlib.sha256(problem.encode()).hexdigest() == info["problem_sha256"]
        if name in result["cases"]:
            assert result["cases"][name]["kernel_sha256"] == info["kernel_sha256"]
        else:
            jobs.append((name, code, problem))
    client = Anthropic(max_retries=0, timeout=180,
                       default_headers={"accept-encoding": "gzip, deflate"})

    def call(job):
        name, code, problem = job
        start = time.monotonic()
        try:
            response = client.messages.create(model=model, max_tokens=8192,
                system=SYSTEM_PROMPT,
                output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
                messages=[{"role": "user", "content": USER_TEMPLATE.format(problem=problem, kernel=code)}])
            raw = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
            row = {"usage": response.usage.model_dump(), "stop_reason": response.stop_reason,
                   "request_id": getattr(response, "_request_id", None), "raw": raw}
            try:
                row.update(json.loads(raw))
            except ValueError:
                row.update(verdict="no_verdict", reason="No parseable final output")
        except Exception as exc:
            # Do not print provider response bodies or credentials.
            row = {"verdict": "no_verdict", "error_type": type(exc).__name__,
                   "status_code": getattr(exc, "status_code", None), "usage": None}
        row["elapsed_s"] = round(time.monotonic() - start, 2)
        return name, row

    print(f"{len(jobs)} new requests; model={model}; max_tokens=8192; concurrency=2", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for name, row in pool.map(call, jobs):
            info = key["cases"][name]
            row.update(ground_truth=info["ground_truth"], family=info["config"]["family"],
                       kernel_sha256=info["kernel_sha256"],
                       correct=row["verdict"] == info["ground_truth"])
            result["cases"][name] = row
            rows = list(result["cases"].values())
            tok_in = sum((r["usage"] or {}).get("input_tokens", 0) for r in rows)
            tok_out = sum((r["usage"] or {}).get("output_tokens", 0) for r in rows)
            result["summary"] = {"n": len(rows), "correct": sum(r["correct"] for r in rows),
                "inconclusive": sum(r["verdict"] == "needs_more_evidence" for r in rows),
                "no_verdict": sum(r["verdict"] == "no_verdict" for r in rows),
                "input_tokens": tok_in, "output_tokens": tok_out,
                "estimated_usd": round((tok_in*5 + tok_out*25)/1e6, 4),
                "pricing_note": "Uses existing baseline assumption $5/$25 per million input/output tokens; not an invoice."}
            output.write_text(json.dumps(result, indent=2))
            print(f"{name}: {row['verdict']} / {row['ground_truth']} correct={row['correct']} "
                  f"cost_so_far~${result['summary']['estimated_usd']}", flush=True)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
