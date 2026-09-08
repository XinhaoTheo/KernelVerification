"""Capture COMPLETE run directories for both agentic arms, for sharing.

The two baseline runners keep only the last 20,000 characters of transcript.md
and 8,000 of stdout; run.json, tool_events.jsonl, claims.json and every probe
script and its captured output stay in the Modal container and die with it. So
nothing complete exists on disk for any run made so far.

This runs one case under one arm and brings the whole run directory back as a
tar archive, which is unpacked under benchmark_fn_fp/traces/<case>/<arm>/.

Both arms run in the same container image on the same GPU, from the same
answer-free copy of the case, so a reader comparing the two traces is seeing a
difference in agent structure and nothing else.

Usage (from repo root):
    modal run benchmark_fn_fp/eval/capture_traces_modal.py --cases case_01 --arm debate
    modal run benchmark_fn_fp/eval/capture_traces_modal.py --cases case_01 --arm solo
"""
from __future__ import annotations

import io
import pathlib
import sys
import tarfile

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CASES_DIR = REPO_ROOT / "benchmark_fn_fp" / "eval_cases"
TRACES_DIR = REPO_ROOT / "benchmark_fn_fp" / "traces"

app = modal.App("kv-fn-fp-trace-capture")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "numpy", "anthropic", "openai", "python-dotenv")
    .add_local_dir(str(REPO_ROOT / "verifier"), "/root/verifier")
    .add_local_dir(str(CASES_DIR), "/root/cases")
)

ARMS = {
    "debate": "describer,skeptic,experimenter,judge",
    "solo": "solo",
}


@app.function(
    image=image,
    gpu="T4",
    timeout=5400,
    max_containers=2,
    secrets=[modal.Secret.from_dotenv(REPO_ROOT)],
)
def capture(entry: str, arm: str, max_rounds: int, model: str, max_tokens: int) -> dict:
    import contextlib
    import io as _io
    import os
    import tarfile as _tarfile
    import traceback
    from pathlib import Path

    os.chdir("/root")
    sys.path.insert(0, "/root")
    os.environ["AGENTIC_MODEL"] = model
    os.environ.setdefault("AGENTIC_PROVIDER", "anthropic")
    os.environ.setdefault("AGENTIC_PROBE_SANDBOX", "off")
    os.environ.setdefault("AGENTIC_LLM_TIMEOUT_SECONDS", "600")

    from verifier.agentic_run import main as agentic_main

    agents = ARMS[arm]
    argv = [
        entry,
        "--dataset-dir", "/root/cases",
        "--agents", agents,
        "--max-debate-rounds", str(max_rounds),
        "--model", model,
        # Adaptive thinking is billed against max_tokens. At the 4096 default a
        # solo turn spends the whole budget inside the thinking block and returns
        # no text and no tool call at all -- four of eleven turns died that way in
        # the first capture attempt, including all three of the final round's.
        "--max-tokens", str(max_tokens),
    ]

    buf = _io.StringIO()
    result: dict = {"entry": entry, "arm": arm, "ok": False, "error": None}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = agentic_main(argv)
        result["returncode"] = rc
    except Exception:
        result["error"] = traceback.format_exc()

    result["stdout"] = buf.getvalue()
    run_dir = Path("/root/cases") / entry / "agentic_runs" / agents.replace(",", "+")
    if run_dir.exists():
        blob = _io.BytesIO()
        # Everything: run.json, tool_events.jsonl, claims.json, transcript.md,
        # verdict.json when one exists, and probes/ with each probe's source and
        # its captured stdout/stderr.
        with _tarfile.open(fileobj=blob, mode="w:gz") as tar:
            tar.add(str(run_dir), arcname=".")
        result["tar"] = blob.getvalue()
        result["ok"] = (run_dir / "verdict.json").exists()
    else:
        result["error"] = result["error"] or f"no run dir at {run_dir}"
    return result


@app.local_entrypoint()
def main(cases: str = "", arm: str = "debate", max_rounds: int = 4,
         model: str = "claude-opus-5", max_tokens: int = 16384):
    if arm not in ARMS:
        print(f"--arm must be one of {sorted(ARMS)}", file=sys.stderr)
        raise SystemExit(1)
    names = [c.strip() for c in cases.split(",") if c.strip()]
    if not names:
        print("pass --cases a,b", file=sys.stderr)
        raise SystemExit(1)

    print(f"capturing {arm} traces for {len(names)} case(s), model={model}, "
          f"rounds={max_rounds}, max_tokens={max_tokens}")
    jobs = [(n, arm, max_rounds, model, max_tokens) for n in names]
    for r in capture.starmap(jobs):
        entry = r["entry"]
        dest = TRACES_DIR / entry / arm
        if r.get("tar"):
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(r["tar"]), mode="r:gz") as tar:
                tar.extractall(dest)
            (dest / "runner_stdout.txt").write_text(r.get("stdout") or "", encoding="utf-8")
        status = "OK" if r["ok"] else "NO VERDICT"
        print(f"\n=== {entry} [{arm}]: {status} ===")
        if r.get("error"):
            print(r["error"][:2000])
        for line in (r.get("stdout") or "").splitlines():
            if line.startswith(("stop_reason:", "turns:", "tools_executed:", "claims:", "llm_time_s:")):
                print("  " + line)
        if r.get("tar"):
            print(f"  wrote {dest}")
