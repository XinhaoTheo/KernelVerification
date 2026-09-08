"""Run an agentic arm over the benchmark on a real Modal GPU.

Replaces three near-identical runners -- baseline2_5_solo_modal.py,
baseline3_debate_modal.py and capture_traces_modal.py -- which were 506 lines
between them and differed in one meaningful line, the agent roster. Keeping
three copies in step was the direct cause of the worst bug in this project's
history: `--max-tokens` was added to two of them and missed on the third, and a
full 32-case debate run at the 4096 default produced three cases with zero
claims and zero probes, one Judge writing "the debate produced no claims and no
evidence" before recording a verdict anyway, and about $33 of nothing. The same
split had already caused three smaller versions of the same mistake (the
write_trace import, streaming results to disk, max_containers). One copy is the
fix.

Why a Modal wrapper at all: verifier/agentic/tools/execution.py runs
agent-written probes with `subprocess.Popen([sys.executable, probe_path])` --
on whatever machine the orchestrator process itself lives on. There is no
remote-execution path, so the whole orchestrator runs inside a GPU container
and its "local" subprocess already has a real CUDA device.

Usage (from repo root):
    modal run benchmark_fn_fp/eval/run_agentic_modal.py --arm solo   --all
    modal run benchmark_fn_fp/eval/run_agentic_modal.py --arm debate --all
    modal run benchmark_fn_fp/eval/run_agentic_modal.py --arm debate --cases case_33

Every run writes a complete trace to benchmark_fn_fp/traces/<case>/<arm>/.
The scoreboard is built from those traces by summarize_traces.py, not from a
summary file this script overwrites -- results_baseline3.json lost 18 of 32
cases exactly that way.
"""
from __future__ import annotations

import json
import pathlib
import sys

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# The answer-free copy: no test.py, no verdict in meta.json, opaque case ids.
CASES_DIR = REPO_ROOT / "benchmark_fn_fp" / "eval_cases"

# The one line that actually differs between arms.
ARMS = {
    "solo": "solo",
    "debate": "describer,skeptic,experimenter,judge",
}

app = modal.App("kv-fn-fp-agentic-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "numpy", "anthropic", "openai", "python-dotenv")
    .add_local_dir(str(REPO_ROOT / "verifier"), "/root/verifier")
    .add_local_dir(str(CASES_DIR), "/root/cases")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=5400,
    # Each run holds a T4 for minutes and burns real API tokens; cap the fan-out
    # so an --all run cannot saturate the workspace GPU quota.
    max_containers=4,
    secrets=[modal.Secret.from_dotenv(REPO_ROOT)],
)
def run_one(entry: str, arm: str, max_rounds: int, model: str, max_tokens: int) -> dict:
    """Run one case under one arm inside this GPU container."""
    import contextlib
    import io
    import os
    import tarfile
    import traceback
    from pathlib import Path

    os.chdir("/root")
    sys.path.insert(0, "/root")
    os.environ["AGENTIC_MODEL"] = model
    os.environ.setdefault("AGENTIC_PROVIDER", "anthropic")
    # The probe sandbox shells out to systemd; Modal containers have no user
    # session bus, so leave it off and rely on the wall-clock and CPU limits the
    # tool already applies.
    os.environ.setdefault("AGENTIC_PROBE_SANDBOX", "off")
    # The client default is 60s, and a single turn with a large max_tokens and
    # adaptive thinking exceeds it: the solo agent's first run died on
    # APITimeoutError, and debate turns were measured at 45-52s, close enough to
    # the default that staying under it was luck.
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
        # turn spends its whole budget inside the thinking block and returns no
        # text and no tool call at all. Measured peaks per role are 6.3k-8.5k,
        # so 16384 leaves about half in reserve.
        "--max-tokens", str(max_tokens),
    ]

    buf = io.StringIO()
    result: dict = {"entry": entry, "arm": arm, "ok": False, "verdict": None, "error": None}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result["returncode"] = agentic_main(argv)
    except Exception:
        result["error"] = traceback.format_exc()

    run_dir = Path("/root/cases") / entry / "agentic_runs" / agents.replace(",", "+")
    verdict_path = run_dir / "verdict.json"
    if run_dir.exists():
        # The whole run directory comes back: run.json, tool_events.jsonl,
        # claims.json, the untruncated transcript, and every probe's source and
        # captured output. Fifteen runs were made before this existed and not
        # one left a complete record; three defects that changed conclusions
        # were invisible until traces were.
        blob = io.BytesIO()
        with tarfile.open(fileobj=blob, mode="w:gz") as tar:
            tar.add(str(run_dir), arcname=".")
        result["tar"] = blob.getvalue()
    if verdict_path.exists():
        result["verdict"] = json.loads(verdict_path.read_text())
        result["ok"] = True
    elif not result["error"]:
        result["error"] = f"no verdict.json at {verdict_path}"

    result["stdout"] = buf.getvalue()[-8000:]
    return result


@app.local_entrypoint()
def main(arm: str = "", cases: str = "", all: bool = False, max_rounds: int = 0,
         model: str = "claude-opus-5", max_tokens: int = 16384,
         skip_existing: bool = False):
    if arm not in ARMS:
        print(f"--arm must be one of {sorted(ARMS)}", file=sys.stderr)
        raise SystemExit(1)
    # The debate reaches a verdict in 7 turns; the solo agent needs more rounds
    # because one agent does every role's work in sequence.
    if not max_rounds:
        max_rounds = 4 if arm == "debate" else 10

    names = [c.strip() for c in cases.split(",") if c.strip()] if cases else None
    if not names and not all:
        print("pass --cases a,b or --all", file=sys.stderr)
        raise SystemExit(1)
    if not names:
        names = sorted(d.name for d in CASES_DIR.iterdir()
                       if d.is_dir() and (d / "meta.json").exists())

    # Imported here, not at module scope: Modal imports this module inside the
    # container too, where only /root/verifier and /root/cases are mounted. A
    # module-level import of a sibling in this directory crashes every container
    # at startup -- which it did, and the run hung for hours retrying.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from traces import write_trace, TRACES_DIR

    if skip_existing:
        # Restarting a --all run used to re-run cases already paid for; two
        # restarts of one batch threw away real money that way. This does NOT
        # protect traces made before a behaviour change -- delete those, or
        # leave the flag off, when the agents themselves have changed.
        kept = [n for n in names if not (TRACES_DIR / n / arm / "verdict.json").exists()]
        for n in names:
            if n not in kept:
                print(f"  skip {n}: trace already has a verdict")
        names = kept
        if not names:
            print("nothing to run: every case already has a trace")
            return

    print(f"running {arm} on {len(names)} case(s), model={model}, "
          f"rounds={max_rounds}, max_tokens={max_tokens}", flush=True)
    jobs = [(n, arm, max_rounds, model, max_tokens) for n in names]
    # NOT list(): starmap yields each result as its container finishes, so each
    # trace reaches disk the moment it arrives. Materialising the iterator first
    # meant a failure at case 30 discarded the 29 runs already paid for.
    done = 0
    failures = []
    for r in run_one.starmap(jobs):
        entry = r["entry"]
        # Written before anything else in the loop: a crash while reporting must
        # not cost the record of a run that already happened.
        trace_dir = write_trace(
            entry, arm,
            tar=r.pop("tar", None),
            files={"runner_stdout.txt": r.get("stdout") or "",
                   "runner_error.txt": r.get("error") or ""},
        )
        done += 1
        v = r.get("verdict") or {}
        status = f"{v.get('verdict')} conf={v.get('confidence')}" if r["ok"] else "ERROR"
        print(f"  [{done}/{len(names)}] {entry}: {status}", flush=True)
        print(f"      trace: {trace_dir}", flush=True)
        if not r["ok"]:
            failures.append(entry)
            print(f"      error: {str(r.get('error'))[:800]}", flush=True)

    print(f"\n{done}/{len(names)} complete, {len(failures)} failed"
          + (f": {', '.join(failures)}" if failures else ""))
    print("score it with: python benchmark_fn_fp/eval/summarize_traces.py")
