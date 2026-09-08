"""Baseline 3: the debate system (verifier/agentic) run on a real Modal GPU.

Why a wrapper is needed: verifier/agentic/tools/execution.py runs agent-written
probes with `subprocess.Popen([sys.executable, probe_path])` -- i.e. on whatever
machine the orchestrator process itself lives on. There is no remote-execution
path. Rather than modify that tool, this runs the WHOLE orchestrator inside a
Modal GPU container, so its "local" subprocess already has a real CUDA device.

Usage (from repo root):
    modal run benchmark_fn_fp/eval/baseline3_debate_modal.py --cases fn7_liger_rmsnorm_eps_placement
    modal run benchmark_fn_fp/eval/baseline3_debate_modal.py --all
"""
from __future__ import annotations

import json
import pathlib
import sys

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CASES_DIR = REPO_ROOT / "benchmark_fn_fp" / "eval_cases"  # answer-free copy: no test.py, no verdict in meta.json

app = modal.App("kv-fn-fp-debate-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "numpy", "anthropic", "openai", "python-dotenv")
    .add_local_dir(str(REPO_ROOT / "verifier"), "/root/verifier")
    .add_local_dir(str(CASES_DIR), "/root/cases")
)

AGENTS = "describer,skeptic,experimenter,judge"


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    # Each debate run holds a T4 for minutes and burns real API tokens; cap the
    # fan-out so an --all run cannot saturate the workspace GPU quota.
    max_containers=4,
    secrets=[modal.Secret.from_dotenv(REPO_ROOT)],
)
def run_debate(entry: str, max_debate_rounds: int, model: str) -> dict:
    """Run the full agentic debate for one case inside this GPU container."""
    import os
    import io as _io
    import io
    import contextlib
    import tarfile as _tarfile
    import traceback
    from pathlib import Path

    os.chdir("/root")
    sys.path.insert(0, "/root")
    os.environ["AGENTIC_MODEL"] = model
    os.environ.setdefault("AGENTIC_PROVIDER", "anthropic")
    # The orchestrator's probe sandbox shells out to systemd on Linux; Modal
    # containers have no user session bus, so leave the sandbox off and rely on
    # the wall-clock/CPU limits the tool already applies.
    os.environ.setdefault("AGENTIC_PROBE_SANDBOX", "off")
    # The client default is 60s. A single turn with a large max_tokens and
    # adaptive thinking exceeds that -- the solo agent's first run died on
    # APITimeoutError, and debate turns were already measured at 45-52s, close
    # enough to the default to be luck.
    os.environ.setdefault("AGENTIC_LLM_TIMEOUT_SECONDS", "600")

    from verifier.agentic_run import main as agentic_main

    argv = [
        entry,
        "--dataset-dir", "/root/cases",
        "--agents", AGENTS,
        "--max-debate-rounds", str(max_debate_rounds),
        "--model", model,
    ]

    buf = io.StringIO()
    result: dict = {"entry": entry, "ok": False, "verdict": None, "error": None}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = agentic_main(argv)
        result["returncode"] = rc
        run_dir = Path("/root/cases") / entry / "agentic_runs" / AGENTS.replace(",", "+")
        verdict_path = run_dir / "verdict.json"
        transcript_path = run_dir / "transcript.md"
        if transcript_path.exists():
            result["transcript"] = transcript_path.read_text()[-20000:]
        # The whole run directory comes back too: probes, tool events, claims,
        # the untruncated transcript. The 20k tail above is only for the summary
        # JSON; a run whose complete record was thrown away cannot be diagnosed.
        blob = _io.BytesIO()
        if run_dir.exists():
            with _tarfile.open(fileobj=blob, mode="w:gz") as tar:
                tar.add(str(run_dir), arcname=".")
            result["tar"] = blob.getvalue()
        if verdict_path.exists():
            result["verdict"] = json.loads(verdict_path.read_text())
            result["ok"] = True
        else:
            result["error"] = f"no verdict.json at {verdict_path}"
    except Exception:
        result["error"] = traceback.format_exc()

    result["stdout"] = buf.getvalue()[-8000:]
    return result


@app.local_entrypoint()
def main(cases: str = "", all: bool = False, max_debate_rounds: int = 3,
         model: str = "claude-opus-5", out: str = "benchmark_fn_fp/eval/results_baseline3.json"):
    names = [c.strip() for c in cases.split(",") if c.strip()] if cases else None
    if not names and not all:
        print("pass --cases a,b or --all", file=sys.stderr)
        raise SystemExit(1)
    if not names:
        names = sorted(d.name for d in CASES_DIR.iterdir()
                       if d.is_dir() and (d / "meta.json").exists())

    print(f"running debate system on {len(names)} case(s), model={model}, rounds={max_debate_rounds}")
    jobs = [(n, max_debate_rounds, model) for n in names]
    # NOT list(): starmap yields each result as its container finishes, and
    # materialising the whole iterator first means nothing reaches disk until
    # all 32 cases are done. A failure at case 30 would then throw away the 29
    # runs already paid for. Iterated directly, each trace is written the
    # moment it arrives.
    outputs = run_debate.starmap(jobs)

    # Imported here, not at module scope: Modal imports this module inside
    # the container as well, and only /root/verifier and /root/cases are
    # mounted there. A module-level import of a sibling in this directory
    # therefore crashes every container at startup -- which it did, and the
    # run hung for hours retrying before I noticed.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from traces import write_trace

    results: dict[str, str] = {}
    details: dict[str, dict] = {}
    done = 0
    for r in outputs:
        entry = r["entry"]
        # Written before anything else in the loop: a crash while scoring must
        # not be what costs us the record of a run that already happened.
        trace_dir = write_trace(
            entry, "debate",
            tar=r.pop("tar", None),
            files={"runner_stdout.txt": r.get("stdout") or "",
                   "runner_error.txt": r.get("error") or ""},
        )
        done += 1
        print(f"  [{done}/{len(names)}] trace: {trace_dir}", flush=True)
        details[entry] = r
        # "the system judged this inconclusive" and "the system never produced a
        # verdict" are different outcomes and must not be scored as the same
        # thing: one is a judgement, the other is a failure to answer.
        verdict = (r.get("verdict") or {}).get("verdict")
        results[entry] = verdict or "no_verdict"
        print(f"\n=== {entry}: {'OK' if r['ok'] else 'ERROR'} ===")
        if r.get("verdict"):
            v = r["verdict"]
            print(f"  verdict={v.get('verdict')} confidence={v.get('confidence')}")
            print(f"  reason={str(v.get('reason'))[:300]}")
        if r.get("error"):
            print(f"  error: {r['error'][:1500]}")

    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "max_debate_rounds": max_debate_rounds,
                                    "results": results, "details": details}, indent=2))
    print(f"\nwrote {out_path}")
