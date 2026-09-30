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
    modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --arm solo   --all
    modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --arm debate --all
    modal run benchmark_fn_fp/eval_scripts/run_agentic_modal.py --arm debate --cases case_33

Every run writes a complete trace to its model profile's trace path, followed
by <case>/<arm>/<trial>/. All GLM profiles share traces_glm/.
The scoreboard is built from those traces by summarize_traces.py, not from a
summary file this script overwrites -- results_baseline3.json lost 18 of 32
cases exactly that way.
"""
from __future__ import annotations

import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# The answer-free copy: no test.py, no verdict in meta.json, opaque case ids.
CASES_DIR = REPO_ROOT / "benchmark_fn_fp" / "triton_eval_cases"
_REMOTE_TRACE_ROOT = "/root/trace_runs"

# The one line that actually differs between arms.
ARMS = {
    "solo": "solo",
    "debate": "describer,skeptic,experimenter,judge",
}

app = modal.App("kv-fn-fp-agentic-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "triton==3.4.0", "numpy==1.26.4", "anthropic", "openai", "python-dotenv")
    .add_local_dir(str(REPO_ROOT / "verifier"), "/root/verifier")
    .add_local_dir(str(CASES_DIR), "/root/cases")
    .add_local_dir(str(REPO_ROOT / "benchmark_fn_fp/correlation_pair/eval_cases"), "/root/correlation_cases")
    .add_local_dir(str(REPO_ROOT / "benchmark_fn_fp/numerical_challenges/eval_cases"), "/root/numerical_cases")
    .add_local_dir(str(REPO_ROOT / "benchmark_fn_fp/evidence_challenges/eval_cases"), "/root/evidence_cases")
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
def run_one(entry: str, arm: str, max_rounds: int, model: str, max_tokens: int,
            provider: str, timeout_s: int, dataset: str = "benchmark_fn_fp", trial: str = "legacy",
            expected_hashes: dict | None = None, total_output_tokens: int = 0) -> dict:
    """Run one case under one arm inside this GPU container."""
    import contextlib
    import io
    import os
    import tarfile
    from pathlib import Path

    os.chdir("/root")
    sys.path.insert(0, "/root")
    # Both set explicitly, not setdefault: Secret.from_dotenv injects the whole
    # .env into the container, and it pins AGENTIC_PROVIDER and AGENTIC_MODEL.
    # A setdefault would silently keep those and run the wrong model under the
    # right flag.
    os.environ["AGENTIC_MODEL"] = model
    os.environ["AGENTIC_PROVIDER"] = provider
    # The probe sandbox shells out to systemd; Modal containers have no user
    # session bus, so leave it off and rely on the wall-clock and CPU limits the
    # tool already applies.
    os.environ.setdefault("AGENTIC_PROBE_SANDBOX", "off")
    # Set explicitly, like the two above: the client default is 60s, .env pins a
    # value of its own, and one measured GLM turn ran 769s. The per-model number
    # comes from models.PROFILES.
    os.environ["AGENTIC_LLM_TIMEOUT_SECONDS"] = str(timeout_s)
    # Clear a potentially injected dotenv setting when no cap was requested.
    if total_output_tokens > 0:
        os.environ["AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET"] = str(total_output_tokens)
    else:
        os.environ.pop("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", None)
    # Explicit experiment setting, also captured in each request. Observed
    # latency differences alone do not establish a cause for provider failures.
    if provider in {"fireworks", "openrouter"}:
        os.environ["AGENTIC_OPENAI_REASONING_EFFORT"] = "low"

    from verifier.agentic_run import main as agentic_main
    from verifier.agentic.llm_trace import _exception_details

    agents = ARMS[arm]
    dataset_dir = {"benchmark_fn_fp": "/root/cases", "correlation_pair": "/root/correlation_cases",
                   "numerical_challenges": "/root/numerical_cases",
                   "evidence_challenges": "/root/evidence_cases"}[dataset]
    if expected_hashes is not None:
        import hashlib
        for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt")):
            actual = hashlib.sha256((Path(dataset_dir) / entry / filename).read_bytes()).hexdigest()
            if actual != expected_hashes[f"{kind}_sha256"]:
                raise ValueError(f"Mounted case differs from checked source: {entry}/{filename}")
    run_dir = Path(_REMOTE_TRACE_ROOT) / trial / dataset / entry / arm
    if run_dir.exists():
        raise FileExistsError(f"Remote trial already exists: {run_dir}")
    os.environ["AGENTIC_LLM_TRACE_DIR"] = str(run_dir / "llm_calls")
    os.environ["AGENTIC_LLM_TRACE_PROGRESS"] = "1"
    argv = [
        entry,
        "--dataset-dir", dataset_dir,
        "--run-dir", str(run_dir),
        "--agents", agents,
        "--max-debate-rounds", str(max_rounds),
        "--model", model,
        "--provider", provider,
        # Adaptive thinking is billed against max_tokens. At the 4096 default a
        # turn spends its whole budget inside the thinking block and returns no
        # text and no tool call at all. Measured peaks per role are 6.3k-8.5k,
        # so 16384 leaves about half in reserve.
        "--max-tokens", str(max_tokens),
    ]

    original_stdout = sys.stdout
    class Tee(io.StringIO):
        def write(self, value):
            original_stdout.write(value)
            original_stdout.flush()
            return super().write(value)
    buf = Tee()
    result: dict = {"entry": entry, "arm": arm, "ok": False, "verdict": None, "error": None}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result["returncode"] = agentic_main(argv)
    except Exception as exc:
        result["error"] = json.dumps(_exception_details(exc), ensure_ascii=False)

    verdict_path = run_dir / "verdict.json"
    if run_dir.exists():
        # The whole run directory comes back: run.json, tool_events.jsonl,
        # claims.json, the untruncated transcript, and every probe's source and
        # captured output. Fifteen runs were made before this existed and not
        # one left a complete record; three defects that changed conclusions
        # were invisible until traces were.
        try:
            blob = io.BytesIO()
            with tarfile.open(fileobj=blob, mode="w:gz") as tar:
                tar.add(str(run_dir), arcname=".")
            result["tar"] = blob.getvalue()
        except Exception as exc:
            result["error"] = json.dumps(_exception_details(exc), ensure_ascii=False)
    try:
        if verdict_path.exists():
            result["verdict"] = json.loads(verdict_path.read_text())
            if not isinstance(result["verdict"], dict):
                raise ValueError("verdict.json must contain an object")
            result["ok"] = not result["error"] and result.get("returncode") == 0
        elif not result["error"]:
            result["error"] = f"no verdict.json at {verdict_path}"
        if not result["ok"] and not result["error"]:
            result["error"] = f"agentic runner returned {result.get('returncode')}"
    except Exception as exc:
        # The archive already contains all raw calls. A malformed verdict must
        # not prevent that paid work from being returned to the local caller.
        result["error"] = json.dumps(_exception_details(exc), ensure_ascii=False)

    result["stdout"] = buf.getvalue()
    return result


def _retain_failed_result(dest: pathlib.Path, error: Exception, *, kind: str,
                          result: dict | None = None) -> dict:
    """Keep a failed collection recoverable without overwriting prior payloads."""
    from traces import update_trace_metadata
    from verifier.agentic.llm_trace import _exception_details

    detail = _exception_details(error)
    detail["failure_kind"] = kind
    dest.mkdir(parents=True, exist_ok=True)
    archive = result.get("tar") if isinstance(result, dict) else None
    if isinstance(archive, bytes):
        archive_path = dest / f"recovery_run_{uuid4().hex}.tar.gz"
        with archive_path.open("xb") as handle:
            handle.write(archive)
        detail["recovery_archive"] = archive_path.name
    failure_path = dest / f"runner_failure_{uuid4().hex}.json"
    with failure_path.open("x") as handle:
        json.dump(detail, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    update_trace_metadata(dest, status="error", failure_kind=kind,
                          failure_file=failure_path.name,
                          **({"recovery_archive": detail["recovery_archive"]}
                             if "recovery_archive" in detail else {}))
    return detail


@app.local_entrypoint()
def main(arm: str = "", cases: str = "", all: bool = False, max_rounds: int = 0,
         provider: str = "anthropic", model: str = "", max_tokens: int = 0,
         skip_existing: bool = False, dataset: str = "benchmark_fn_fp", trial: str = "",
         total_output_tokens: int = 0):
    if total_output_tokens < 0:
        raise ValueError("total_output_tokens must be nonnegative")
    if arm not in (*ARMS, "both"):
        print(f"--arm must be one of {sorted(ARMS)} or both", file=sys.stderr)
        raise SystemExit(1)

    # Imported here, not at module scope: Modal imports this module inside the
    # container too, where only /root/verifier and /root/cases are mounted. A
    # module-level import of a sibling in this directory crashes every container
    # at startup -- which it did, and the run hung for hours retrying.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from traces import write_trace, trace_path, reserve_trace, new_trial_id
    from models import DEFAULT_MODEL_FOR_PROVIDER, profile_for
    from datasets import cases_dir as dataset_cases_dir, checked_case_hashes
    # The debate reaches a verdict in 7 turns; the solo agent needs more rounds
    # because one agent does every role's work in sequence.
    selected_arms = list(ARMS) if arm == "both" else [arm]
    if skip_existing and not trial:
        raise ValueError("--skip-existing requires --trial to identify the batch")
    trial = trial or new_trial_id()
    cases_dir = dataset_cases_dir(REPO_ROOT, dataset)
    if not model:
        model = DEFAULT_MODEL_FOR_PROVIDER.get(provider, "")
        if not model:
            print(f"pass --model for provider {provider}", file=sys.stderr)
            raise SystemExit(1)

    # Every per-model setting comes from one row, so switching model cannot
    # leave max_tokens or the timeout behind at another model's value.
    profile = profile_for(model)
    if profile.known and profile.provider != provider:
        raise ValueError(f"Model {model} is profiled for {profile.provider}, not {provider}")
    if not max_tokens:
        max_tokens = profile.max_tokens
    if not profile.known:
        print(f"note: no profile for {model}; using max_tokens={max_tokens}, "
              f"timeout={profile.timeout_s}s, and its cost will read as unknown. "
              f"Add a row to models.PROFILES to fix both.", file=sys.stderr)

    names = [c.strip() for c in cases.split(",") if c.strip()] if cases else None
    if not names and not all:
        print("pass --cases a,b or --all", file=sys.stderr)
        raise SystemExit(1)
    if not names:
        names = sorted(d.name for d in cases_dir.iterdir()
                       if d.is_dir() and (d / "meta.json").exists())
    if len(names) != len(set(names)):
        raise ValueError("Duplicate case names")
    # Validate the entire batch before reserving traces or submitting any work.
    source_hashes = {name: checked_case_hashes(REPO_ROOT, dataset, name) for name in names}
    jobs = []
    for n in names:
        if not (cases_dir/n/"kernel.py").is_file():raise FileNotFoundError(n)
        for selected_arm in selected_arms:
            dest = trace_path(n, selected_arm, traces_dir=profile.traces_dir, trial=trial)
            if skip_existing and (dest/"verdict.json").exists():
                print(f"skip {n}/{selected_arm}/{trial}: completed trace")
                continue
            if dest.exists():raise FileExistsError(f"Use a new trial; refusing overwrite: {dest}")
            jobs.append((n, selected_arm, max_rounds or (4 if selected_arm == "debate" else 10),
                         model, max_tokens, provider, profile.timeout_s, dataset, trial, source_hashes[n],
                         total_output_tokens))
    for job in jobs:
        n, selected_arm, rounds = job[:3]
        reserve_trace(n,selected_arm,traces_dir=profile.traces_dir,trial=trial,metadata={
            "dataset":dataset,"provider":provider,"model":model,"max_tokens":max_tokens,
            "reasoning_effort":"low" if provider in {"fireworks", "openrouter"} else "default",
            "timeout_s":profile.timeout_s,
            "max_rounds":rounds,"raw_api_capture":provider == "fireworks",
            "total_output_token_budget":total_output_tokens or None,
            **source_hashes[n]})
    print(f"running {len(jobs)} tool trials: {provider}/{model}, dataset={dataset}, trial={trial}, max_tokens={max_tokens}",flush=True)
    # Submit each remote call once, preserving its job identity even when Modal
    # raises instead of returning a result. Unlike a fail-fast starmap iterator,
    # one remote or local persistence failure cannot discard other paid runs.
    done = 0
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one.remote, *job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            entry, selected_arm = job[:2]
            identity = f"{entry}/{selected_arm}"
            dest = trace_path(entry, selected_arm, traces_dir=profile.traces_dir, trial=trial)
            result = None
            failure_kind = "remote_call"
            done += 1
            try:
                result = future.result()
                failure_kind = "trace_write"
                if (result.get("entry"), result.get("arm")) != (entry, selected_arm):
                    raise ValueError(f"Remote result identity does not match {identity}")
                trace_dir = write_trace(
                    entry, selected_arm, traces_dir=profile.traces_dir, trial=trial,
                    tar=result.get("tar"),
                    files={"runner_stdout.txt": result.get("stdout") or "",
                           "runner_error.txt": result.get("error") or ""},
                    metadata={"status": "completed" if result["ok"] else "error"},
                )
            except Exception as exc:
                failures.append(identity)
                try:
                    detail = _retain_failed_result(dest, exc, kind=failure_kind, result=result)
                    diagnostic = detail["message"]
                except Exception as save_error:
                    # Keep draining every future even if the local disk itself
                    # is failing. The failure still produces a nonzero exit.
                    diagnostic = f"{type(exc).__name__}; failure record could not be saved: {type(save_error).__name__}"
                print(f"  [{done}/{len(jobs)}] {identity}: ERROR ({failure_kind})", flush=True)
                print(f"      trace: {dest}\n      error: {diagnostic[:800]}", flush=True)
                continue
            verdict = result.get("verdict") or {}
            status = f"{verdict.get('verdict')} conf={verdict.get('confidence')}" if result["ok"] else "ERROR"
            print(f"  [{done}/{len(jobs)}] {identity}: {status}", flush=True)
            print(f"      trace: {trace_dir}", flush=True)
            if not result["ok"]:
                failures.append(identity)
                print(f"      error: {str(result.get('error'))[:800]}", flush=True)

    print(f"\n{done}/{len(jobs)} complete, {len(failures)} failed"
          + (f": {', '.join(failures)}" if failures else ""))
    print("score it with: python benchmark_fn_fp/eval_scripts/summarize_traces.py")
    if failures:
        raise SystemExit(1)
