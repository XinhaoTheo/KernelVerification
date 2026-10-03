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
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import modal

_IS_LOCAL = modal.is_local()
# Modal can reconstruct this module as /root/run_agentic_modal.py. Host repo
# ancestors, local source mounts and dotenv paths only exist on the submitter.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if _IS_LOCAL else pathlib.Path("/root")
# The answer-free copy: no test.py, no verdict in meta.json, opaque case ids.
CASES_DIR = REPO_ROOT / "benchmark_fn_fp" / "triton_eval_cases" if _IS_LOCAL else pathlib.Path("/root/cases")
_REMOTE_CASE_ROOT = "/root/cases"
_REMOTE_TRACE_ROOT = "/root/trace_runs"
_PUBLIC_FILES = frozenset({"kernel.py", "problem.txt", "meta.json", "LICENSE"})
_NEUTRAL_META_STATUSES = ("under_test", "unverified", "unknown")
_NEUTRAL_META_FAMILIES = ("sequence_audit", "joint_audit", "mutation_audit", "box_relu_audit")

# The one line that actually differs between arms.
ARMS = {
    "solo": "solo",
    "debate": "describer,skeptic,experimenter,judge",
}

app = modal.App("kv-fn-fp-agentic-eval")

def _make_image(numpy_version: str):
    worker_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.8.0", "triton==3.4.0", f"numpy=={numpy_version}",
                     "anthropic", "openai", "python-dotenv")
    )
    if _IS_LOCAL:
        worker_image = worker_image.add_local_dir(str(REPO_ROOT / "verifier"), "/root/verifier")
    return worker_image


image = _make_image("1.26.4")


def _public_files_for_case(case_dir: pathlib.Path) -> dict[str, bytes]:
    """Only the public problem bundle crosses into an agent worker."""
    files = {}
    for name in sorted(_PUBLIC_FILES):
        path = case_dir / name
        if path.is_symlink():
            raise ValueError(f"Public input must not be a symlink: {path}")
        if path.is_file():
            files[name] = path.read_bytes()
    _validate_public_payload(case_dir.name, files)
    return files


def _validate_public_payload(entry: str, public_files: dict | None) -> None:
    if not re.fullmatch(r"case_[A-Za-z0-9_]+", entry):
        raise ValueError(f"Invalid case directory name: {entry!r}")
    if not isinstance(public_files, dict):
        raise ValueError("An explicit per-case public_files payload is required")
    if not {"kernel.py", "problem.txt", "meta.json"} <= set(public_files) <= _PUBLIC_FILES:
        raise ValueError("Public bundle must contain kernel.py, problem.txt and neutral meta.json only, with optional LICENSE")
    if any(not isinstance(value, bytes) for value in public_files.values()):
        raise ValueError("Public bundle contents must be bytes")
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Agent-visible meta.json must not contain duplicate fields")
            result[key] = value
        return result

    meta = json.loads(public_files["meta.json"], object_pairs_hook=unique_fields)
    if meta == {}:
        return
    # Historical public bundles use an explicit unknown result. Preserve their
    # original bytes, but never admit a result, a label, or arbitrary metadata.
    required = {"name", "status", "passed"}
    neutral = (isinstance(meta, dict)
               and required <= set(meta) <= required | {"family"}
               and meta["name"] == entry
               and meta["passed"] is None
               and meta["status"] in _NEUTRAL_META_STATUSES
               and ("family" not in meta or meta["family"] in _NEUTRAL_META_FAMILIES))
    if not neutral:
        raise ValueError("Agent-visible meta.json must be empty or use the strict neutral legacy schema")


def _materialize_public_case(entry: str, public_files: dict | None,
                             expected_hashes: dict | None = None) -> dict:
    """Reset reused workers; expose only this case and no previous arm traces."""
    import hashlib
    import shutil

    _validate_public_payload(entry, public_files)
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in public_files.items()}
    if expected_hashes is not None:
        for kind, name in (("kernel", "kernel.py"), ("problem", "problem.txt")):
            if digests[name] != expected_hashes[f"{kind}_sha256"]:
                raise ValueError(f"Public payload differs from checked source: {entry}/{name}")
    for root in (pathlib.Path(_REMOTE_CASE_ROOT), pathlib.Path(_REMOTE_TRACE_ROOT)):
        if root.is_symlink() or root.is_file():
            root.unlink()
        elif root.exists():
            shutil.rmtree(root)
    dest = pathlib.Path(_REMOTE_CASE_ROOT) / entry
    dest.mkdir(parents=True)
    for name, data in public_files.items():
        (dest / name).write_bytes(data)
    visible_cases = sorted(path.name for path in pathlib.Path(_REMOTE_CASE_ROOT).iterdir())
    if visible_cases != [entry]:
        raise RuntimeError(f"Unexpected visible cases: {visible_cases}")
    return {"visible_cases": visible_cases, "public_files_sha256": digests,
            "previous_trace_root_removed": not pathlib.Path(_REMOTE_TRACE_ROOT).exists()}


def _run_one_impl(entry: str, arm: str, max_rounds: int, model: str, max_tokens: int,
                  provider: str, timeout_s: int, dataset: str = "benchmark_fn_fp", trial: str = "legacy",
                  expected_hashes: dict | None = None, total_output_tokens: int = 0,
                  debate_budget_closeout: bool = False, require_claim_scope_fields: bool = False,
                  public_files: dict | None = None) -> dict:
    """Run one case under one arm inside this GPU container."""
    import contextlib
    import io
    import os
    import tarfile
    from pathlib import Path

    os.chdir("/root")
    sys.path.insert(0, "/root")
    visibility = _materialize_public_case(entry, public_files, expected_hashes)
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
    if debate_budget_closeout and arm == "debate":
        os.environ["AGENTIC_DEBATE_BUDGET_CLOSEOUT"] = "1"
    else:
        os.environ.pop("AGENTIC_DEBATE_BUDGET_CLOSEOUT", None)
    if require_claim_scope_fields:
        os.environ["AGENTIC_REQUIRE_CLAIM_SCOPE_FIELDS"] = "1"
    else:
        os.environ.pop("AGENTIC_REQUIRE_CLAIM_SCOPE_FIELDS", None)
    # Explicit experiment setting, also captured in each request. Observed
    # latency differences alone do not establish a cause for provider failures.
    if provider in {"fireworks", "openrouter"}:
        os.environ["AGENTIC_OPENAI_REASONING_EFFORT"] = "low"

    from verifier.agentic_run import main as agentic_main
    from verifier.agentic.llm_trace import _exception_details
    from verifier.agentic.provenance import runtime_fingerprint

    agents = ARMS[arm]
    if dataset not in {"benchmark_fn_fp", "correlation_pair", "numerical_challenges",
                       "evidence_challenges", "real_kernel_challenges"}:
        raise ValueError(f"Unsupported source dataset: {dataset}")
    dataset_dir = _REMOTE_CASE_ROOT
    run_dir = Path(_REMOTE_TRACE_ROOT) / trial / dataset / entry / arm
    if run_dir.exists():
        raise FileExistsError(f"Remote trial already exists: {run_dir}")
    os.environ["AGENTIC_LLM_TRACE_DIR"] = str(run_dir / "llm_calls")
    os.environ["AGENTIC_LLM_TRACE_PROGRESS"] = "1"
    run_dir.mkdir(parents=True)
    (run_dir / "artifact_visibility.json").write_text(json.dumps(visibility, indent=2))
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
        # The actual GPU environment, not the local submitter's environment.
        (run_dir / "runtime.json").write_text(json.dumps(runtime_fingerprint(capture_gpu=True), indent=2))
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


def _worker_options(worker_image):
    # Modal hydrates explicit dependencies positionally in the container.
    # Keep one Secret before the Image on both sides; the remote placeholder
    # receives the submitted secret's object id without reading a host .env.
    secret = modal.Secret.from_dotenv(REPO_ROOT) if _IS_LOCAL else modal.Secret.from_dict({})
    return dict(image=worker_image, gpu="T4", timeout=5400, max_containers=4,
                single_use_containers=True,
                secrets=[secret])


@app.function(**_worker_options(image))
def preflight_remote(entries: list[str], expected_hashes: dict,
                     public_files: dict | None = None) -> dict:
    """Exercise the same container/dependencies without any model calls or traces."""
    import os
    import torch
    from pathlib import Path

    sys.path.insert(0, "/root")
    from verifier.agentic.provenance import runtime_fingerprint

    if not os.environ.get("FIREWORKS_API_KEY"):
        raise RuntimeError("FIREWORKS_API_KEY is missing in the worker environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the worker environment")
    checked = {}
    for entry in entries:
        visibility = _materialize_public_case(entry, (public_files or {}).get(entry), expected_hashes[entry])
        checked[entry] = {**expected_hashes[entry], **visibility}
        # The existing artifact loader requires neutral meta.json too.
        from verifier.dataset import load_entry
        load_entry(entry, dataset_dir=Path(_REMOTE_CASE_ROOT))
    return {"fireworks_key_present": True, "cases": checked,
            "runtime": runtime_fingerprint(capture_gpu=True), "model_calls": 0}


@app.function(**_worker_options(image))
def run_one(entry: str, arm: str, max_rounds: int, model: str, max_tokens: int,
            provider: str, timeout_s: int, dataset: str = "benchmark_fn_fp", trial: str = "legacy",
            expected_hashes: dict | None = None, total_output_tokens: int = 0,
            debate_budget_closeout: bool = False, require_claim_scope_fields: bool = False,
            public_files: dict | None = None) -> dict:
    if dataset == "numerical_pilot":
        raise ValueError("numerical_pilot is archived; excluded from new experiments")
    return _run_one_impl(entry, arm, max_rounds, model, max_tokens, provider, timeout_s,
                         dataset, trial, expected_hashes, total_output_tokens, debate_budget_closeout,
                         require_claim_scope_fields, public_files)


def existing_valid_slots(benchmark_dir: pathlib.Path, traces_dir: str, dataset: str) -> set[tuple[str, str]]:
    """Reuse a completed judgment across trials/providers, including wrong answers.

    This is a coverage check, not a same-configuration accuracy comparison.
    Legacy runs may lack status/raw-call fields; known failures never fill a slot.
    """
    from traces import iter_trace_records
    from summarize_traces import build_report
    from datasets import dataset_members

    benchmark = pathlib.Path(benchmark_dir).resolve()
    target = benchmark / traces_dir
    records = [record for record in iter_trace_records(benchmark_dir=benchmark)
               if record["dataset"] in dataset_members(dataset)
               and pathlib.Path(record["path"]).resolve().is_relative_to(target)]
    metadata = {str(pathlib.Path(record["path"]).resolve()): record["metadata"] for record in records}
    report = build_report(records=records, benchmark_dir=benchmark, labels={})
    slots = set()
    for group in report["arms"].values():
        for row in group["per_case"].values():
            status = metadata[str((benchmark / row["path"]).resolve())].get("status")
            if status not in {None, "completed", "historical"} or row.get("runner_error"):
                continue
            if row.get("verdict") in {"trust", "reject", "needs_more_evidence"} and row["outcome"] != "token_limit":
                slots.add((row["case"], row["arm"]))
    return slots


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
         total_output_tokens: int = 0, only_missing: bool = False,
         preflight_only: bool = False, debate_budget_closeout: bool = False,
         require_claim_scope_fields: bool = False):
    if total_output_tokens < 0:
        raise ValueError("total_output_tokens must be nonnegative")
    if debate_budget_closeout and total_output_tokens <= 8192:
        raise ValueError("Budget closeout requires a total output budget greater than 8192")
    if preflight_only and not arm:
        arm = "both"
    if arm not in (*ARMS, "both"):
        print(f"--arm must be one of {sorted(ARMS)} or both", file=sys.stderr)
        raise SystemExit(1)

    # Imported here, not at module scope: Modal imports this module inside the
    # container too, where only /root/verifier and /root/cases are mounted. A
    # module-level import of a sibling in this directory crashes every container
    # at startup -- which it did, and the run hung for hours retrying.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from traces import write_trace, trace_path, reserve_trace, new_trial_id, next_trial_id, experiment_trial_id
    from models import DEFAULT_MODEL_FOR_PROVIDER, profile_for
    from datasets import (case_dataset, case_names, cases_dir as dataset_cases_dir,
                          checked_case_hashes, ensure_active_dataset)
    from verifier.agentic.provenance import file_sha256, verifier_sha256
    ensure_active_dataset(dataset)
    # The debate reaches a verdict in 7 turns; the solo agent needs more rounds
    # because one agent does every role's work in sequence.
    selected_arms = list(ARMS) if arm == "both" else [arm]
    if skip_existing and not trial:
        raise ValueError("--skip-existing requires --trial to identify the batch")
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
        names = case_names(REPO_ROOT, dataset)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate case names")
    original_trial = experiment_trial_id(trial)
    trial = trial or (next_trial_id(names, selected_arms, traces_dir=profile.traces_dir)
                      if profile.traces_dir == "traces_glm" else new_trial_id())
    if skip_existing and re.fullmatch(r"r[1-9][0-9]*", trial):
        # rN is local to each case/arm, so matching directory names can belong
        # to different experiments. A continuation must keep one existing batch
        # identity, rather than silently splitting it or combining unrelated runs.
        batches = set()
        for name in names:
            for selected_arm in selected_arms:
                existing = trace_path(name, selected_arm, traces_dir=profile.traces_dir, trial=trial)
                if existing.exists():
                    metadata = json.loads((existing / "trace_meta.json").read_text())
                    batch = metadata.get("original_trial") or metadata.get("trial") or trial
                    batches.add(batch)
        if len(batches) > 1:
            raise ValueError(f"Cannot resume {trial}: selected traces belong to multiple experiment batches")
        if batches:
            original_trial = batches.pop()
    # Validate the entire batch before reserving traces or submitting any work.
    source_datasets = {name: case_dataset(REPO_ROOT, dataset, name, require_active=True) for name in names}
    source_hashes = {name: checked_case_hashes(REPO_ROOT, dataset, name) for name in names}
    public_payloads = {name: _public_files_for_case(cases_dir / name) for name in names}
    if preflight_only:
        print(json.dumps(preflight_remote.remote(names, source_hashes, public_payloads), indent=2), flush=True)
        return
    filled = (existing_valid_slots(REPO_ROOT / "benchmark_fn_fp", profile.traces_dir, dataset)
              if only_missing else set())
    jobs = []
    for n in names:
        if not (cases_dir/n/"kernel.py").is_file():raise FileNotFoundError(n)
        for selected_arm in selected_arms:
            if (n, selected_arm) in filled:
                print(f"skip {n}/{selected_arm}: existing valid judgment in {profile.traces_dir}")
                continue
            dest = trace_path(n, selected_arm, traces_dir=profile.traces_dir, trial=trial)
            if skip_existing and (dest/"verdict.json").exists():
                print(f"skip {n}/{selected_arm}/{trial}: completed trace")
                continue
            if dest.exists():raise FileExistsError(f"Use a new trial; refusing overwrite: {dest}")
            jobs.append((n, selected_arm, max_rounds or (4 if selected_arm == "debate" else 10),
                         model, max_tokens, provider, profile.timeout_s, source_datasets[n], trial, source_hashes[n],
                         total_output_tokens, debate_budget_closeout, require_claim_scope_fields, public_payloads[n]))
    provenance = {"verifier_sha256": verifier_sha256(), "runner_sha256": file_sha256(__file__),
                  "case_visibility": "current_case_only"}
    for job in jobs:
        n, selected_arm, rounds = job[:3]
        reserve_trace(n,selected_arm,traces_dir=profile.traces_dir,trial=trial,metadata={
            "dataset":job[7],"provider":provider,"model":model,"max_tokens":max_tokens,
            "reasoning_effort":"low" if provider in {"fireworks", "openrouter"} else "default",
            "timeout_s":profile.timeout_s,
            "max_rounds":rounds,"raw_api_capture":provider == "fireworks",
            "original_trial":original_trial,
            "total_output_token_budget":total_output_tokens or None,
            "debate_budget_closeout":bool(debate_budget_closeout and selected_arm == "debate"),
            "require_claim_scope_fields":require_claim_scope_fields,
            "normal_turn_output_cap":min(max_tokens, 8192) if debate_budget_closeout and selected_arm == "debate" else max_tokens,
            "closeout_reserve":8192 if debate_budget_closeout and selected_arm == "debate" else 0,
            "closeout_stage_caps":{"evidence":4096,"skeptic":1024,"judge":3072}
                if debate_budget_closeout and selected_arm == "debate" else None,
            "public_input_files":sorted(public_payloads[n]),
            **source_hashes[n], **provenance})
    print(f"running {len(jobs)} tool trials: {provider}/{model}, dataset={dataset}, trial={trial}, max_tokens={max_tokens}",flush=True)
    # Submit each remote call once, preserving its job identity even when Modal
    # raises instead of returning a result. Unlike a fail-fast starmap iterator,
    # one remote or local persistence failure cannot discard other paid runs.
    done = 0
    failures = []
    worker = run_one
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(worker.remote, *job): job for job in jobs}
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
