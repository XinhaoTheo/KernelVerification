"""Run the numerical-check baseline on every active registered case.

Two predeclared cohorts are retained: original test.py conventional checks and
modern public make_inputs(seed=0) / reference / run checks. These have different
comparison rules, so reports show each cohort as well as the aggregate. No model
API or private qualification inputs are used. A passing sampled check is not a
proof of correctness over the full input domain.

    modal run benchmark_fn_fp/eval_scripts/baseline1_allclose_modal.py --all
    # Resume an interrupted batch without rerunning saved cases:
    modal run benchmark_fn_fp/eval_scripts/baseline1_allclose_modal.py --all --resume
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys
import time
import traceback

import modal

HERE = pathlib.Path(__file__).resolve().parent
# Modal imports this module as /root/baseline1_allclose_modal.py. Only the
# local entrypoint uses repository paths; the worker receives source payloads.
REPO_ROOT = HERE.parents[1] if modal.is_local() else pathlib.Path("/root")
BENCHMARK = REPO_ROOT / "benchmark_fn_fp"
sys.path.insert(0, str(HERE))
from baseline_allclose_protocol import (  # noqa: E402
    PROTOCOL_VERSION, SEED, comparison_profile, generator_defaults, sha256,
)

app = modal.App("kv-fn-fp-allclose-baseline")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "triton==3.4.0", "numpy==1.26.4"))
if modal.is_local():
    image = image.add_local_file(HERE / "baseline_allclose_protocol.py", "/root/baseline_allclose_protocol.py")


@app.function(image=image, gpu="T4", cpu=2, memory=8192, timeout=900, max_containers=3)
def run_case(payload: dict) -> dict:
    from baseline_allclose_protocol import execute_payload
    return execute_payload(payload)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def payload_hash(payload: dict) -> str:
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def load_jobs(cases: str = "", benchmark: pathlib.Path = BENCHMARK) -> list[dict]:
    from case_registry import case_public_dir, case_sort_key, is_active_case, load_registry
    registry = load_registry(benchmark)
    wanted = {name.strip() for name in cases.split(",") if name.strip()}
    jobs = []
    for case, detail in sorted(registry["case_details"].items(), key=lambda item: case_sort_key(item[0])):
        if not is_active_case(detail) or (wanted and case not in wanted):
            continue
        public = case_public_dir(benchmark, case, detail)
        source_paths = {"problem.txt": public / "problem.txt", "kernel.py": public / "kernel.py"}
        payload = {"case": case, "dataset": detail["dataset"], "seed": SEED,
                   "protocol_version": PROTOCOL_VERSION}
        if detail["dataset"] == "benchmark_fn_fp":
            answer = benchmark / detail.get("answer_dir", "triton/" + registry["cases"][case])
            source_paths.update({"kernel.py": answer / "kernel.py", "test.py": answer / "test.py"})
            payload.update(mode="legacy_naive", comparison={"kind": "legacy_test_marker"},
                           generator={"global_rng_seed": SEED, "note": "test.py may set its own seeds and shapes; the exact source is saved"})
        elif detail["dataset"] == "real_kernel_challenges":
            payload.update(mode="public_default", comparison=comparison_profile(detail),
                           generator=generator_defaults((public / "kernel.py").read_text()))
        else:
            raise ValueError(f"No allclose protocol for active dataset {detail['dataset']!r}")
        payload["source_paths"] = {name: str(path.relative_to(benchmark)) for name, path in source_paths.items()}
        payload["sources"] = {name: path.read_text() for name, path in source_paths.items()}
        payload["source_hashes"] = {name: sha256(source.encode()) for name, source in payload["sources"].items()}
        jobs.append(payload)
    missing = wanted - {row["case"] for row in jobs}
    if missing:
        raise ValueError(f"Unknown or archived cases: {sorted(missing)}")
    if not jobs:
        raise ValueError("No active cases selected")
    return jobs


def prepare_run(jobs: list[dict], trial: str, root: pathlib.Path, resume: bool = False) -> dict:
    if not trial.startswith("r") or not trial[1:].isdigit() or int(trial[1:]) < 1:
        raise ValueError("trial must be r followed by a positive integer")
    manifest_path = root / f"manifest_{trial}.json"
    inventory = [{key: value for key, value in job.items() if key != "sources"} |
                 {"payload_sha256": payload_hash(job)} for job in jobs]
    harness = {name: (HERE / name).read_bytes() for name in
               ("baseline1_allclose_modal.py", "baseline_allclose_protocol.py")}
    harness_hashes = {name: sha256(source) for name, source in harness.items()}
    protocol = {
        "version": PROTOCOL_VERSION, "seed": SEED, "modern_calls_per_case": 1,
        "modern_input_selection": "Public make_inputs(device='cuda', seed=0), all other defaults unchanged",
        "modern_comparison": "Public reference evaluated before run; public numerical tolerances or exact equality",
        "legacy_comparison": "Execute the unchanged original test.py; read NAIVE_ALLCLOSE_VERDICT only after successful completion",
        "coverage": "Numerical outputs only. No exhaustive values, shape sweep, private qualification inputs, storage/alias audit or model calls.",
        "reference_limit": "The supplied public reference is a numerical comparison implementation. For example, erf reference returns rounded FP32 values; this check is not a mathematical proof.",
        "failure_policy": "Any exception or missing legacy verdict is no_verdict, not a kernel rejection. No automatic retries.",
        "environment": {"gpu": "NVIDIA T4", "torch": "2.8.0", "triton": "3.4.0", "numpy": "1.26.4", "python": "3.11"},
        "maximum_containers": 3, "requested_cpus_per_container": 2, "requested_memory_mib_per_container": 8192,
        "model_api_calls": 0,
    }
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; choose a new trial or --resume")
        saved = json.loads(manifest_path.read_text())
        if (saved["protocol"] != protocol or saved["cases"] != inventory
                or saved.get("harness_sha256") != harness_hashes):
            raise ValueError("Prepared sources or protocol changed; a saved batch cannot silently change")
        return saved
    manifest = {"trial": trial, "prepared_at": utc_now(), "protocol": protocol,
                "harness_sha256": harness_hashes, "cases": inventory}
    # Persist the frozen protocol and executable sources before the first remote call.
    write_json(manifest_path, manifest)
    harness_directory = root / f"protocol_{trial}"
    harness_directory.mkdir()
    for name, source in harness.items():
        (harness_directory / name).write_bytes(source)
    for job in jobs:
        directory = root / job["case"] / trial
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "request.json", {key: value for key, value in job.items() if key != "sources"})
        for name, source in job["sources"].items():
            (directory / name).write_text(source)
    return manifest


def score_saved(jobs: list[dict], trial: str, root: pathlib.Path, benchmark: pathlib.Path = BENCHMARK) -> dict:
    from summarize_traces import build_report
    labels = build_report(benchmark_dir=benchmark)["ground_truth_by_dataset"]
    rows = []
    for job in jobs:
        path = root / job["case"] / trial / "result.json"
        result = json.loads(path.read_text()) if path.exists() else {"verdict": "no_verdict", "phase": "not_returned"}
        truth = labels[job["dataset"]][job["case"]]
        rows.append({"case": job["case"], "mode": job["mode"], "verdict": result["verdict"], "ground_truth": truth,
                     "correct": result["verdict"] == truth, "phase": result["phase"],
                     "elapsed_seconds": result.get("elapsed_seconds"),
                     "trace": str(path.relative_to(benchmark))})
    def counts(subset):
        return {"total": len(subset), "correct": sum(row["correct"] for row in subset),
                "wrong": sum(not row["correct"] and row["verdict"] != "no_verdict" for row in subset),
                "no_verdict": sum(row["verdict"] == "no_verdict" for row in subset),
                "accuracy": sum(row["correct"] for row in subset) / len(subset) if subset else None}
    return {"trial": trial, "updated_at": utc_now(), "baseline": PROTOCOL_VERSION,
            "overall": counts(rows), "cohorts": {mode: counts([row for row in rows if row["mode"] == mode])
                for mode in sorted({row["mode"] for row in rows})},
            "sum_case_elapsed_seconds": sum(row["elapsed_seconds"] or 0 for row in rows),
            "model_api_usd": 0,
            "cost_note": "Per-case elapsed time excludes image builds, container startup/idle time and billing rounding. It is not an invoice.",
            "cases": rows}


@app.local_entrypoint()
def main(cases: str = "", all: bool = False, trial: str = "r1", resume: bool = False, prepare_only: bool = False):
    if not all and not cases:
        raise ValueError("Pass --all or --cases case_01,case_02")
    jobs = load_jobs(cases)
    root = BENCHMARK / "traces_allclose"
    prepare_run(jobs, trial, root, resume)
    if prepare_only:
        print(f"Prepared {len(jobs)} cases; no GPU calls made. Manifest: {root / f'manifest_{trial}.json'}", flush=True)
        return
    pending = [job for job in jobs if not (root / job["case"] / trial / "result.json").exists()]
    print(f"Running {len(pending)} cases; {len(jobs)-len(pending)} saved cases retained; no model API calls.", flush=True)
    start = time.perf_counter()
    # Ordered collection identifies infrastructure exceptions without guessing a case.
    for job, returned in zip(pending, run_case.map(pending, return_exceptions=True)):
        if isinstance(returned, BaseException):
            result = {"case": job["case"], "verdict": "no_verdict", "passed": None, "phase": "modal_call",
                      "error": "".join(traceback.format_exception(returned)), "stdout": "", "elapsed_seconds": None}
        else:
            result = returned
        directory = root / job["case"] / trial
        (directory / "stdout.txt").write_text(result.pop("stdout", ""))
        result["recorded_at"] = utc_now()
        result["payload_sha256"] = payload_hash(job)
        write_json(directory / "result.json", result)
        print(f"{job['case']}: {result['verdict']} ({result['phase']})", flush=True)
    summary = score_saved(jobs, trial, root)
    summary["latest_collection_wall_seconds"] = time.perf_counter() - start
    write_json(root / f"summary_{trial}.json", summary)
    print(json.dumps(summary["overall"], indent=2), flush=True)
    print(f"Saved {root / f'summary_{trial}.json'}", flush=True)
