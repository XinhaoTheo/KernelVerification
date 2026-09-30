"""Private T4 oracle and pre-payment freeze for evidence-audit workloads."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("kv-evidence-challenges-oracle")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "triton==3.4.0", "numpy==1.26.4")
         .add_local_dir(str(ROOT / "eval_cases"), "/root/evidence/eval_cases")
         .add_local_dir(str(ROOT / "families"), "/root/evidence/families"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hashes(path):
    return {f"{kind}_sha256": hashlib.sha256((path / filename).read_bytes()).hexdigest()
            for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt"))}


def cpu_answers():
    rows = {}
    for source in sorted((ROOT / "private_data").glob("answer_key_*.json")):
        for name, row in json.loads(source.read_text())["cases"].items():
            if name in rows:
                raise ValueError(f"Duplicate CPU label: {name}")
            rows[name] = row
    return rows


@app.function(image=image, gpu="T4", timeout=1200)
def validate(names):
    import platform
    import torch
    import triton
    import numpy as np
    root = Path("/root/evidence")
    registry = {}
    for source in sorted((root / "families").glob("*.py")):
        if source.name.startswith("_"):
            continue
        family = load_module(source, f"evidence_oracle_{source.stem}")
        for name in family.CASES:
            if name in registry:
                raise ValueError(f"Duplicate family registration: {name}")
            registry[name] = (source, family)
    result = {"cases": {}, "failures": {}, "environment": {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "triton": triton.__version__, "numpy": np.__version__,
        "python": platform.python_version()}}
    for name in names:
        try:
            source, family = registry[name]
            directory = root / "eval_cases" / name
            kernel = load_module(directory / "kernel.py", f"evidence_kernel_{name}")
            row = family.validate_case(name, kernel)
            if row["ground_truth"] not in {"trust", "reject"}:
                raise ValueError("No binary oracle label")
            if "initial_probe" not in row or "input_sha256" not in row:
                raise ValueError("Missing initial evidence or input hashes")
            row.update(hashes(directory))
            row["family"] = source.stem
            row["oracle_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            row["environment"] = result["environment"]
            result["cases"][name] = row
        except Exception as exc:
            import traceback
            result["failures"][name] = {"type": type(exc).__name__,
                                        "message": str(exc), "traceback": traceback.format_exc()}
    return result


@app.local_entrypoint()
def main(cases: str = "", freeze: bool = False):
    cpu = cpu_answers()
    names = [name.strip() for name in cases.split(",") if name.strip()] or sorted(cpu)
    if not names or len(names) != len(set(names)):
        raise ValueError("Choose unique cases")
    for name in names:
        for key, value in hashes(ROOT / "eval_cases" / name).items():
            if cpu[name][key] != value:
                raise ValueError(f"CPU/public source mismatch: {name}/{key}")
    result = validate.remote(names)
    result["recorded_at"] = datetime.now(timezone.utc).isoformat()
    logdir = ROOT / "private_data" / "validation_attempts"
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = logdir / f"{stamp}_{'freeze' if freeze else 'preview'}.json"
    log.write_text(json.dumps(result, indent=2) + "\n")
    if result["failures"]:
        print(json.dumps(result["failures"], indent=2))
        raise ValueError(f"GPU validation failed; full attempt retained at {log}")
    for name, row in result["cases"].items():
        expected = cpu[name]
        for key in ("kernel_sha256", "problem_sha256", "input_sha256"):
            if row[key] != expected[key]:
                raise ValueError(f"CPU/GPU mismatch: {name}/{key}; retained {log}")
        if row["ground_truth"] != expected["cpu_ground_truth"]:
            raise ValueError(f"CPU/GPU label mismatch: {name}; retained {log}")
        print(f"{name}: {row['ground_truth']}; initial_probe={json.dumps(row['initial_probe'])}")
    if freeze:
        dest = ROOT / "private_data" / "validation_gpu.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(dest.read_text()) if dest.exists() else {"cases": {}}
        overlap = previous["cases"].keys() & result["cases"].keys()
        if overlap:
            raise ValueError(f"Refusing to overwrite frozen cases: {sorted(overlap)}")
        previous["cases"].update(result["cases"])
        previous["updated_at"] = result["recorded_at"]
        dest.write_text(json.dumps(previous, indent=2) + "\n")
        print(f"Frozen: {dest}")
    print(f"Validation attempt: {log}")
