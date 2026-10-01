"""Private GPU oracle for frozen numerical challenges; no LLM is called.

Only this oracle image mounts families and CPU answer keys. The common agent
runner mounts eval_cases alone, so probes cannot import these references.

Local public inputs live in ../triton_eval_cases; private output lives in
single_call_vs_tools_challenges/private_data. This explicit Modal entry point may run T4
validation; importing report modules never invokes it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[2] / "single_call_vs_tools_challenges"
app = modal.App("kv-numerical-challenges-oracle")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "triton==3.4.0", "numpy==1.26.4")
         .add_local_dir(str(ROOT.parent / "triton_eval_cases"), "/root/challenges/eval_cases")
         .add_local_dir(str(Path(__file__).resolve().parent / "families"), "/root/challenges/families"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def family_registry(root: Path) -> dict:
    """Register a case once and fail closed on an ambiguous oracle."""
    registry = {}
    for source in sorted((root / "families").glob("*.py")):
        if source.name.startswith("_"):
            continue
        family = load_module(source, f"numerical_oracle_{source.stem}")
        for name in family.CASES:
            if name in registry:
                raise ValueError(f"Multiple families declare {name}")
            registry[name] = (source.stem, family)
    return registry


def cpu_answers(root: Path) -> dict:
    cases = {}
    for path in sorted((root / "private_data").glob("answer_key_*.json")):
        for name, row in json.loads(path.read_text())["cases"].items():
            if name in cases:
                raise ValueError(f"Multiple CPU labels for {name}")
            cases[name] = row
    return cases


def check_cpu_agreement(result: dict, cpu: dict) -> None:
    """Do not publish a freeze if sources, inputs or independently set labels differ."""
    for name, row in result["cases"].items():
        expected = cpu[name]
        for key in ("input_sha256", "kernel_sha256", "problem_sha256"):
            if row[key] != expected[key]:
                raise ValueError(f"CPU/GPU mismatch: {name}/{key}")
        if row["ground_truth"] != expected["cpu_ground_truth"]:
            raise ValueError(f"CPU/GPU label mismatch: {name}")


@app.function(image=image, gpu="T4", timeout=1200)
def validate(names: list[str]) -> dict:
    import hashlib
    import platform
    import numpy as np
    import torch
    import triton

    root = Path("/root/challenges")
    registry = family_registry(root)
    results = {}

    def input_hashes(inputs):
        return [hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for t in inputs]

    for name in names:
        family_name, family = registry[name]
        source = root / "eval_cases" / name / "kernel.py"
        kernel = load_module(source, f"numerical_kernel_{name}")
        inputs = kernel.make_inputs()
        before_hashes = input_hashes(inputs)
        if input_hashes(kernel.make_inputs()) != before_hashes:
            raise ValueError(f"make_inputs is not deterministic: {name}")
        arrays = tuple(t.detach().cpu().numpy().copy() for t in inputs)
        reference = np.asarray(family.reference(arrays), dtype=np.float64)
        independent = np.asarray(family.independent_reference(arrays), dtype=np.float64)
        if reference.shape != independent.shape or not np.isfinite(reference).all():
            raise ValueError(f"Invalid reference: {name}")
        if not np.allclose(reference, independent, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Independent references disagree: {name}")
        emulated = np.asarray(family.emulate(arrays), dtype=np.float64)
        cpu_error = float(family.metric(emulated, reference))
        errors, output_hashes = [], []
        for _ in range(10):
            output = kernel.run(*inputs)
            torch.cuda.synchronize()
            output = output.detach().cpu().numpy()
            if output.shape != reference.shape or not np.isfinite(output).all():
                raise ValueError(f"Invalid kernel output: {name}")
            if input_hashes(inputs) != before_hashes:
                raise ValueError(f"Kernel mutated an input: {name}")
            error = float(family.metric(output.astype(np.float64), reference))
            if not np.isfinite(error):
                raise ValueError(f"Non-finite error: {name}")
            errors.append(error)
            output_hashes.append(hashlib.sha256(output.tobytes()).hexdigest())
        budget = float(family.TOLERANCE)
        results[name] = {
            "family": family_name, "errors": errors, "budget": budget,
            "ground_truth": "trust" if max(errors) <= budget else "reject",
            "input_sha256": before_hashes,
            "output_sha256": output_hashes, "bitwise_repeatable": len(set(output_hashes)) == 1,
            "kernel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "problem_sha256": hashlib.sha256(source.with_name("problem.txt").read_bytes()).hexdigest(),
            "oracle_source_sha256": hashlib.sha256((root / "families" / f"{family_name}.py").read_bytes()).hexdigest(),
            "oracle_max_abs_disagreement": float(np.max(np.abs(reference - independent))),
            "cpu_emulated_error": cpu_error,
            "gpu_cpu_error_difference": max(abs(error - cpu_error) for error in errors),
            "input_nonmutation_checked": True,
        }
    return {"environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "triton": triton.__version__, "numpy": np.__version__, "python": platform.python_version()},
            "repetitions": 10, "cases": results}


@app.local_entrypoint()
def main(cases: str = ""):
    cpu = cpu_answers(ROOT)
    names = [name.strip() for name in cases.split(",") if name.strip()] if cases else sorted(cpu)
    if not names or len(names) != len(set(names)):
        raise ValueError("Select at least one case, without duplicates")
    import hashlib
    for name in names:
        expected = cpu[name]
        for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt")):
            actual = hashlib.sha256((ROOT.parent / "triton_eval_cases" / name / filename).read_bytes()).hexdigest()
            if expected[f"{kind}_sha256"] != actual:
                raise ValueError(f"CPU source changed: {name}/{filename}")
    result = validate.remote(names)
    check_cpu_agreement(result, cpu)
    dest = ROOT / "private_data" / "validation_gpu.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Validating a later family adds its frozen rows without deleting earlier
    # validated families. Each row retains its own environment for audit.
    previous = json.loads(dest.read_text()) if dest.exists() else {"cases": {}}
    for row in result["cases"].values():
        row["environment"] = result["environment"]
    previous.update({key: value for key, value in result.items() if key != "cases"})
    previous["cases"].update(result["cases"])
    temporary = dest.with_suffix(".tmp")
    temporary.write_text(json.dumps(previous, indent=2) + "\n")
    temporary.replace(dest)
    for name, row in result["cases"].items():
        print(f"{name}: {row['ground_truth']}, error={max(row['errors']):.9g}, "
              f"budget={row['budget']:.9g}, repeats=10")
    print(f"wrote {dest}")
