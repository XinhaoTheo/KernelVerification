"""Validate the frozen pair on Modal T4; model trials use the shared runners."""
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[3] / "single_call_vs_tools_challenges"
PRIVATE = Path("private_data") / "correlation_pair"
app = modal.App("kv-correlation-pair-eval")
base_image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("torch==2.8.0", "triton==3.4.0", "numpy==1.26.4"))
image = base_image.add_local_dir(str(ROOT.parent / "triton_eval_cases"), "/root/cases")


@app.function(image=image, gpu="T4", timeout=600)
def validate(names):
    import hashlib
    import importlib.util
    import math
    import numpy as np
    import torch
    import triton
    results = {}
    for name in names:
        path = Path("/root/cases") / name / "kernel.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        inputs = module.make_inputs()
        x, a, b = [t.cpu().double() for t in inputs]
        ref = a @ x + b @ x
        independent = torch.tensor([
            math.fsum(float(a[i,j]) * float(x[j]) for j in range(len(x))) +
            math.fsum(float(b[i,j]) * float(x[j]) for j in range(len(x)))
            for i in range(len(a))], dtype=torch.float64)
        assert torch.max(torch.abs(ref - independent)).item() < 1e-12
        errors = []
        for _ in range(10):
            y = module.run(*inputs).cpu().double()
            assert torch.isfinite(y).all()
            errors.append((torch.linalg.vector_norm(y-ref) / max(torch.linalg.vector_norm(ref).item(), .008)).item())
        results[name] = {"errors":errors,"ground_truth":"trust" if max(errors)<=.1 else "reject",
            "input_sha256":[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in inputs],
            "kernel_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "problem_sha256":hashlib.sha256(path.with_name("problem.txt").read_bytes()).hexdigest(),
            "oracle_max_abs_disagreement":torch.max(torch.abs(ref-independent)).item()}
    return {"environment":{"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
                           "triton":triton.__version__,"numpy":np.__version__},"cases":results}

@app.local_entrypoint()
def main(cases: str = "case_36,case_37"):
    names = [name.strip() for name in cases.split(",") if name.strip()]
    private = ROOT / PRIVATE
    cpu = json.loads((private / "answer_key_cpu.json").read_text())
    if not names or any(name not in cpu["cases"] for name in names):
        raise ValueError("Pass validated case_36 and/or case_37 names")
    result = validate.remote(names)
    for name, row in result["cases"].items():
        assert row["input_sha256"] == cpu["cases"][name]["input_sha256"]
        assert row["ground_truth"] == cpu["cases"][name]["cpu_ground_truth"]
    (private / "validation_gpu.json").write_text(json.dumps(result, indent=2))
    for name, row in result["cases"].items():
        print(name, row["ground_truth"], row["errors"][0])
