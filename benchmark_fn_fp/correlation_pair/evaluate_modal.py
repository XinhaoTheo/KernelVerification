"""GPU oracle validation and isolated tool-enabled evaluations of the pair."""
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
app = modal.App("kv-correlation-pair-eval")
base_image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("torch==2.8.0", "triton==3.4.0", "numpy==1.26.4"))
image = base_image.add_local_dir(str(ROOT / "eval_cases"), "/root/cases")
agent_image = (base_image.pip_install("anthropic", "openai", "python-dotenv")
               .add_local_dir(str(REPO / "verifier"), "/root/verifier")
               .add_local_dir(str(ROOT / "eval_cases"), "/root/cases"))


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


@app.function(image=agent_image, gpu="T4", timeout=3600, max_containers=2,
              secrets=[modal.Secret.from_dotenv(REPO)])
def run_agent(name, arm, provider="anthropic", model="claude-opus-5"):
    import contextlib
    import io
    import os
    import tarfile
    import traceback
    os.chdir("/root")
    sys.path.insert(0,"/root")
    os.environ.update(AGENTIC_MODEL=model,AGENTIC_PROVIDER=provider,
                      AGENTIC_PROBE_SANDBOX="off",AGENTIC_LLM_TIMEOUT_SECONDS="1800")
    from verifier.agentic_run import main
    agents="solo" if arm=="solo" else "describer,skeptic,experimenter,judge"
    from uuid import uuid4
    run=Path("/root/trace_runs")/uuid4().hex/name/arm
    os.environ["AGENTIC_LLM_TRACE_DIR"]=str(run/"llm_calls")
    argv=[name,"--dataset-dir","/root/cases","--run-dir",str(run),"--agents",agents,
          "--max-debate-rounds","10" if arm=="solo" else "4",
          "--model",model,"--provider",provider,"--max-tokens","32768"]
    buf=io.StringIO()
    error=None
    try:
        with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
            main(argv)
    except Exception:
        error=traceback.format_exc()
    blob=io.BytesIO()
    if run.exists():
        with tarfile.open(fileobj=blob,mode="w:gz") as tar:
            tar.add(run,arcname=".")
    verdict=json.loads((run/"verdict.json").read_text()) if (run/"verdict.json").exists() else None
    return {"tar":blob.getvalue(),"verdict":verdict,"stdout":buf.getvalue(),"error":error}


@app.local_entrypoint()
def main(action: str="validate", cases: str="case_36,case_37", trial: str="r1",
         provider: str="anthropic", model: str="claude-opus-5"):
    names=cases.split(",")
    if action=="validate":
        result=validate.remote(names)
        cpu=json.loads((ROOT/"answer_key_cpu.json").read_text())
        for name,r in result["cases"].items():
            assert r["input_sha256"]==cpu["cases"][name]["input_sha256"]
            assert r["ground_truth"]==cpu["cases"][name]["cpu_ground_truth"]
        (ROOT/"validation_gpu.json").write_text(json.dumps(result,indent=2))
        for name,r in result["cases"].items():print(name,r["ground_truth"],r["errors"][0])
    elif action in ("solo","debate"):
        sys.path.insert(0,str(REPO/"benchmark_fn_fp/eval_scripts"))
        from models import profile_for
        from traces import reserve_trace, write_trace
        profile=profile_for(model)
        for name in names:
            dest=reserve_trace(name,action,traces_dir=profile.traces_dir,trial=trial,metadata={
                "provider":provider,"model":model,"dataset":"correlation_pair","raw_api_capture":provider=="fireworks"})
            (dest/"request.json").write_text(json.dumps({"case":name,"arm":action,
                "provider":provider,"model":model,"max_tokens":32768,
                "max_debate_rounds":10 if action=="solo" else 4},indent=2))
            r=run_agent.remote(name,action,provider,model)
            write_trace(name,action,traces_dir=profile.traces_dir,trial=trial,tar=r["tar"],
                files={"runner_stdout.txt":r["stdout"],"runner_error.txt":r["error"] or ""},
                metadata={"status":"error" if r["error"] else "completed"})
            print(name,action,json.dumps(r["verdict"]))
            print("trace:",dest)
    else:raise ValueError(action)
