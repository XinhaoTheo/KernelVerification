"""Run benchmark_fn_fp/triton/<case>/test.py on a real Modal GPU.

These cases wrap REAL, currently-maintained production Triton kernels
(NSA, vLLM, AutoGPTQ, Liger-Kernel, flash-attention, mamba, ...), so they
must actually execute on CUDA hardware -- unlike benchmark_fn_fp/'s pure
PyTorch CPU cases, these cannot be faked on CPU.

Usage:
    modal run benchmark_fn_fp/triton/modal_runner.py --cases fn1_foo,fp2_bar
    modal run benchmark_fn_fp/triton/modal_runner.py --all
"""
from __future__ import annotations

import pathlib
import sys

import modal

APP_NAME = "kv-fn-fp-triton-benchmark"
CASES_DIR = pathlib.Path(__file__).parent

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "numpy")
)


# max_containers caps how many T4s this fan-out can hold at once. Without it
# a --all run requests one container per case and saturates the workspace
# GPU quota (the 14-case run pinned 10 T4s and tripped Modal's limit).
@app.function(image=image, gpu="T4", timeout=600, max_containers=3)
def run_case(name: str, kernel_code: str, test_code: str) -> dict:
    import io
    import contextlib
    import importlib.util
    import sys as _sys
    import tempfile
    import traceback
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "kernel.py").write_text(kernel_code)
        (tmp_path / "test.py").write_text(test_code)
        _sys.path.insert(0, str(tmp_path))
        _sys.modules.pop("kernel", None)
        _sys.modules.pop("test_mod", None)

        buf = io.StringIO()
        result = {"name": name, "ok": False, "returned": None, "error": None}
        try:
            spec = importlib.util.spec_from_file_location("test_mod", tmp_path / "test.py")
            mod = importlib.util.module_from_spec(spec)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                spec.loader.exec_module(mod)
                returned = mod.test_kernel()
            result["ok"] = True
            result["returned"] = bool(returned)
        except Exception:
            result["error"] = traceback.format_exc()
        finally:
            _sys.path.remove(str(tmp_path))

        result["stdout"] = buf.getvalue()
        return result


def _iter_case_dirs(names: list[str] | None):
    if names:
        for n in names:
            d = CASES_DIR / n
            if not d.is_dir():
                raise SystemExit(f"no such case dir: {d}")
            yield d
    else:
        for d in sorted(CASES_DIR.iterdir()):
            if d.is_dir() and (d / "kernel.py").exists() and (d / "test.py").exists():
                yield d


@app.local_entrypoint()
def main(cases: str = "", all: bool = False):
    names = [c.strip() for c in cases.split(",") if c.strip()] if cases else None
    if not names and not all:
        print("pass --cases a,b,c or --all", file=sys.stderr)
        raise SystemExit(1)

    dirs = list(_iter_case_dirs(names))
    jobs = []
    for d in dirs:
        kernel_code = (d / "kernel.py").read_text()
        test_code = (d / "test.py").read_text()
        jobs.append((d.name, kernel_code, test_code))

    print(f"running {len(jobs)} case(s) on Modal GPU...")
    results = list(run_case.starmap(jobs))

    n_ok = 0
    for r in results:
        status = "OK" if r["ok"] and r["returned"] else ("ERROR" if not r["ok"] else "DEMONSTRATION-FAILED")
        print(f"\n=== {r['name']}: {status} ===")
        if r["stdout"].strip():
            print(r["stdout"].strip())
        if r["error"]:
            print(r["error"])
        if r["ok"] and r["returned"]:
            n_ok += 1

    print(f"\n{n_ok}/{len(results)} cases demonstrated their claimed FN/FP behavior on real GPU hardware.")
