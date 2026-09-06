"""Run benchmark_fn_fp/<case>/test.py on a real Modal GPU (Triton needs CUDA).

Usage:
    modal run benchmark_fn_fp/modal_runner.py --cases fn1_foo,fp2_bar
    modal run benchmark_fn_fp/modal_runner.py --all

Each case directory must contain kernel.py and test.py (test.py must define
`test_kernel()` that returns a truthy value on success, per the convention
used by the rest of dataset/). Case source is read locally and shipped to the
remote function as plain strings, so no Modal mount/volume setup is needed.
"""
from __future__ import annotations

import pathlib
import sys

import modal

APP_NAME = "kv-fn-fp-benchmark"
CASES_DIR = pathlib.Path(__file__).parent

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "numpy")
)


@app.function(image=image, gpu="T4", timeout=300)
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
        # Each Modal container may serve multiple calls in the same process;
        # clear any previously imported "kernel"/"test_mod" so this call's
        # own kernel.py/test.py are actually the ones loaded, not a stale
        # cached module from an earlier case in the same batch.
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
