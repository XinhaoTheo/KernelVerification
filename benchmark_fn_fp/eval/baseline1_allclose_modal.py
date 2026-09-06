"""Baseline 1: what a conventional `allclose` test concludes.

This is the target the whole FN/FP benchmark was built against. It is not an LLM:
each case's test.py computes the boolean a normal CI test would produce -- random
inputs, a standard tolerance, one run -- and prints it as NAIVE_ALLCLOSE_VERDICT.
That boolean maps to the shared verdict vocabulary:

    allclose passed -> the test reports the kernel is fine   -> "trust"
    allclose failed -> the test reports the kernel is broken -> "reject"

By construction this baseline should be wrong on every case: an FN case is one
where a loose tolerance passes a genuinely buggy kernel, and an FP case is one
where a strict tolerance fails a correct one. Measuring it anyway matters --
the claim is only worth as much as the run behind it.

Which variable inside each test.py counts as "the conventional check" is a
judgement; it is recorded next to the inserted print in each test.py so it can
be argued with rather than taken on faith.

Usage (from repo root):
    modal run benchmark_fn_fp/eval/baseline1_allclose_modal.py --all
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = REPO_ROOT / "benchmark_fn_fp" / "triton"

app = modal.App("kv-fn-fp-allclose-baseline")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "triton", "numpy")

VERDICT_RE = re.compile(r"NAIVE_ALLCLOSE_VERDICT:\s*(True|False)")


@app.function(image=image, gpu="T4", timeout=900, max_containers=3)
def run_case(name: str, kernel_code: str, test_code: str) -> dict:
    import contextlib
    import importlib.util
    import io
    import sys as _sys
    import tempfile
    import traceback
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "kernel.py").write_text(kernel_code)
        (tmp_path / "test.py").write_text(test_code)
        _sys.path.insert(0, str(tmp_path))
        for stale in ("kernel", "test_mod"):
            _sys.modules.pop(stale, None)

        buf = io.StringIO()
        result: dict = {"name": name, "ok": False, "error": None}
        try:
            spec = importlib.util.spec_from_file_location("test_mod", tmp_path / "test.py")
            mod = importlib.util.module_from_spec(spec)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                spec.loader.exec_module(mod)
                mod.test_kernel()
            result["ok"] = True
        except Exception:
            result["error"] = traceback.format_exc()[-3000:]
        result["stdout"] = buf.getvalue()[-6000:]
        return result


@app.local_entrypoint()
def main(cases: str = "", all: bool = False,
         out: str = "benchmark_fn_fp/eval/results_baseline1.json"):
    sys.path.insert(0, str(REPO_ROOT / "benchmark_fn_fp" / "eval"))
    from common import REJECT, TRUST, NO_VERDICT, load_cases, print_report  # noqa: E402

    all_cases = load_cases()
    if cases:
        wanted = {c.strip() for c in cases.split(",") if c.strip()}
        all_cases = [c for c in all_cases if c.name in wanted or c.real_name in wanted]
    elif not all:
        print("pass --cases a,b or --all", file=sys.stderr)
        raise SystemExit(1)

    jobs = [
        (c.real_name,
         (SOURCE_DIR / c.real_name / "kernel.py").read_text(),
         (SOURCE_DIR / c.real_name / "test.py").read_text())
        for c in all_cases
    ]
    print(f"running the conventional allclose test on {len(jobs)} case(s)")
    outputs = list(run_case.starmap(jobs))

    by_real = {c.real_name: c for c in all_cases}
    results: dict[str, str] = {}
    details: dict[str, dict] = {}
    for raw in outputs:
        case = by_real[raw["name"]]
        match = VERDICT_RE.search(raw.get("stdout", "") or "")
        if match is None:
            # No printed verdict means the harness failed, not that the test
            # reached a conclusion; those are scored separately.
            verdict = NO_VERDICT
            naive_passed = None
        else:
            naive_passed = match.group(1) == "True"
            verdict = TRUST if naive_passed else REJECT
        results[case.name] = verdict
        details[case.name] = {
            "real_name": case.real_name,
            "group": case.group,
            "seed_class": case.seed_class,
            "naive_allclose_passed": naive_passed,
            "verdict": verdict,
            "ground_truth": case.ground_truth,
            "correct": verdict == case.ground_truth,
            "error": raw.get("error"),
            "stdout": raw.get("stdout"),
        }
        if raw.get("error"):
            print(f"  {case.name} ({case.real_name}) ERROR:\n{raw['error'][:800]}")

    print_report("Baseline 1: conventional allclose test", results, all_cases)

    out_path = REPO_ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "baseline": "conventional_allclose",
        "results": results,
        "cases": details,
    }, indent=2))
    print(f"  wrote {out_path}")
